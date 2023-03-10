from typing import List, Dict, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url
from torchvision.models.detection import fasterrcnn_resnet50_fpn, retinanet_resnet50_fpn, RetinaNet_ResNet50_FPN_Weights
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.rpn import RegionProposalNetwork, RPNHead
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, fasterrcnn_resnet50_fpn_v2, _default_anchorgen, \
    TwoMLPHead
from collections import OrderedDict
import torchvision.models as models
import torchvision.ops as ops
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import FeaturePyramidNetwork


class FasterRCNN_FPN(nn.Module):
    """
    A Faster R-CNN FPN inspired parking lot detector.
    Passes the whole image through a CNN -->
    pools ROIs from the feature pyramid --> passes
    each ROI separately through regression and classification heads.
    This is an extended version of a Faster R-CNN for rotated
    boxes with the shape of (x1, y1, x2, y2, x3, y3, x4, y4)
    """

    def __init__(self, num_classes=2, num_outputs=8):
        super(FasterRCNN_FPN, self).__init__()

        # Load the pre-trained FPN backbone
        self.backbone = resnet_fpn_backbone('resnet50', pretrained=True)

        # Load the pre-trained Resnet 50 + FPN backbone
        # self.backbone = models.resnet50(pretrained=True)
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[256, 512, 1024, 2048],
            out_channels=256
        )
        # out_channels = self.backbone.out_channels
        out_channels = 256

        # Add the RPN network
        self.rpn = create_RPN(out_channels)

        # Add the ROI pooling layer
        self.roi_pooling = ops.MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'], output_size=7, sampling_ratio=2)
        resolution = self.roi_pooling.output_size[0]

        # Add 2 fully connected layers on top of the ROI pooling layer
        self.box_head = TwoMLPHead(out_channels * resolution ** 2, 1024)

        # Add the FastRCNN box predictor (classification and regression heads
        self.box_predictor = FastRCNNPredictor(1024, num_classes)

        self.transform = GeneralizedRCNNTransform(min_size=800, max_size=1333,
                                                  image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225])

    def forward(self, images, targets):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        """
        Args:
            images (list[Tensor]): images to be processed
            targets (list[Dict[str, Tensor]]): ground-truth boxes present in the image (optional)
        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).
        """

        if self.training:
            if targets is None:
                torch._assert(False, "targets should not be none when in training mode")
            else:
                for target in targets:
                    boxes = target["boxes"]
                    if isinstance(boxes, torch.Tensor):
                        torch._assert(
                            len(boxes.shape) == 2 and boxes.shape[-1] == 8,
                            f"Expected target boxes to be a tensor of shape [N, 8], got {boxes.shape}.",
                        )
                    else:
                        torch._assert(False, f"Expected target boxes to be of type Tensor, got {type(boxes)}.")

        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            torch._assert(
                len(val) == 2,
                f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
            )
            original_image_sizes.append((val[0], val[1]))

        # images, targets = self.transform(images, targets)
        print("original_image_sizes : ", original_image_sizes)
        images = ImageList(images, original_image_sizes)

        # Extract features from the backbone
        features = self.backbone(images.tensors)
        # print("features : ", features)

        # Add a batch dimension to the features dictionary
        # for key in features:
        #     features[key] = features[key].unsqueeze(0)

        # fpn_features = self.fpn(features)
        # print("fpn_features : ", fpn_features)

        # Extract region proposals per image
        proposals, proposal_losses = self.rpn(images, features, targets)
        # Generate features for each ROI
        box_features = self.roi_pooling(features, proposals, image_shapes=original_image_sizes)

        # Pass the ROI features through the fully connected layers
        box_features = self.box_head(box_features)

        # Pass the ROI features through the classification and regression heads
        class_logits, box_regression = self.box_predictor(box_features)

        labels = [t['labels'] for t in targets]
        regression_targets = [t['boxes'] for t in targets]

        detections: List[Dict[str, torch.Tensor]] = []
        detector_losses = {}
        if self.training:
            if labels is None:
                raise ValueError("labels cannot be None")
            if regression_targets is None:
                raise ValueError("regression_targets cannot be None")
            loss_classifier, loss_box_reg = fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
            detector_losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}
        else:
            boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals,
                                                                original_image_sizes)  # TODO
            num_images = len(boxes)
            for i in range(num_images):
                detections.append(
                    {
                        "boxes": boxes[i],
                        "labels": labels[i],
                        "scores": scores[i],
                    }
                )
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        return losses, detections


class FastRCNNPredictor(nn.Module):
    """
    Standard classification + bounding box regression layers
    for Fast R-CNN.
    Args:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
    """

    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 8)

    def forward(self, x):
        # if x.dim() == 4:
        #     torch._assert(
        #         list(x.shape[2:]) == [1, 1],
        #         f"x has the wrong shape, expecting the last two dimensions to be [1,1] instead of {list(x.shape[2:])}",
        #     )
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas


def create_RPN(in_channels):
    # RPN parameters
    rpn_anchor_generator = _default_anchorgen()
    rpn_head = RPNHead(in_channels, rpn_anchor_generator.num_anchors_per_location()[0])
    rpn_pre_nms_top_n_train = 2000
    rpn_pre_nms_top_n_test = 1000
    rpn_post_nms_top_n_train = 2000
    rpn_post_nms_top_n_test = 1000
    rpn_nms_thresh = 0.7
    rpn_fg_iou_thresh = 0.7
    rpn_bg_iou_thresh = 0.3
    rpn_batch_size_per_image = 256
    rpn_positive_fraction = 0.5
    rpn_score_thresh = 0.0

    rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
    rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)

    return RegionProposalNetwork(
        rpn_anchor_generator,
        rpn_head,
        rpn_fg_iou_thresh,
        rpn_bg_iou_thresh,
        rpn_batch_size_per_image,
        rpn_positive_fraction,
        rpn_pre_nms_top_n,
        rpn_post_nms_top_n,
        rpn_nms_thresh,
        score_thresh=rpn_score_thresh,
    )


def postprocess_detections(
        self,
        class_logits,  # type: Tensor
        box_regression,  # type: Tensor
        proposals,  # type: List[Tensor]
        image_shapes,  # type: List[Tuple[int, int]]
):
    # type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]
    device = class_logits.device
    num_classes = class_logits.shape[-1]
    box_ops = ops.boxes

    boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
    pred_boxes = self.box_coder.decode(box_regression, proposals)

    pred_scores = F.softmax(class_logits, -1)

    pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
    pred_scores_list = pred_scores.split(boxes_per_image, 0)

    all_boxes = []
    all_scores = []
    all_labels = []
    for boxes, scores, image_shape in zip(pred_boxes_list, pred_scores_list, image_shapes):
        boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

        # create labels for each prediction
        labels = torch.arange(num_classes, device=device)
        labels = labels.view(1, -1).expand_as(scores)

        # remove predictions with the background label
        boxes = boxes[:, 1:]
        scores = scores[:, 1:]
        labels = labels[:, 1:]

        # batch everything, by making every class prediction be a separate instance
        boxes = boxes.reshape(-1, 4)
        scores = scores.reshape(-1)
        labels = labels.reshape(-1)

        # remove low scoring boxes
        inds = torch.where(scores > self.score_thresh)[0]
        boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

        # remove empty boxes
        keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

        # non-maximum suppression, independently done per class
        keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
        # keep only topk scoring predictions
        keep = keep[: self.detections_per_img]
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

        all_boxes.append(boxes)
        all_scores.append(scores)
        all_labels.append(labels)

    return all_boxes, all_scores, all_labels


# class FasterRCNN_FPN(nn.Module):

#     """
#     A Faster R-CNN FPN inspired parking lot classifier.
#     Passes the whole image through a CNN -->
#     pools ROIs from the feature pyramid --> passes
#     each ROI separately through a classification head.
#     """
#     def __init__(self, roi_res=7, pooling_type='square'):
#         super().__init__()
#
#         # backbone
#         # by default, uses frozen batchnorm and 3 trainable layers
#         self.backbone = resnet_fpn_backbone('resnet50', pretrained=True)
#         hidden_dim = 256
#
#         # pooling
#         self.roi_res = roi_res
#         self.pooling_type = pooling_type
#
#         # classification head
#         in_channels = hidden_dim * self.roi_res**2
#         self.class_head = ClassificationHead(in_channels)
#
#         # load coco weights
#         # url taken from: https://github.com/pytorch/vision/blob/master/torchvision/models/detection/faster_rcnn.py
#         weights_url = 'https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth'
#         state_dict = load_state_dict_from_url(weights_url, progress=False)
#         self.load_state_dict(state_dict, strict=False)
#
#
#     def forward(self, image, rois):
#         # get backbone features
#         features = self.backbone(image[None])
#
#         # pool ROIs from features pyramid
#         features = list(features.values())
#         features = pooling.pool_FPN_features(features, rois, self.roi_res, self.pooling_type)
#
#         # pass pooled ROIs through classification head to get class logits
#         features = features.flatten(1)
#         class_logits = self.class_head(features)
#
#         return class_logits

def create_model():
    # load a model pre-trained on COCO
    model = fasterrcnn_resnet50_fpn(weights='DEFAULT')
    model = fasterrcnn_resnet50_fpn_v2(weights='DEFAULT')

    # replace the classifier with a new one, that has
    # num_classes which is user-defined
    num_classes = 2  # occupied + vacant

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained model head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def fastrcnn_loss(class_logits, box_regression, labels, regression_targets):
    # type: (Tensor, Tensor, List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]
    """
    Computes the loss for Faster R-CNN.
    Args:
        class_logits (Tensor)
        box_regression (Tensor)
        labels (list[BoxList])
        regression_targets (Tensor)
    Returns:
        classification_loss (Tensor)
        box_loss (Tensor)
    """

    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)

    classification_loss = F.cross_entropy(class_logits, labels)

    # get indices that correspond to the regression targets for
    # the corresponding ground truth labels, to be used with
    # advanced indexing
    sampled_pos_inds_subset = torch.where(labels > 0)[0]
    labels_pos = labels[sampled_pos_inds_subset]
    N, num_classes = class_logits.shape
    box_regression = box_regression.reshape(N, box_regression.size(-1) // 4, 4)

    box_loss = F.smooth_l1_loss(
        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        beta=1 / 9,
        reduction="sum",
    )
    box_loss = box_loss / labels.numel()

    return classification_loss, box_loss
