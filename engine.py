# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable
import copy

import torch
from utils import utils
from utils import transforms
from utils.rotated_coco_eval import RotatedCocoEvaluator
# from utils import PanopticEvaluator
from utils.coco_utils import get_coco_api_from_dataset
from utils.data_prefetcher import data_prefetcher
from utils.visualize import plot_log_per_epoch


def train_one_epoch(model, optimizer, data_loader, resolution, device, epoch, print_freq=10, log_dir=None, scaler=None, max_norm=0):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)

    # prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    # samples, targets = prefetcher.next()

    # for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
    for images, targets in metric_logger.log_every(data_loader, print_freq, log_dir, header):
        # outputs = model(samples)
        #assert torch.isnan(outputs).sum() == 0, print(outputs)
        # augment data
        # images, targets = transforms.augment(images, targets)

        # preprocess image
        res_images, res_rois = transforms.preprocess(images, rois=[t["boxes"] for t in targets], device=device,
                                                     res=resolution)
        # update boxed according to the new resolution
        new_target = []
        for idx, target in enumerate(targets):
            if resolution is not None:
                target["boxes"] = torch.tensor(res_rois[idx])
            new_target.append({k: v.to(device) for k, v in target.items()})
        targets = new_target

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(res_images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        # loss_dict_reduced = utils.reduce_dict(loss_dict)
        # loss_dict_reduced_unscaled = {f'{k}_unscaled': v
        #                               for k, v in loss_dict_reduced.items()}
        # loss_dict_reduced_scaled = {k: v * weight_dict[k]
        #                             for k, v in loss_dict_reduced.items() if k in weight_dict}
        # losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        #
        # loss_value = losses_reduced_scaled.item()

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        #assert torch.isnan(model.mu).sum() == 0, print(model.mu)
        optimizer.step()
        #assert torch.isnan(model.mu).sum() == 0, print(model.mu)
        #assert torch.isnan(model.mu.grad).sum() == 0, print(model.mu.grad)
        # metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

        # samples, targets = prefetcher.next()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types

@torch.no_grad()
def evaluate(model, data_loader, resolution, log_dir, device):
    model.eval()
    cpu_device = torch.device("cpu")
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = _get_iou_types(model)
    coco_evaluator = None#CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]
    res_all = {}
    target_all = {}
    imgToAnns = {}
    panoptic_evaluator = None

    for images, targets in metric_logger.log_every(data_loader, 10, log_dir, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # preprocess image
        res_images, res_rois = transforms.preprocess(images, rois=[t["boxes"] for t in targets], device=device,
                                                     res=resolution)
        # update boxed according to the new resolution
        if resolution is not None:
            for idx, target in enumerate(targets):
                target["boxes"] = torch.tensor(res_rois[idx])

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        outputs = model(res_images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]

        # reduce losses over all GPUs for logging purposes
        # loss_dict_reduced = utils.reduce_dict(loss_dict)
        # loss_dict_reduced_scaled = {k: v * weight_dict[k]
        #                             for k, v in loss_dict_reduced.items() if k in weight_dict}
        # loss_dict_reduced_unscaled = {f'{k}_unscaled': v
        #                               for k, v in loss_dict_reduced.items()}
        # metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
        #                      **loss_dict_reduced_scaled,
        #                      **loss_dict_reduced_unscaled)
        # metric_logger.update(class_error=loss_dict_reduced['class_error'])

        rotated_target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        # results = postprocessors['bbox'](outputs, rotated_target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, outputs)}
        res_all = {**res_all,**res}
        for target in targets:
            target_tmp = {int(id):{**target,**{'area':float(target['area'][i]),'iscrowd':int(target['iscrowd'][i]),'id':int(id),'image_id':int(target['image_id'][0]),'category_id':int(target['labels'][i]),'bbox':(target['polys'][i]*torch.tensor([target['size'][1],target['size'][0],target['size'][1],target['size'][0],target['size'][1],target['size'][0],target['size'][1],target['size'][0]],device=target['size'].device)).tolist()}} for i,id in enumerate(target['ids'])}
            target_all = {**target_all,**target_tmp}
            imgToAnns = {**imgToAnns,**{int(target['image_id']):[{**target,**{'area':float(target['area'][i]),'iscrowd':int(target['iscrowd'][i]),'id':int(id),'image_id':int(target['image_id'][0]),'category_id':int(target['labels'][i]),'bbox':(target['polys'][i]*torch.tensor([target['size'][1],target['size'][0],target['size'][1],target['size'][0],target['size'][1],target['size'][0],target['size'][1],target['size'][0]],device=target['size'].device)).tolist()}} for i,id in enumerate(target['ids'])]}}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    avg_stats_str = f"Averaged stats: {metric_logger}"
    with open(f'{log_dir}/logs.txt', 'a', newline='\n', encoding='utf-8') as f:
        f.write(avg_stats_str + '\n')

    final_ds = get_coco_api_from_dataset(data_loader.dataset)
    final_ds.anns = target_all
    final_ds.imgToAnns = imgToAnns
    coco_evaluator = RotatedCocoEvaluator(final_ds, iou_types)
    coco_evaluator.update(res_all)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # CalculateMAP(copy.deepcopy(base_ds), res_all, target_all, iou_types)
    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()

    print("stats : ", stats)

    return stats, coco_evaluator


def CalculateMAP(base_ds, results, targets, iou_types):
    base_ds.anns = targets
    coco_evaluator = RotatedCocoEvaluator(base_ds,iou_types)
    coco_evaluator.update(results)
    pass

def train_model(model, train_ds, valid_ds, test_ds, model_dir, device, lr=1e-5, epochs=30, lr_decay=50, res=None,
                verbose=False):
    """
    Trains any model which takes (image, rois) and outputs class_logits.
    Expects dataset.pdosp.PDOSP utils.
    Uses cross-entropy loss.
    """
    # transfer model to device
    model = model.to(device)

    # construct an Adam optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_decay, gamma=0.1)


    # construct an SGD optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    # optimizer = torch.optim.SGD(params, lr=lr,
    #                             momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
    lr_scheduler = None
    # cosine lr shceduler
    # lr = 8e-5
    # plot losses
    # increase epochs
    model_dir = f'./{model_dir}/LR_{lr}_{optimizer.__class__.__name__}_{lr_scheduler.__class__.__name__}'

    # and a learning rate scheduler which decreases the learning rate by
    # 10x every 3 epochs
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
    #                                                step_size=3,
    #                                                gamma=0.1)

    losses, mAP_results = [], []
    # train
    for epoch in range(1, epochs + 1):
        # train for one epoch

        # if this is the first epoch
        if epoch == 1:
            # ensure (an empty) model dir exists
            # shutil.rmtree(model_dir, ignore_errors=True)
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            # os.makedirs(model_dir, exist_ok=False)

            create_logs_header(model_dir)

        print("*********** training step ***********")
        metric_logger = train_one_epoch(model, optimizer, train_ds, res, device, epoch, print_freq=10,
                                        log_dir=model_dir)
        if lr_scheduler is not None:
            lr_scheduler.step()
        epoch_losses = get_metric_epoch_losses(metric_logger)
        losses.append(epoch_losses)

        # save training losses
        save_metric_losses(f'{model_dir}/train_losses.csv', epoch_losses)

        # evaluate on the valid dataset
        with open(f'{model_dir}/logs.txt', 'a', newline='\n', encoding='utf-8') as f:
            f.write("*********** evaluation step ***********" + '\n')

        print("*********** evaluation step ***********")
        coco_eval = evaluate(model, valid_ds, res, model_dir, device)
        epoch_mAPs = coco_eval.get_mAP_results()[0]
        mAP_results.append(epoch_mAPs)

        # save mAP evaluation result
        save_evaluation_results(f'{model_dir}/evaluation_result.csv', epoch_mAPs)

        # save weights
        torch.save(model.state_dict(), f'{model_dir}/weights_last_epoch.pt')

    # Plot training losses
    plot_log_per_epoch(range(1, epochs + 1), [round(loss[0], 3) for loss in losses], "Losses")

    # Plot evaluation AP results [IoU=0.50:0.95]
    plot_log_per_epoch(range(1, epochs + 1), [round(mAP[1], 3) for mAP in mAP_results], "mAPs")

    with open(f'{model_dir}/logs.txt', 'a', newline='\n', encoding='utf-8') as f:
        f.write("*********** testing step ***********" + '\n')

    # test model on test dataset
    print("*********** testing step ***********")
    evaluate(model, test_ds, res, model_dir, device)

    # delete model from memory
    del model


def create_logs_header(model_dir):
    with open(f'{model_dir}/logs.txt', 'w', newline='\n', encoding='utf-8') as f:
        f.write("*********** training step ***********" + '\n')

    # create loss log header
    with open(f'{model_dir}/train_losses.csv', 'w', newline='\n', encoding='utf-8') as f:
        f.write('loss,loss_classifier,loss_box_reg,loss_objectness,loss_rpn_box_reg\n')

    # create mAP log header
    with open(f'{model_dir}/evaluation_result.csv', 'w', newline='\n', encoding='utf-8') as f:
        f.write('AP [IoU=0.50:0.95], AP [IoU=0.50]\n')


def get_metric_epoch_losses(metric_logger):
    return [round(float(str(epoch_loss).split(" ")[0]), 3) for epoch_loss in metric_logger.meters.values()][1:6]


def save_metric_losses(log_file, metric_loss):
    with open(log_file, 'a', newline='\n', encoding='utf-8') as f:
        f.write(
            f'{metric_loss[0]},{metric_loss[1]},{metric_loss[2]},{metric_loss[3]},{metric_loss[4]}\n')


def save_evaluation_results(log_file, mAPs):
    with open(log_file, 'a', newline='\n', encoding='utf-8') as f:
        f.write(f'{mAPs[0]:.3f},{mAPs[1]:.3f}\n')

