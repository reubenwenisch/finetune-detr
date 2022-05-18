import torch
from torchmetrics import Accuracy, AveragePrecision, Precision, Recall, F1Score, AUROC, Specificity, ConfusionMatrix
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import numpy as np
# simulate a classification problem
# preds = torch.randn(10, 5).softmax(dim=-1)
# target = torch.randint(5, (10,))

num_classes = 7

accuracy = Accuracy()
average_precision = AveragePrecision(num_classes=num_classes, average=None)
precision = Precision(average='macro', num_classes=num_classes)
recall = Recall(average='macro', num_classes=num_classes)
specificity = Specificity(average='macro', num_classes=num_classes)
f1 = F1Score(num_classes=num_classes)
auroc = AUROC(num_classes=num_classes)
confmat = ConfusionMatrix(num_classes=num_classes)
mAP_metric = MeanAveragePrecision()

def atom360_metrics(preds, target):
    acc = accuracy(preds, target)
    AP = average_precision(preds, target)
    precision = precision(preds, target)
    recall = recall(preds, target)
    specificity = specificity(preds, target)
    f1 = f1(preds, target)
    auroc = auroc(preds, target)
    confmat = confmat(preds, target)
    mAP_metric.update(preds, target)
    print(f"Metrics on batch {i} Accuracy :{acc}, Avg Precision : {AP}, Precision: {precision}, Recall: {recall}, Specificity: {specificity}, F1: {f1}, AUROC: {auroc}")

def calculate_iou(input_boxes, target_boxes, form = 'coco'):
    """_summary_

    Args:
        input_boxes (_type_): _description_
        target_boxes (_type_): _description_
        form (str, optional): 
                pascal_voc: min/max coordinates [x_min, y_min, x_max, y_max]
                coco: width/height instead of maxes [x_min, y_min, width, height]
                Defaults to 'coco'.
    Returns:
        _type_: IOU
    """
    if form == 'coco':
        gt = input_boxes.copy()
        pr = target_boxes.copy()

        gt[2] = gt[0] + gt[2]
        gt[3] = gt[1] + gt[3]
        pr[2] = pr[0] + pr[2]
        pr[3] = pr[1] + pr[3]
    # Calculate overlap area
    if (gt[0] > gt[2]) or (gt[1]> gt[3]):
        raise AssertionError("Ground Truth Bounding Box is not correct")
    if (pr[0] > pr[2]) or (pr[1]> pr[3]):
        raise AssertionError("Ground Truth Bounding Box is not correct")
    dx = min(gt[2], pr[2]) - max(gt[0], pr[0]) + 1
    if dx < 0:
        return 0.0
    dy = min(gt[3], pr[3]) - max(gt[1], pr[1]) + 1
    if dy < 0:
        return 0.0
    overlap_area = dx * dy
    # Calculate union area
    union_area = (
            (gt[2] - gt[0] + 1) * (gt[3] - gt[1] + 1) +
            (pr[2] - pr[0] + 1) * (pr[3] - pr[1] + 1) -
            overlap_area
    )
    return overlap_area / union_area

def get_single_image_results(gt_boxes, pred_boxes, iou_thr, gt_classes, pred_classes):
    """Calculates number of true_pos, false_pos, false_neg from single batch of boxes.
    Args:
        gt_boxes (list of list of floats): list of locations of ground truth
            objects as [xmin, ymin, xmax, ymax]
        pred_boxes (dict): dict of dicts of 'boxes' (formatted like `gt_boxes`)
            and 'scores'
        iou_thr (float): value of IoU to consider as threshold for a
            true prediction.
    Returns:
        dict: true positives (int), false positives (int), false negatives (int)
    """
    all_pred_indices = range(len(pred_boxes))
    all_gt_indices = range(len(gt_boxes))
    preds = []
    targets = []
    if len(all_pred_indices)==0:
        tp = 0
        fp = 0
        fn = len(gt_boxes)
        tn = 0
        targets = gt_classes
        return {'bbx_true_positive':tp, 'bbx_false_positive':fp, 'bbx_false_negative':fn, 'bbx_true_negative': tn, 'preds': preds, 'targets': targets}
    if len(all_gt_indices)==0:
        tp = 0
        fp = len(pred_boxes)
        fn = 0
        tn = 0
        preds = pred_classes
        return {'bbx_true_positive':tp, 'bbx_false_positive':fp, 'bbx_false_negative':fn, 'bbx_true_negative': tn, 'preds': preds, 'targets': targets}
    gt_idx_thr=[]
    pred_idx_thr=[]
    ious=[]
    for ipb, pred_box in enumerate(pred_boxes):
        for igb, gt_box in enumerate(gt_boxes):
            # iou= calc_iou(gt_box, pred_box)
            iou = calculate_iou(pred_box, gt_box, form = 'coco')
            if iou > iou_thr:
                gt_idx_thr.append(igb)
                pred_idx_thr.append(ipb)
                ious.append(iou)
    iou_sort = np.argsort(ious)[::1] # sort based on higher IOU
    if len(gt_boxes)==0 and len(pred_boxes)==0:  # Prediction at image level considering all boxes
        tp=0
        fp=0
        fn=0
        tn = 1
        return {'bbx_true_positive':tp, 'bbx_false_positive':fp, 'bbx_false_negative':fn, 'bbx_true_negative': tn, 'preds': preds, 'targets': targets}
    if len(iou_sort)==0: # no overlaps but bounding box is there
        tp = 0
        fp = len(pred_boxes)
        fn = len(gt_boxes)
        tn = 0
        return {'bbx_true_positive':tp, 'bbx_false_positive':fp, 'bbx_false_negative':fn, 'bbx_true_negative': tn}
    else:
        gt_match_idx=[]
        pred_match_idx=[]
        for idx in iou_sort: # Loop based on higher IOU
            gt_idx = gt_idx_thr[idx]
            pr_idx = pred_idx_thr[idx]
            # If the boxes are unmatched, add them to matches
            if(gt_idx not in gt_match_idx) and (pr_idx not in pred_match_idx):
                gt_match_idx.append(gt_idx)
                pred_match_idx.append(pr_idx)
        tp = len(gt_match_idx)
        fp = len(pred_boxes) - len(pred_match_idx)
        fn = len(gt_boxes) - len(gt_match_idx)
        tn = 0
    return {'bbx_true_positive': tp, 'bbx_false_positive': fp, 'bbx_false_negative': fn, 'bbx_true_negative': tn}


# metric on all batches using custom accumulation
acc = accuracy.compute()
# pprint(mAP_metric.compute())
print(f"Accuracy on all data: {acc}")
# Reseting internal state such that metric ready for new data
acc.reset()