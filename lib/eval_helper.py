import os
import sys
import json
from scipy.spatial.kdtree import distance_matrix
import torch
import pickle
import argparse

import numpy as np

from tqdm import tqdm
from copy import deepcopy
from torch.utils.data import DataLoader
from numpy.linalg import inv
from scipy.optimize import linear_sum_assignment

sys.path.append(os.path.join(os.getcwd())) # HACK add the root folder

import lib.capeval.bleu.bleu as capblue
import lib.capeval.cider.cider as capcider
import lib.capeval.rouge.rouge as caprouge
import lib.capeval.meteor.meteor as capmeteor

from data.scannet.model_util_scannet import ScannetDatasetConfig
from lib.config import CONF
from lib.ap_helper import parse_predictions
from lib.loss_helper import get_loss
from utils.box_util import box3d_iou_batch_tensor, generalized_box3d_iou

# constants
DC = ScannetDatasetConfig()

SCANREFER_RAW = {
    "train": json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_train.json"))),
    "val": json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_val.json")))
}
REFERIT3D_RAW = {
    "train": json.load(open(os.path.join(CONF.PATH.DATA, "nr3d_train.json"))),
    "val": json.load(open(os.path.join(CONF.PATH.DATA, "nr3d_val.json")))
}

def get_organized(dataset_name, phase):
    if dataset_name == "ScanRefer":
        raw_data = SCANREFER_RAW[phase]
    elif dataset_name == "ReferIt3D":
        raw_data = REFERIT3D_RAW[phase]
    else:
        raise ValueError("Invalid dataset.")

    organized = {}
    for data in raw_data:
        scene_id = data["scene_id"]
        object_id = data["object_id"]

        if scene_id not in organized: organized[scene_id] = {}
        if object_id not in organized[scene_id]: organized[scene_id][object_id] = []

        organized[scene_id][object_id].append(data)

    return organized

def prepare_corpus(raw_data, candidates, special_tokens, max_len=CONF.EVAL.MAX_DES_LEN+2):
    # get involved scene IDs
    scene_list = []
    for key in candidates.keys():
        # scene_id, _, _ = key.split("|")
        scene_id, _ = key.split("|")
        if scene_id not in scene_list: scene_list.append(scene_id)

    corpus = {}
    for data in raw_data:
        scene_id = data["scene_id"]
        if scene_id not in scene_list: continue
        object_id = data["object_id"]
        object_name = data["object_name"]
        token = data["token"][:max_len]
        description = " ".join(token)

        # add start and end token
        description = "{} {} {}".format(special_tokens["bos_token"], description, special_tokens["eos_token"])

        # key = "{}|{}|{}".format(scene_id, object_id, object_name)
        key = "{}|{}".format(scene_id, object_id)

        if key not in corpus:
            corpus[key] = []

        corpus[key].append(description)

    return corpus

def filter_candidates(candidates, min_iou):
    new_candidates = {}
    for key, value in candidates.items():
        des, iou = value[0], value[1]
        if iou >= min_iou:
            new_candidates[key] = des

    return new_candidates

def check_candidates(corpus, candidates, special_tokens):
    placeholder = "{} {}".format(special_tokens["bos_token"], special_tokens["eos_token"])
    corpus_keys = list(corpus.keys())
    candidate_keys = list(candidates.keys())
    missing_keys = [key for key in corpus_keys if key not in candidate_keys]

    if len(missing_keys) != 0:
        for key in missing_keys:
            candidates[key] = [placeholder]

    return candidates

def organize_candidates(corpus, candidates):
    new_candidates = {}
    for key in corpus.keys():
        new_candidates[key] = candidates[key]

    return new_candidates

def construct_bbox_corners(center, box_size):
    sx, sy, sz = box_size
    x_corners = [sx/2, sx/2, -sx/2, -sx/2, sx/2, sx/2, -sx/2, -sx/2]
    y_corners = [sy/2, -sy/2, -sy/2, sy/2, sy/2, -sy/2, -sy/2, sy/2]
    z_corners = [sz/2, sz/2, sz/2, sz/2, -sz/2, -sz/2, -sz/2, -sz/2]
    corners_3d = np.vstack([x_corners, y_corners, z_corners])
    corners_3d[0,:] = corners_3d[0,:] + center[0];
    corners_3d[1,:] = corners_3d[1,:] + center[1];
    corners_3d[2,:] = corners_3d[2,:] + center[2];
    corners_3d = np.transpose(corners_3d)

    return corners_3d

def decode_detections(data_dict):
    pred_center = data_dict['center'].detach().cpu().numpy() # (B,K,3)
    pred_heading_class = torch.argmax(data_dict['heading_scores'], -1) # B,num_proposal
    pred_heading_residual = torch.gather(data_dict['heading_residuals'], 2, pred_heading_class.unsqueeze(-1)).detach().cpu().numpy() # B,num_proposal
    pred_heading_class = pred_heading_class.detach().cpu().numpy() # B,num_proposal
    pred_size_class = torch.argmax(data_dict['size_scores'], -1) # B,num_proposal
    pred_size_residual = torch.gather(data_dict['size_residuals'], 2, pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,3)) # B,num_proposal,1,3
    pred_size_class = pred_size_class.detach().cpu().numpy() # B,num_proposal
    pred_size_residual = pred_size_residual.squeeze(2).detach().cpu().numpy() # B,num_proposal,3

    batch_size, num_bbox, _ = pred_center.shape
    bbox_corners = []
    for batch_id in range(batch_size):
        batch_corners = []
        for bbox_id in range(num_bbox):
            pred_obb = DC.param2obb(pred_center[batch_id, bbox_id], pred_heading_class[batch_id, bbox_id], pred_heading_residual[batch_id, bbox_id],
                    pred_size_class[batch_id, bbox_id], pred_size_residual[batch_id, bbox_id])
            pred_bbox = construct_bbox_corners(pred_obb[0:3], pred_obb[3:6])
            batch_corners.append(pred_bbox)
        
        batch_corners = np.stack(batch_corners, axis=0)
        bbox_corners.append(batch_corners)

    bbox_corners = np.stack(bbox_corners, axis=0) # batch_size, num_proposals, 8, 3

    return bbox_corners

def decode_targets(data_dict):
    pred_center = data_dict['center_label'] # (B,MAX_NUM_OBJ,3)
    pred_heading_class = data_dict['heading_class_label'] # B,K2
    pred_heading_residual = data_dict['heading_residual_label'] # B,K2
    pred_size_class = data_dict['size_class_label'] # B,K2
    pred_size_residual = data_dict['size_residual_label'] # B,K2,3

    # assign
    pred_center = torch.gather(pred_center, 1, data_dict["object_assignment"].unsqueeze(2).repeat(1, 1, 3)).detach().cpu().numpy()
    pred_heading_class = torch.gather(pred_heading_class, 1, data_dict["object_assignment"]).detach().cpu().numpy()
    pred_heading_residual = torch.gather(pred_heading_residual, 1, data_dict["object_assignment"]).unsqueeze(-1).detach().cpu().numpy()
    pred_size_class = torch.gather(pred_size_class, 1, data_dict["object_assignment"]).detach().cpu().numpy()
    pred_size_residual = torch.gather(pred_size_residual, 1, data_dict["object_assignment"].unsqueeze(2).repeat(1, 1, 3)).detach().cpu().numpy()

    batch_size, num_bbox, _ = pred_center.shape
    bbox_corners = []
    for batch_id in range(batch_size):
        batch_corners = []
        for bbox_id in range(num_bbox):
            pred_obb = DC.param2obb(pred_center[batch_id, bbox_id], pred_heading_class[batch_id, bbox_id], pred_heading_residual[batch_id, bbox_id],
                    pred_size_class[batch_id, bbox_id], pred_size_residual[batch_id, bbox_id])
            pred_bbox = construct_bbox_corners(pred_obb[0:3], pred_obb[3:6])
            batch_corners.append(pred_bbox)
        
        batch_corners = np.stack(batch_corners, axis=0)
        bbox_corners.append(batch_corners)

    bbox_corners = np.stack(bbox_corners, axis=0) # batch_size, num_proposals, 8, 3

    return bbox_corners

def assign_dense_caption(pred_captions, pred_boxes, gt_boxes, gt_box_ids, gt_box_masks, gt_scene_list, idx2word, special_tokens, strategy="giou"):
    """assign the densely predicted captions to GT boxes

    Args:
        pred_captions (torch.Tensor): predicted captions for all boxes, shape: (B, K1, L)
        pred_boxes (torch.Tensor): predicted bounding boxes, shape: (B, K1, 8, 3)
        gt_boxes (torch.Tensor): GT bounding boxes, shape: (B, K2)
        gt_box_ids (torch.Tensor): GT bounding boxes object IDs, shape: (B, K2)
        gt_box_masks (torch.Tensor): GT bounding boxes masks in the batch, shape: (B, K2)
        gt_scene_list (list): scene list in the batch, length: B
        idx2word (dict): vocabulary dictionary of all words, idx -> str
        special_tokens (dict): vocabulary dictionary of special tokens, e.g. [SOS], [PAD], etc.
        strategy ("giou" or "center"): assignment strategy, default: "giou"

    Returns:
        Dict: dictionary of assigned boxes and captions
    """

    def box_assignment(pred_boxes, gt_boxes, gt_masks):
        """assign GT boxes to predicted boxes

        Args:
            pred_boxes (torch.Tensor): predicted boxes, shape: (B, K1, 8, 3)
            gt_boxes (torch.Tensor): GT boxes, shape: (B, K2, 8, 3)
        """

        batch_size, nprop, *_ = pred_boxes.shape
        _, ngt, *_ = gt_boxes.shape
        nactual_gt = gt_masks.sum(1)

        # assignment
        if strategy == "giou":
            # gious
            gious = generalized_box3d_iou(
                pred_boxes,
                gt_boxes,
                nactual_gt,
                rotated_boxes=False,
                needs_grad=False,
            ) # B, K1, K2

            # hungarian assignment
            final_cost = -gious.detach().cpu().numpy()
        elif strategy == "center":
            # center distance
            dist = torch.cdist(pred_boxes.mean(2).float(), gt_boxes.mean(2).float())

            # hungarian assignment
            final_cost = dist.detach().cpu().numpy()
        else:
            raise ValueError("invalid strategy.")

        assignments = []

        # assignments from GTs to proposals
        per_gt_prop_inds = torch.zeros(
            [batch_size, ngt], dtype=torch.int64, device=pred_boxes.device
        )
        gt_matched_mask = torch.zeros(
            [batch_size, ngt], dtype=torch.float32, device=pred_boxes.device
        )

        for b in range(batch_size):
            assign = []
            if nactual_gt[b] > 0:
                assign = linear_sum_assignment(final_cost[b, :, : nactual_gt[b]])
                assign = [
                    torch.from_numpy(x).long().to(device=pred_boxes.device)
                    for x in assign
                ]

                per_gt_prop_inds[b, assign[1]] = assign[0]
                gt_matched_mask[b, assign[1]] = 1

            assignments.append(assign)

        return {
            "assignments": assignments,
            "per_gt_prop_inds": per_gt_prop_inds,
            "gt_matched_mask": gt_matched_mask
        }

    def decode_caption(raw_caption, idx2word, special_tokens):
        decoded = [special_tokens["bos_token"]]
        for token_idx in raw_caption:
            token_idx = token_idx.item()
            token = idx2word[str(token_idx)]
            decoded.append(token)
            if token == special_tokens["eos_token"]: break

        if special_tokens["eos_token"] not in decoded: decoded.append(special_tokens["eos_token"])
        decoded = " ".join(decoded)

        return decoded
    
    candidates = {}

    # assign GTs to predicted boxes
    assignments = box_assignment(pred_boxes, gt_boxes, gt_box_masks)

    batch_size, num_gts = gt_box_ids.shape
    per_gt_prop_inds = assignments["per_gt_prop_inds"]
    matched_prop_box_corners = torch.gather(
        pred_boxes, 1, per_gt_prop_inds[:, :, None, None].repeat(1, 1, 8, 3)
    ) # batch_size, num_gts, 8, 3 
    matched_ious = box3d_iou_batch_tensor(
        matched_prop_box_corners.reshape(-1, 8, 3), 
        gt_boxes.reshape(-1, 8, 3)
    ).reshape(batch_size, num_gts)

    candidates = {}
    for batch_id in range(batch_size):
        scene_id = gt_scene_list[batch_id]
        for gt_id in range(num_gts):
            if gt_box_masks[batch_id, gt_id] == 0: continue

            object_id = str(gt_box_ids[batch_id, gt_id].item())
            caption_decoded = decode_caption(pred_captions[batch_id, per_gt_prop_inds[batch_id, gt_id]], idx2word, special_tokens)
            iou = matched_ious[batch_id, gt_id].item()
            box = matched_prop_box_corners[batch_id, gt_id].detach().cpu().numpy().tolist()
            gt_box = gt_boxes[batch_id, gt_id].detach().cpu().numpy().tolist()

            # store
            key = "{}|{}".format(scene_id, object_id)
            entry = [
                [caption_decoded],
                iou,
                box,
                gt_box
            ]
            if key not in candidates:
                candidates[key] = entry
            else:
                # update the stored prediction if IoU is higher
                if iou > candidates[key][1]:
                    candidates[key] = entry

    return candidates

def eval_caption_step(data_dict, dataset, detection, phase="val", min_iou=CONF.EVAL.MIN_IOU_THRESHOLD):
    candidates = {}

    if not detection:
        raise NotImplementedError()
    else:

        candidates = assign_dense_caption(
            pred_captions=data_dict["lang_cap"], # batch_size, num_proposals, num_words - 1/max_len
            pred_boxes=data_dict["bbox_corner"], 
            gt_boxes=data_dict["gt_box_corner_label"], 
            gt_box_ids=data_dict["gt_box_object_ids"], 
            gt_box_masks=data_dict["gt_box_masks"], 
            gt_scene_list=data_dict["scene_id"],
            idx2word=dataset.vocabulary["idx2word"],
            special_tokens=dataset.vocabulary["special_tokens"],
            strategy="center"
        )

    return candidates


# def eval_caption_step(data_dict, dataset, detection, phase="val", min_iou=CONF.EVAL.MIN_IOU_THRESHOLD):
#     candidates = {}

#     organized = get_organized(dataset.name, phase)

#     # unpack
#     captions = data_dict["lang_cap"] # [batch_size...[num_proposals...[num_words...]]]
#     # NOTE the captions are stacked
#     bbox_corners = data_dict["bbox_corner"]
#     dataset_ids = data_dict["dataset_idx"]
#     batch_size, num_proposals, _, _ = bbox_corners.shape

#     # # collapse
#     # chunk_size = captions.shape[0] // batch_size
#     # captions = captions.reshape(batch_size, chunk_size, num_proposals, -1)
#     # captions = captions[:, 0] #

#     # post-process
#     # config
#     POST_DICT = {
#         "remove_empty_box": True, 
#         "use_3d_nms": True, 
#         "nms_iou": 0.25,
#         "use_old_type_nms": False, 
#         "cls_nms": True, 
#         "per_class_proposal": True,
#         "conf_thresh": 0.05,
#         "dataset_config": DC
#     }

#     if not detection:
#         nms_masks = data_dict["bbox_mask"]

#         detected_object_ids = data_dict["bbox_object_ids"]
#         ious = torch.ones(batch_size, num_proposals).type_as(bbox_corners)
#     else:
#         # nms mask
#         _ = parse_predictions(data_dict, POST_DICT)
#         nms_masks = torch.FloatTensor(data_dict["pred_mask"]).type_as(bbox_corners).long()

#         # objectness mask
#         obj_masks = torch.argmax(data_dict["objectness_scores"], 2).long()

#         # final mask
#         nms_masks = nms_masks * obj_masks

#         # pick out object ids of detected objects
#         detected_object_ids = torch.gather(data_dict["scene_object_ids"], 1, data_dict["object_assignment"])

#         # bbox corners
#         assigned_target_bbox_corners = torch.gather(
#             data_dict["gt_box_corner_label"], 
#             1, 
#             data_dict["object_assignment"].view(batch_size, num_proposals, 1, 1).repeat(1, 1, 8, 3)
#         ) # batch_size, num_proposals, 8, 3
#         detected_bbox_corners = data_dict["bbox_corner"] # batch_size, num_proposals, 8, 3
        
#         # compute IoU between each detected box and each ground truth box
#         ious = box3d_iou_batch_tensor(
#             assigned_target_bbox_corners.view(-1, 8, 3), # batch_size * num_proposals, 8, 3
#             detected_bbox_corners.view(-1, 8, 3) # batch_size * num_proposals, 8, 3
#         ).view(batch_size, num_proposals)

#         # change shape
#         assigned_target_bbox_corners = assigned_target_bbox_corners.view(-1, num_proposals, 8, 3) # batch_size, num_proposals, 8, 3
#         detected_bbox_corners = detected_bbox_corners.view(-1, num_proposals, 8, 3) # batch_size, num_proposals, 8, 3

#     # find good boxes (IoU > threshold)
#     good_bbox_masks = ious > min_iou # batch_size, num_proposals

#     # dump generated captions
#     for batch_id in range(batch_size):
#         dataset_idx = dataset_ids[batch_id].item()
#         scene_id = dataset.scanrefer_new[dataset_idx][0]["scene_id"]
#         for prop_id in range(num_proposals):
#             # if nms_masks[batch_id, prop_id] == 1 and good_bbox_masks[batch_id, prop_id] == 1:
#             if nms_masks[batch_id, prop_id] == 1:
#                 object_id = str(detected_object_ids[batch_id, prop_id].item())
#                 caption_decoded = decode_caption(captions[batch_id][prop_id], dataset.vocabulary["idx2word"])

#                 try:
#                     object_name = organized[scene_id][object_id][0]["object_name"]

#                     # store
#                     key = "{}|{}|{}".format(scene_id, object_id, object_name)
#                     entry = [
#                         [caption_decoded],
#                         ious[batch_id, prop_id].item(),
#                         detected_bbox_corners[batch_id, prop_id].detach().cpu().numpy().tolist(),
#                         assigned_target_bbox_corners[batch_id, prop_id].detach().cpu().numpy().tolist()
#                     ]
#                     if key not in candidates:
#                         candidates[key] = entry
#                     else:
#                         # update the stored prediction if IoU is higher
#                         if ious[batch_id, prop_id].item() > candidates[key][1]:
#                             candidates[key] = entry

#                     # print(key, caption_decoded)

#                 except KeyError:
#                     continue

#     return candidates

def eval_caption_epoch(candidates, dataset, folder, device, phase="val", force=False, max_len=CONF.EVAL.MAX_DES_LEN+2, min_iou=CONF.EVAL.MIN_IOU_THRESHOLD):
    # corpus
    corpus_path = os.path.join(CONF.PATH.OUTPUT, folder, "corpus_{}_{}.json".format(phase, str(device.index)))
    
    # print("preparing corpus...")
    if dataset.name == "ScanRefer":
        raw_data = SCANREFER_RAW[phase]
    elif dataset.name == "ReferIt3D":
        raw_data = REFERIT3D_RAW[phase]
    else:
        raise ValueError("Invalid dataset.")

    corpus = prepare_corpus(raw_data, candidates, dataset.vocabulary["special_tokens"], max_len)
    
    if not os.path.exists(corpus_path) or force:
        with open(corpus_path, "w") as f:
            json.dump(corpus, f, indent=4)

    pred_path = os.path.join(CONF.PATH.OUTPUT, folder, "pred_{}_{}.json".format(phase, str(device.index)))
    with open(pred_path, "w") as f:
        json.dump(candidates, f, indent=4)

    # check candidates
    # NOTE: make up the captions for the undetected object by "sos eos"
    candidates = filter_candidates(candidates, min_iou)
    candidates = check_candidates(corpus, candidates, dataset.vocabulary["special_tokens"])
    candidates = organize_candidates(corpus, candidates)

    # candidates for evaluation -> debug
    temp_path = os.path.join(CONF.PATH.OUTPUT, folder, "pred_processed_{}_{}_{}.json".format(phase, str(device.index), str(min_iou)))
    with open(temp_path, "w") as f:
        json.dump(candidates, f, indent=4)

    # compute scores
    # print("computing scores...")
    bleu = capblue.Bleu(4).compute_score(corpus, candidates)
    cider = capcider.Cider().compute_score(corpus, candidates)
    rouge = caprouge.Rouge().compute_score(corpus, candidates) 
    meteor = capmeteor.Meteor().compute_score(corpus, candidates)

    # save results
    result_path = os.path.join(CONF.PATH.OUTPUT, folder, "eval_{}_{}_{}.txt".format(phase, str(device.index), str(min_iou)))
    with open(result_path, "w") as f:
        f.write("----------------------Evaluation-----------------------\n")
        f.write("[BLEU-1] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}\n".format(bleu[0][0], max(bleu[1][0]), min(bleu[1][0])))
        f.write("[BLEU-2] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}\n".format(bleu[0][1], max(bleu[1][1]), min(bleu[1][1])))
        f.write("[BLEU-3] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}\n".format(bleu[0][2], max(bleu[1][2]), min(bleu[1][2])))
        f.write("[BLEU-4] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}\n".format(bleu[0][3], max(bleu[1][3]), min(bleu[1][3])))
        f.write("[CIDEr] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}\n".format(cider[0], max(cider[1]), min(cider[1])))
        f.write("[ROUGE-L] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}\n".format(rouge[0], max(rouge[1]), min(rouge[1])))
        f.write("[METEOR] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}\n".format(meteor[0], max(meteor[1]), min(meteor[1])))

    return bleu, cider, rouge, meteor

