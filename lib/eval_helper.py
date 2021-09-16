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

sys.path.append(os.path.join(os.getcwd())) # HACK add the root folder

import lib.capeval.bleu.bleu as capblue
import lib.capeval.cider.cider as capcider
import lib.capeval.rouge.rouge as caprouge
import lib.capeval.meteor.meteor as capmeteor

from data.scannet.model_util_scannet import ScannetDatasetConfig
from lib.config import CONF
from lib.ap_helper import parse_predictions
from lib.loss_helper import get_loss
from utils.box_util import box3d_iou_batch_tensor

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

def prepare_corpus(raw_data, candidates, max_len=CONF.EVAL.MAX_DES_LEN+2):
    # get involved scene IDs
    scene_list = []
    for key in candidates.keys():
        scene_id, _, _ = key.split("|")
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
        description = "sos " + description
        description += " eos"

        key = "{}|{}|{}".format(scene_id, object_id, object_name)
        # key = "{}|{}".format(scene_id, object_id)

        if key not in corpus:
            corpus[key] = []

        corpus[key].append(description)

    return corpus

def decode_caption(raw_caption, idx2word):
    decoded = ["sos"]
    for token_idx in raw_caption:
        token_idx = token_idx.item()
        token = idx2word[str(token_idx)]
        decoded.append(token)
        if token == "eos": break

    if "eos" not in decoded: decoded.append("eos")
    decoded = " ".join(decoded)

    return decoded

def filter_candidates(candidates, min_iou):
    new_candidates = {}
    for key, value in candidates.items():
        des, iou = value
        if iou >= min_iou:
            new_candidates[key] = des

    return new_candidates

def check_candidates(corpus, candidates):
    placeholder = "sos eos"
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

def eval_caption_step(data_dict, dataset, detection, phase="val", min_iou=CONF.EVAL.MIN_IOU_THRESHOLD):
    candidates = {}

    organized = get_organized(dataset.name, phase)

    # unpack
    captions = data_dict["lang_cap"] # [batch_size...[num_proposals...[num_words...]]]
    # NOTE the captions are stacked
    bbox_corners = data_dict["bbox_corner"]
    dataset_ids = data_dict["dataset_idx"]
    batch_size, num_proposals, _, _ = bbox_corners.shape

    # # collapse
    # chunk_size = captions.shape[0] // batch_size
    # captions = captions.reshape(batch_size, chunk_size, num_proposals, -1)
    # captions = captions[:, 0] #

    # post-process
    # config
    POST_DICT = {
        "remove_empty_box": True, 
        "use_3d_nms": True, 
        "nms_iou": 0.25,
        "use_old_type_nms": False, 
        "cls_nms": True, 
        "per_class_proposal": True,
        "conf_thresh": 0.05,
        "dataset_config": DC
    }

    if not detection:
        nms_masks = data_dict["bbox_mask"]

        detected_object_ids = data_dict["bbox_object_ids"]
        ious = torch.ones(batch_size, num_proposals).type_as(bbox_corners)
    else:
        # nms mask
        _ = parse_predictions(data_dict, POST_DICT)
        nms_masks = torch.FloatTensor(data_dict["pred_mask"]).type_as(bbox_corners).long()

        # objectness mask
        obj_masks = torch.argmax(data_dict["objectness_scores"], 2).long()

        # final mask
        nms_masks = nms_masks * obj_masks

        # pick out object ids of detected objects
        detected_object_ids = torch.gather(data_dict["scene_object_ids"], 1, data_dict["object_assignment"])

        # bbox corners
        assigned_target_bbox_corners = torch.gather(
            data_dict["gt_box_corner_label"], 
            1, 
            data_dict["object_assignment"].view(batch_size, num_proposals, 1, 1).repeat(1, 1, 8, 3)
        ) # batch_size, num_proposals, 8, 3
        detected_bbox_corners = data_dict["bbox_corner"] # batch_size, num_proposals, 8, 3
        
        # compute IoU between each detected box and each ground truth box
        ious = box3d_iou_batch_tensor(
            assigned_target_bbox_corners.view(-1, 8, 3), # batch_size * num_proposals, 8, 3
            detected_bbox_corners.view(-1, 8, 3) # batch_size * num_proposals, 8, 3
        ).view(batch_size, num_proposals)

    # find good boxes (IoU > threshold)
    good_bbox_masks = ious > min_iou # batch_size, num_proposals

    # dump generated captions
    for batch_id in range(batch_size):
        dataset_idx = dataset_ids[batch_id].item()
        scene_id = dataset.scanrefer_new[dataset_idx][0]["scene_id"]
        for prop_id in range(num_proposals):
            # if nms_masks[batch_id, prop_id] == 1 and good_bbox_masks[batch_id, prop_id] == 1:
            if nms_masks[batch_id, prop_id] == 1:
                object_id = str(detected_object_ids[batch_id, prop_id].item())
                caption_decoded = decode_caption(captions[batch_id][prop_id], dataset.vocabulary["idx2word"])

                try:
                    object_name = organized[scene_id][object_id][0]["object_name"]

                    # store
                    key = "{}|{}|{}".format(scene_id, object_id, object_name)
                    if key not in candidates:
                        candidates[key] = (
                            [caption_decoded],
                            ious[batch_id, prop_id].item()
                        )
                    else:
                        # update the stored prediction if IoU is higher
                        if ious[batch_id, prop_id].item() > candidates[key][1]:
                            candidates[key] = (
                                [caption_decoded],
                                ious[batch_id, prop_id].item()
                            )

                    # print(key, caption_decoded)

                except KeyError:
                    continue

    return candidates

def eval_caption_epoch(candidates, dataset, folder, device, phase="val", force=False, max_len=CONF.EVAL.MAX_DES_LEN+2, min_iou=CONF.EVAL.MIN_IOU_THRESHOLD):
    # corpus
    corpus_path = os.path.join(CONF.PATH.OUTPUT, folder, "corpus_{}_{}.json".format(phase, str(device)))
    
    # print("preparing corpus...")
    if dataset.name == "ScanRefer":
        raw_data = SCANREFER_RAW[phase]
    elif dataset.name == "ReferIt3D":
        raw_data = REFERIT3D_RAW[phase]
    else:
        raise ValueError("Invalid dataset.")

    corpus = prepare_corpus(raw_data, candidates, max_len)
    
    if not os.path.exists(corpus_path) or force:
        with open(corpus_path, "w") as f:
            json.dump(corpus, f, indent=4)

    pred_path = os.path.join(CONF.PATH.OUTPUT, folder, "pred_{}_{}.json".format(phase, str(device)))
    with open(pred_path, "w") as f:
        json.dump(candidates, f, indent=4)

    # check candidates
    # NOTE: make up the captions for the undetected object by "sos eos"
    candidates = filter_candidates(candidates, min_iou)
    candidates = check_candidates(corpus, candidates)
    candidates = organize_candidates(corpus, candidates)

    # compute scores
    # print("computing scores...")
    bleu = capblue.Bleu(4).compute_score(corpus, candidates)
    cider = capcider.Cider().compute_score(corpus, candidates)
    rouge = caprouge.Rouge().compute_score(corpus, candidates)
    meteor = capmeteor.Meteor().compute_score(corpus, candidates)

    return bleu, cider, rouge, meteor

