""" 
Helper functions and class to calculate Average Precisions for 3D object detection.

Modified from: https://github.com/facebookresearch/votenet/blob/master/models/ap_helper.py
"""
import os
import sys
import numpy as np

from benchmark.box_util import extract_pc_in_box3d
from benchmark.eval_det import eval_det, get_iou_obb
from benchmark.nms import nms_2d_faster, nms_3d_faster, nms_3d_faster_samecls

def flip_axis_to_camera(pc):
    """ Flip X-right,Y-forward,Z-up to X-right,Y-down,Z-forward
    Input and output are both (N,3) array
    """
    pc2 = np.copy(pc)
    pc2[...,[0,1,2]] = pc2[...,[0,2,1]] # cam X,Y,Z = depth X,-Z,Y
    pc2[...,1] *= -1
    return pc2

def flip_axis_to_depth(pc):
    pc2 = np.copy(pc)
    pc2[...,[0,1,2]] = pc2[...,[0,2,1]] # depth X,Y,Z = cam X,Z,-Y
    pc2[...,2] *= -1
    return pc2

def softmax(x):
    """ Numpy function for softmax"""
    shape = x.shape
    probs = np.exp(x - np.max(x, axis=len(shape)-1, keepdims=True))
    probs /= np.sum(probs, axis=len(shape)-1, keepdims=True)
    return probs

def parse_predictions(end_points, config_dict):
    """ Parse predictions to OBB parameters and suppress overlapping boxes
    
    Args:
        end_points: dict
            {point_clouds, boxes, sem_prob, obj_prob}
        config_dict: dict
            {dataset_config, remove_empty_box, use_3d_nms, nms_iou,
            use_old_type_nms, conf_thresh, per_class_proposal}

    Returns:
        batch_pred_map_cls: a list of len == batch size (BS)
            [pred_list_i], i = 0, 1, ..., BS-1
            where pred_list_i = [(pred_sem_cls, box_params, box_score)_j]
            where j = 0, ..., num of valid detections - 1 from sample input i
    """


    pred_sem_cls = np.argmax(end_points["sem_prob"], -1) # B,num_proposal
    sem_cls_probs = end_points["sem_prob"] # B,num_proposal,num_cls
    pred_sem_cls_prob = np.max(sem_cls_probs,-1) # B,num_proposal

    pred_corners_3d_upright_camera = end_points["boxes"] # B,num_proposal,8,3

    obj_prob = end_points["obj_prob"][:,:,1] # (B,K)

    bsize = pred_corners_3d_upright_camera.shape[0]
    K = pred_corners_3d_upright_camera.shape[1] # K==num_proposal
    nonempty_box_mask = np.ones((bsize, K))

    if config_dict["remove_empty_box"]:
        # # -------------------------------------
        # # Remove predicted boxes without any point within them..
        # batch_pc = end_points["point_clouds"][:,:,0:3] # B,N,3
        # for i in range(bsize):
        #     pc = batch_pc[i,:,:] # (N,3)
        #     for j in range(K):
        #         box3d = pred_corners_3d_upright_camera[i,j,:,:] # (8,3)
        #         # box3d = flip_axis_to_depth(box3d)
        #         pc_in_box,inds = extract_pc_in_box3d(pc, box3d)
        #         if len(pc_in_box) < 5:
        #             nonempty_box_mask[i,j] = 0
        # # -------------------------------------
        pass
    
    if not config_dict["use_3d_nms"]:
        # ---------- NMS input: pred_with_prob in (B,K,7) -----------
        pred_mask = np.zeros((bsize, K))
        for i in range(bsize):
            boxes_2d_with_prob = np.zeros((K,5))
            for j in range(K):
                boxes_2d_with_prob[j,0] = np.min(pred_corners_3d_upright_camera[i,j,:,0])
                boxes_2d_with_prob[j,2] = np.max(pred_corners_3d_upright_camera[i,j,:,0])
                boxes_2d_with_prob[j,1] = np.min(pred_corners_3d_upright_camera[i,j,:,2])
                boxes_2d_with_prob[j,3] = np.max(pred_corners_3d_upright_camera[i,j,:,2])
                boxes_2d_with_prob[j,4] = obj_prob[i,j]
            nonempty_box_inds = np.where(nonempty_box_mask[i,:]==1)[0]
            pick = nms_2d_faster(boxes_2d_with_prob[nonempty_box_mask[i,:]==1,:],
                config_dict["nms_iou"], config_dict["use_old_type_nms"])
            assert(len(pick)>0)
            pred_mask[i, nonempty_box_inds[pick]] = 1
        end_points["pred_mask"] = pred_mask
        # ---------- NMS output: pred_mask in (B,K) -----------
    elif config_dict["use_3d_nms"] and (not config_dict["cls_nms"]):
        # ---------- NMS input: pred_with_prob in (B,K,7) -----------
        pred_mask = np.zeros((bsize, K))
        for i in range(bsize):
            boxes_3d_with_prob = np.zeros((K,7))
            for j in range(K):
                boxes_3d_with_prob[j,0] = np.min(pred_corners_3d_upright_camera[i,j,:,0])
                boxes_3d_with_prob[j,1] = np.min(pred_corners_3d_upright_camera[i,j,:,1])
                boxes_3d_with_prob[j,2] = np.min(pred_corners_3d_upright_camera[i,j,:,2])
                boxes_3d_with_prob[j,3] = np.max(pred_corners_3d_upright_camera[i,j,:,0])
                boxes_3d_with_prob[j,4] = np.max(pred_corners_3d_upright_camera[i,j,:,1])
                boxes_3d_with_prob[j,5] = np.max(pred_corners_3d_upright_camera[i,j,:,2])
                boxes_3d_with_prob[j,6] = obj_prob[i,j]
            nonempty_box_inds = np.where(nonempty_box_mask[i,:]==1)[0]
            pick = nms_3d_faster(boxes_3d_with_prob[nonempty_box_mask[i,:]==1,:],
                config_dict["nms_iou"], config_dict["use_old_type_nms"])
            assert(len(pick)>0)
            pred_mask[i, nonempty_box_inds[pick]] = 1
        end_points["pred_mask"] = pred_mask
        # ---------- NMS output: pred_mask in (B,K) -----------
    elif config_dict["use_3d_nms"] and config_dict["cls_nms"]:
        # ---------- NMS input: pred_with_prob in (B,K,8) -----------
        pred_mask = np.zeros((bsize, K))
        for i in range(bsize):
            boxes_3d_with_prob = np.zeros((K,8))
            for j in range(K):
                boxes_3d_with_prob[j,0] = np.min(pred_corners_3d_upright_camera[i,j,:,0])
                boxes_3d_with_prob[j,1] = np.min(pred_corners_3d_upright_camera[i,j,:,1])
                boxes_3d_with_prob[j,2] = np.min(pred_corners_3d_upright_camera[i,j,:,2])
                boxes_3d_with_prob[j,3] = np.max(pred_corners_3d_upright_camera[i,j,:,0])
                boxes_3d_with_prob[j,4] = np.max(pred_corners_3d_upright_camera[i,j,:,1])
                boxes_3d_with_prob[j,5] = np.max(pred_corners_3d_upright_camera[i,j,:,2])
                boxes_3d_with_prob[j,6] = obj_prob[i,j]
                boxes_3d_with_prob[j,7] = pred_sem_cls[i,j] # only suppress if the two boxes are of the same class!!
            nonempty_box_inds = np.where(nonempty_box_mask[i,:]==1)[0]
            pick = nms_3d_faster_samecls(boxes_3d_with_prob[nonempty_box_mask[i,:]==1,:],
                config_dict["nms_iou"], config_dict["use_old_type_nms"])
            assert(len(pick)>0)
            pred_mask[i, nonempty_box_inds[pick]] = 1
        end_points["pred_mask"] = pred_mask
        # ---------- NMS output: pred_mask in (B,K) -----------

    batch_pred_map_cls = [] # a list (len: batch_size) of list (len: num of predictions per sample) of tuples of pred_cls, pred_box and conf (0-1)
    for i in range(bsize):
        if config_dict["per_class_proposal"]:
            cur_list = []
            for ii in range(config_dict["dataset_config"].num_class):
                cur_list += [(ii, pred_corners_3d_upright_camera[i,j], sem_cls_probs[i,j,ii]*obj_prob[i,j]) \
                    for j in range(pred_corners_3d_upright_camera.shape[1]) if pred_mask[i,j]==1 and obj_prob[i,j]>config_dict["conf_thresh"]]
            batch_pred_map_cls.append(cur_list)
        else:
            batch_pred_map_cls.append([(pred_sem_cls[i,j].item(), pred_corners_3d_upright_camera[i,j], obj_prob[i,j]) \
                for j in range(pred_corners_3d_upright_camera.shape[1]) if pred_mask[i,j]==1 and obj_prob[i,j]>config_dict["conf_thresh"]])
    end_points["batch_pred_map_cls"] = batch_pred_map_cls

    return batch_pred_map_cls

def parse_groundtruths(end_points, config_dict):
    """ Parse groundtruth labels to OBB parameters.
    
    Args:
        end_points: dict
            {box_label, sem_cls_label}
        config_dict: dict
            {dataset_config}

    Returns:
        batch_gt_map_cls: a list  of len == batch_size (BS)
            [gt_list_i], i = 0, 1, ..., BS-1
            where gt_list_i = [(gt_sem_cls, gt_box_params)_j]
            where j = 0, ..., num of objects - 1 at sample input i
    """

    gt_corners_3d_upright_camera = end_points["box_label"]
    sem_cls_label = end_points["sem_cls_label"]
    bsize = gt_corners_3d_upright_camera.shape[0]

    batch_gt_map_cls = []
    for i in range(bsize):
        batch_gt_map_cls.append([(sem_cls_label[i,j].item(), gt_corners_3d_upright_camera[i,j]) for j in range(gt_corners_3d_upright_camera.shape[1])])
    end_points["batch_gt_map_cls"] = batch_gt_map_cls

    return batch_gt_map_cls

class APCalculator(object):
    """ Calculating Average Precision """
    def __init__(self, ap_iou_thresh=0.25, class2type_map=None):
        """
        Args:
            ap_iou_thresh: float between 0 and 1.0
                IoU threshold to judge whether a prediction is positive.
            class2type_map: [optional] dict {class_int:class_name}
        """
        self.ap_iou_thresh = ap_iou_thresh
        self.class2type_map = class2type_map
        self.reset()
        
    def step(self, batch_pred_map_cls, batch_gt_map_cls):
        """ Accumulate one batch of prediction and groundtruth.
        
        Args:
            batch_pred_map_cls: a list of lists [[(pred_cls, pred_box_params, score),...],...]
            batch_gt_map_cls: a list of lists [[(gt_cls, gt_box_params),...],...]
                should have the same length with batch_pred_map_cls (batch_size)
        """
        
        bsize = len(batch_pred_map_cls)
        assert(bsize == len(batch_gt_map_cls))
        for i in range(bsize):
            self.gt_map_cls[self.scan_cnt] = batch_gt_map_cls[i] 
            self.pred_map_cls[self.scan_cnt] = batch_pred_map_cls[i] 
            self.scan_cnt += 1
    
    def compute_metrics(self):
        """ Use accumulated predictions and groundtruths to compute Average Precision.
        """
        # rec, prec, ap = eval_det_multiprocessing(self.pred_map_cls, self.gt_map_cls, ovthresh=self.ap_iou_thresh, get_iou_func=get_iou_obb)
        rec, prec, ap = eval_det(self.pred_map_cls, self.gt_map_cls, ovthresh=self.ap_iou_thresh, get_iou_func=get_iou_obb)
        ret_dict = {} 
        for key in sorted(ap.keys()):
            clsname = self.class2type_map[key] if self.class2type_map else str(key)
            ret_dict["%s Average Precision"%(clsname)] = ap[key]
        ret_dict["mAP"] = np.mean(list(ap.values()))
        rec_list = []
        for key in sorted(ap.keys()):
            clsname = self.class2type_map[key] if self.class2type_map else str(key)
            try:
                ret_dict["%s Recall"%(clsname)] = rec[key][-1]
                rec_list.append(rec[key][-1])
            except:
                ret_dict["%s Recall"%(clsname)] = 0
                rec_list.append(0)
        ret_dict["AR"] = np.mean(rec_list)
        return ret_dict

    def reset(self):
        self.gt_map_cls = {} # {scan_id: [(classname, bbox)]}
        self.pred_map_cls = {} # {scan_id: [(classname, bbox, score)]}
        self.scan_cnt = 0
