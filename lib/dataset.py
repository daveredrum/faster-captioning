'''
File Created: Monday, 25th November 2019 1:35:30 pm
Author: Dave Zhenyu Chen (zhenyu.chen@tum.de)
'''

import os
import sys
import time
import h5py
import json
import pickle
import random
import numpy as np
import multiprocessing as mp
from torch.utils.data import Dataset

from itertools import chain
from collections import Counter
from transformers import BertTokenizer

sys.path.append(os.path.join(os.getcwd(), "lib")) # HACK add the lib folder
from lib.config import CONF
from utils.pc_utils import random_sampling, rotx, roty, rotz
from utils.box_util import get_3d_box, get_3d_box_batch
from data.scannet.model_util_scannet import rotate_aligned_boxes, ScannetDatasetConfig, rotate_aligned_boxes_along_axis

# data setting
DC = ScannetDatasetConfig()
MAX_NUM_OBJ = 128
MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8])

# data path
SCANNET_V2_TSV = os.path.join(CONF.PATH.SCANNET_META, "scannetv2-labels.combined.tsv")
MULTIVIEW_DATA = CONF.MULTIVIEW
GLOVE_PICKLE = os.path.join(CONF.PATH.DATA, "glove.p")

VOCAB = os.path.join(CONF.PATH.DATA, "{}_vocabulary.json") # dataset_name
GLOVE_PATH = os.path.join(CONF.PATH.DATA, "glove_trimmed_{}.npy") # dataset_name

class ScannetReferenceDataset(Dataset):
       
    def __init__(self, scanrefer, scanrefer_all_scene, 
        split="train", 
        name="ScanRefer",
        num_points=40000,
        use_height=False, 
        use_color=False, 
        use_normal=False, 
        use_multiview=False, 
        augment=False,
        debug=False,
        scan2cad_rotation=None,
        use_bert_vocab=False):

        self.scanrefer = scanrefer
        self.scanrefer_all_scene = scanrefer_all_scene # all scene_ids in scanrefer
        self.split = split
        self.name = name
        self.num_points = num_points
        self.use_color = use_color        
        self.use_height = use_height
        self.use_normal = use_normal        
        self.use_multiview = use_multiview
        self.augment = augment
        self.debug = debug
        self.scan2cad_rotation = scan2cad_rotation
        self.use_bert_vocab = use_bert_vocab

        # load data
        self._load_data()

        # chunk data
        self.scanrefer_new = self._get_chunked_data(self.scanrefer)

        self.multiview_data = {}
       
    def __len__(self):
        return len(self.scanrefer_new)

    def __getitem__(self, idx):
        start = time.time()

        chunk_size = len(self.scanrefer_new[idx])

        scene_id = self.scanrefer_new[idx][0]["scene_id"]

        chunk_id_list = np.zeros((CONF.TRAIN.NUM_DES_PER_SCENE))

        object_id_list = np.zeros((CONF.TRAIN.NUM_DES_PER_SCENE))
        ann_id_list = np.zeros((CONF.TRAIN.NUM_DES_PER_SCENE))

        lang_feat_list = np.zeros((CONF.TRAIN.NUM_DES_PER_SCENE, CONF.TRAIN.MAX_DES_LEN + 2, 300))
        lang_len_list = np.zeros((CONF.TRAIN.NUM_DES_PER_SCENE))
        lang_id_list = np.zeros((CONF.TRAIN.NUM_DES_PER_SCENE, CONF.TRAIN.MAX_DES_LEN + 2))
        
        annotated_list = np.zeros((CONF.TRAIN.NUM_DES_PER_SCENE))

        unique_multiple_list = np.zeros((CONF.TRAIN.NUM_DES_PER_SCENE))

        object_cat_list = np.zeros((CONF.TRAIN.NUM_DES_PER_SCENE))

        for i in range(CONF.TRAIN.NUM_DES_PER_SCENE):
            if i < chunk_size:
                chunk_id = i

                object_id = int(self.scanrefer_new[idx][i]["object_id"])

                if object_id != "SYNTHETIC":
                    annotated = 1

                    object_name = " ".join(self.scanrefer_new[idx][i]["object_name"].split("_"))
                    ann_id = self.scanrefer_new[idx][i]["ann_id"]

                    object_cat = self.raw2label[object_name] if object_name in self.raw2label else 17
                    
                    # get language features
                    lang_feat = self.lang[scene_id][str(object_id)][ann_id]
                    lang_len = len(self.scanrefer_new[idx][i]["token"]) + 2
                    lang_len = lang_len if lang_len <= CONF.TRAIN.MAX_DES_LEN + 2 else CONF.TRAIN.MAX_DES_LEN + 2

                    # # NOTE 50% chance that 20% of the tokens are erased during training
                    # if self.split == "train" and random.random() < 0.5:
                    #     lang_feat = self._tranform_des_with_erase(lang_feat, lang_len, p=0.2)

                    lang_ids = self.lang_ids[scene_id][str(object_id)][ann_id]

                    unique_multiple_flag = self.unique_multiple_lookup[scene_id][str(object_id)][ann_id]
                else:
                    annotated = 0

                    object_id = -1
                    object_name = ""
                    ann_id = -1

                    object_cat = 17 # will be changed in the model

                    # synthesize language features
                    lang_feat = np.zeros((CONF.TRAIN.MAX_DES_LEN + 2, 300))
                    lang_len = 0

                    lang_ids = np.zeros(CONF.TRAIN.MAX_DES_LEN + 2)

                    unique_multiple_flag = 0

            # store
            # HACK the last sample will be repeated if chunk size 
            # is smaller than CONF.TRAIN.NUM_DES_PER_SCENE
            chunk_id_list[i] = chunk_id
            
            object_id_list[i] = object_id
            ann_id_list[i] = ann_id

            lang_feat_list[i] = lang_feat
            lang_len_list[i] = lang_len
            lang_id_list[i] = lang_ids
            
            annotated_list[i] = annotated
            
            unique_multiple_list[i] = unique_multiple_flag

            object_cat_list[i] = object_cat

        # get pc
        mesh_vertices = self.scene_data[scene_id]["mesh_vertices"]
        instance_labels = self.scene_data[scene_id]["instance_labels"]
        semantic_labels = self.scene_data[scene_id]["semantic_labels"]
        instance_bboxes = self.scene_data[scene_id]["instance_bboxes"]

        if not self.use_color:
            point_cloud = mesh_vertices[:,0:3] # do not use color for now
            pcl_color = mesh_vertices[:,3:6]
        else:
            point_cloud = mesh_vertices[:,0:6] 
            point_cloud[:,3:6] = (point_cloud[:,3:6]-MEAN_COLOR_RGB)/256.0
            pcl_color = point_cloud[:,3:6]
        
        if self.use_normal:
            normals = mesh_vertices[:,6:9]
            point_cloud = np.concatenate([point_cloud, normals],1)

        if self.use_multiview:
            # load multiview database
            pid = mp.current_process().pid
            if pid not in self.multiview_data:
                self.multiview_data[pid] = h5py.File(MULTIVIEW_DATA, "r", libver="latest")

            multiview = self.multiview_data[pid][scene_id]
            point_cloud = np.concatenate([point_cloud, multiview],1)

        if self.use_height:
            floor_height = np.percentile(point_cloud[:,2],0.99)
            height = point_cloud[:,2] - floor_height
            point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)],1) 
        
        point_cloud, choices = random_sampling(point_cloud, self.num_points, return_choices=True)        
        instance_labels = instance_labels[choices]
        semantic_labels = semantic_labels[choices]
        pcl_color = pcl_color[choices]
        
        # ------------------------------- LABELS ------------------------------    
        target_bboxes = np.zeros((MAX_NUM_OBJ, 6))
        target_bboxes_mask = np.zeros((MAX_NUM_OBJ))
        target_object_ids = np.zeros((MAX_NUM_OBJ,)) # object ids of all objects
        angle_classes = np.zeros((MAX_NUM_OBJ,))
        angle_residuals = np.zeros((MAX_NUM_OBJ,))
        size_classes = np.zeros((MAX_NUM_OBJ,))
        size_residuals = np.zeros((MAX_NUM_OBJ, 3))
        bbox_num_pts = np.zeros((MAX_NUM_OBJ,))

        ref_box_label_list = np.zeros((CONF.TRAIN.NUM_DES_PER_SCENE, MAX_NUM_OBJ))
        ref_center_label_list = np.zeros((CONF.TRAIN.NUM_DES_PER_SCENE, 3))
        ref_heading_class_label_list = np.zeros((CONF.TRAIN.NUM_DES_PER_SCENE))
        ref_heading_residual_label_list = np.zeros((CONF.TRAIN.NUM_DES_PER_SCENE))
        ref_size_class_label_list = np.zeros((CONF.TRAIN.NUM_DES_PER_SCENE))
        ref_size_residual_label_list = np.zeros((CONF.TRAIN.NUM_DES_PER_SCENE, 3))
        ref_box_corner_label_list = np.zeros((CONF.TRAIN.NUM_DES_PER_SCENE, 8, 3)) # NOTE the grounding GT should be decoded

        point_votes = np.zeros([self.num_points, 3])
        point_votes_mask = np.zeros(self.num_points)

        gt_box_corner_label = np.zeros((MAX_NUM_OBJ, 8, 3))
        gt_box_masks = np.zeros((MAX_NUM_OBJ,))
        gt_box_object_ids = np.zeros((MAX_NUM_OBJ,))

        target_bboxes_semcls = np.zeros((MAX_NUM_OBJ))
        ref_box_label = np.zeros(MAX_NUM_OBJ)

        # Transformation matrix
        trans_mat = np.eye(4)

        # object rotations
        scene_object_rotations = np.zeros((MAX_NUM_OBJ, 3, 3))
        scene_object_rotation_masks = np.zeros((MAX_NUM_OBJ,)) # NOTE this is not object mask!!!

        if self.split != "test":
            num_bbox = instance_bboxes.shape[0] if instance_bboxes.shape[0] < MAX_NUM_OBJ else MAX_NUM_OBJ
            target_bboxes_mask[0:num_bbox] = 1
            target_bboxes[0:num_bbox,:] = instance_bboxes[:MAX_NUM_OBJ,0:6]
            target_object_ids[0:num_bbox] = instance_bboxes[:, -1][0:num_bbox]

            # ------------------------------- DATA AUGMENTATION ------------------------------        
            if self.augment and not self.debug:
                flip_mat = np.eye(3)
                if np.random.random() > 0.5:
                    # Flipping along the YZ plane
                    point_cloud[:,0] = -1 * point_cloud[:,0]
                    target_bboxes[:,0] = -1 * target_bboxes[:,0]
                    flip_mat[0, 0] *= -1
                    
                if np.random.random() > 0.5:
                    # Flipping along the XZ plane
                    point_cloud[:,1] = -1 * point_cloud[:,1]
                    target_bboxes[:,1] = -1 * target_bboxes[:,1]
                    flip_mat[1, 1] *= -1

                # Rotation along X-axis
                rot_angle = (np.random.random()*np.pi/18) - np.pi/36 # -5 ~ +5 degree
                rot_mat_x = rotx(rot_angle)
                point_cloud[:,0:3] = np.dot(point_cloud[:,0:3], np.transpose(rot_mat_x))
                target_bboxes = rotate_aligned_boxes_along_axis(target_bboxes, rot_mat_x, "x")

                # Rotation along Y-axis
                rot_angle = (np.random.random()*np.pi/18) - np.pi/36 # -5 ~ +5 degree
                rot_mat_y = roty(rot_angle)
                point_cloud[:,0:3] = np.dot(point_cloud[:,0:3], np.transpose(rot_mat_y))
                target_bboxes = rotate_aligned_boxes_along_axis(target_bboxes, rot_mat_y, "y")

                # Rotation along up-axis/Z-axis
                rot_angle = (np.random.random()*np.pi/18) - np.pi/36 # -5 ~ +5 degree
                rot_mat_z = rotz(rot_angle)
                point_cloud[:,0:3] = np.dot(point_cloud[:,0:3], np.transpose(rot_mat_z))
                target_bboxes = rotate_aligned_boxes_along_axis(target_bboxes, rot_mat_z, "z")

                # Translation
                point_cloud, target_bboxes, factor = self._translate(point_cloud, target_bboxes)

                # Transformation matrix
                # NOTE total rotation is ZYX, to use it -> P = A dot (ZYX)^T
                rot_mat_zy = np.dot(rot_mat_z, rot_mat_y)
                rot_mat_zyx = np.dot(rot_mat_zy, rot_mat_x)

                # apply flip
                rot_mat_zyx *= flip_mat

                # store
                trans_mat[:3, :3] = rot_mat_zyx # rotation
                trans_mat[:3, 3] = factor # translation

            # compute votes *AFTER* augmentation
            # generate votes
            # Note: since there's no map between bbox instance labels and
            # pc instance_labels (it had been filtered 
            # in the data preparation step) we'll compute the instance bbox
            # from the points sharing the same instance label. 
            for i_instance in np.unique(instance_labels):            
                # find all points belong to that instance
                ind = np.where(instance_labels == i_instance)[0]
                # find the semantic label            
                if semantic_labels[ind[0]] in DC.nyu40ids:
                    x = point_cloud[ind,:3]
                    center = 0.5*(x.min(0) + x.max(0))
                    point_votes[ind, :] = center - x
                    point_votes_mask[ind] = 1.0
            point_votes = np.tile(point_votes, (1, 3)) # make 3 votes identical 
            
            class_ind = [DC.nyu40id2class[int(x)] for x in instance_bboxes[:num_bbox,-2]]
            # NOTE: set size class as semantic class. Consider use size2class.
            size_classes[0:num_bbox] = class_ind
            size_residuals[0:num_bbox, :] = target_bboxes[0:num_bbox, 3:6] - DC.mean_size_arr[class_ind,:]

            # construct all GT bbox corners
            all_obb = DC.param2obb_batch(target_bboxes[:num_bbox, 0:3], angle_classes[:num_bbox].astype(np.int64), angle_residuals[:num_bbox],
                                    size_classes[:num_bbox].astype(np.int64), size_residuals[:num_bbox])
            all_box_corner_label = get_3d_box_batch(all_obb[:, 3:6], all_obb[:, 6], all_obb[:, 0:3])
            
            # store
            gt_box_corner_label[:num_bbox] = all_box_corner_label
            gt_box_masks[:num_bbox] = 1
            gt_box_object_ids[:num_bbox] = instance_bboxes[:, -1]
            
            try:
                target_bboxes_semcls[0:num_bbox] = [DC.nyu40id2class[int(x)] for x in instance_bboxes[:,-2][0:num_bbox]]
            except KeyError:
                pass

            for i in range(num_bbox):
                # count the number of points for each object
                cur_ins_mask = instance_labels == (gt_box_object_ids[i])
                cur_int_num_pts = cur_ins_mask.sum()

                bbox_num_pts[i] = cur_int_num_pts

            # if scene is not in scan2cad annotations, skip
            # if the instance is not in scan2cad annotations, skip
            if self.scan2cad_rotation and scene_id in self.scan2cad_rotation:
                for i, instance_id in enumerate(instance_bboxes[:num_bbox,-1].astype(int)):
                    try:
                        rotation = np.array(self.scan2cad_rotation[scene_id][str(instance_id)])

                        scene_object_rotations[i] = rotation
                        scene_object_rotation_masks[i] = 1
                    except KeyError:
                        pass

            # construct the reference target label for each bbox
            for j in range(CONF.TRAIN.NUM_DES_PER_SCENE):
                for i, gt_id in enumerate(instance_bboxes[:num_bbox, -1]):
                    if gt_id == object_id_list[j]:
                        ref_box_label[i] = 1
                        ref_center_label = target_bboxes[i, 0:3]
                        ref_heading_class_label = angle_classes[i]
                        ref_heading_residual_label = angle_residuals[i]
                        ref_size_class_label = size_classes[i]
                        ref_size_residual_label = size_residuals[i]
                        ref_obb = DC.param2obb(ref_center_label, ref_heading_class_label, ref_heading_residual_label,
                            ref_size_class_label, ref_size_residual_label)
                        ref_box_corner_label = get_3d_box(ref_obb[3:6], ref_obb[6], ref_obb[0:3])

                        ref_box_label_list[j] = ref_box_label
                        ref_center_label_list[j] = ref_center_label
                        ref_heading_class_label_list[j] = ref_heading_class_label
                        ref_heading_residual_label_list[j] = ref_heading_residual_label
                        ref_size_class_label_list[j] = ref_size_class_label
                        ref_size_residual_label_list[j] = ref_size_residual_label
                        ref_box_corner_label_list[j] = ref_box_corner_label
                        
        else:
            num_bbox = 1
            point_votes = np.zeros([self.num_points, 9])
            point_votes_mask = np.zeros(self.num_points)

        data_dict = {}
        
        # basic info
        data_dict["istrain"] = 1 if self.split == "train" else 0
        data_dict["dataset_idx"] = np.array(idx).astype(np.int64)
        data_dict["annotated"] = np.array(annotated_list).astype(np.int64)
        data_dict["chunk_ids"] = np.array(chunk_id_list).astype(np.int64)
        data_dict["scene_id"] = scene_id

        # point cloud
        data_dict["point_clouds"] = point_cloud.astype(np.float32) # point cloud data including features
        data_dict["pcl_color"] = pcl_color
        data_dict["transformation"] = trans_mat # (4, 4) transformation matrix
        
        # detection
        data_dict["center_label"] = target_bboxes.astype(np.float32)[:,0:3] # (MAX_NUM_OBJ, 3) for GT box center XYZ
        data_dict["heading_class_label"] = angle_classes.astype(np.int64) # (MAX_NUM_OBJ,) with int values in 0,...,NUM_HEADING_BIN-1
        data_dict["heading_residual_label"] = angle_residuals.astype(np.float32) # (MAX_NUM_OBJ,)
        data_dict["size_class_label"] = size_classes.astype(np.int64) # (MAX_NUM_OBJ,) with int values in 0,...,NUM_SIZE_CLUSTER
        data_dict["size_residual_label"] = size_residuals.astype(np.float32) # (MAX_NUM_OBJ, 3)
        data_dict["num_bbox"] = np.array(num_bbox).astype(np.int64)
        data_dict["sem_cls_label"] = target_bboxes_semcls.astype(np.int64) # (MAX_NUM_OBJ,) semantic class index
        data_dict["box_label_mask"] = target_bboxes_mask.astype(np.float32) # (MAX_NUM_OBJ) as 0/1 with 1 indicating a unique box
        data_dict["vote_label"] = point_votes.astype(np.float32)
        data_dict["vote_label_mask"] = point_votes_mask.astype(np.int64)
        data_dict["bbox_num_pts"] = bbox_num_pts.astype(np.int64) # mask indicating the valid objects
        
        # rotation
        data_dict["scene_object_ids"] = target_object_ids.astype(np.int64) # (MAX_NUM_OBJ,) object ids of all objects
        data_dict["scene_object_rotations"] = scene_object_rotations.astype(np.float32) # (MAX_NUM_OBJ, 3, 3)
        data_dict["scene_object_rotation_masks"] = scene_object_rotation_masks.astype(np.int64) # (MAX_NUM_OBJ)

        # detection-decoded
        data_dict["gt_box_corner_label"] = gt_box_corner_label.astype(np.float64) # all GT box corners NOTE type must be double
        data_dict["gt_box_masks"] = gt_box_masks.astype(np.int64) # valid bbox masks
        data_dict["gt_box_object_ids"] = gt_box_object_ids.astype(np.int64) # valid bbox object ids

        # language
        data_dict["lang_feat"] = np.array(lang_feat_list).astype(np.float32) # language feature vectors
        data_dict["lang_len"] = np.array(lang_len_list).astype(np.int64) # length of each description
        data_dict["lang_ids"] = np.array(lang_id_list).astype(np.int64)
        
        # reference
        data_dict["object_id"] = np.array(object_id_list).astype(np.int64)
        data_dict["ann_id"] = np.array(ann_id_list).astype(np.int64)
        data_dict["object_cat"] = np.array(object_cat_list).astype(np.int64)
        data_dict["unique_multiple"] = np.array(unique_multiple_list).astype(np.int64)

        # reference (detection)
        data_dict["ref_box_label"] = np.array(ref_box_label_list).astype(np.int64) # 0/1 reference labels for each object bbox
        data_dict["ref_center_label"] = np.array(ref_center_label_list).astype(np.float32)
        data_dict["ref_heading_class_label"] = np.array(ref_heading_class_label_list).astype(np.int64)
        data_dict["ref_heading_residual_label"] = np.array(ref_heading_residual_label_list).astype(np.int64)
        data_dict["ref_size_class_label"] = np.array(ref_size_class_label_list).astype(np.int64)
        data_dict["ref_size_residual_label"] = np.array(ref_size_residual_label_list).astype(np.float32)
        data_dict["ref_box_corner_label"] = np.array(ref_box_corner_label_list).astype(np.float32)

        data_dict["load_time"] = time.time() - start

        return data_dict

    def _get_chunked_data(self, raw_data, chunk_size=CONF.TRAIN.NUM_DES_PER_SCENE):
        # scene data lookup dict: <scene_id> -> [scene_data_1, scene_data_2, ...]
        scene_data_dict = {}
        for data in raw_data:
            scene_id = data["scene_id"]

            if scene_id not in scene_data_dict: scene_data_dict[scene_id] = []

            scene_data_dict[scene_id].append(data)

        # chunk data
        new_data = []
        for scene_id, scene_data_list in scene_data_dict.items():
            for cur_chunk in self._chunks(scene_data_list, chunk_size):
                new_data.append(cur_chunk)

        return new_data

    def _chunks(self, lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def _get_raw2label(self):
        # mapping
        scannet_labels = DC.type2class.keys()
        scannet2label = {label: i for i, label in enumerate(scannet_labels)}

        lines = [line.rstrip() for line in open(SCANNET_V2_TSV)]
        lines = lines[1:]
        raw2label = {}
        for i in range(len(lines)):
            label_classes_set = set(scannet_labels)
            elements = lines[i].split('\t')
            raw_name = elements[1]
            nyu40_name = elements[7]
            if nyu40_name not in label_classes_set:
                raw2label[raw_name] = scannet2label['others']
            else:
                raw2label[raw_name] = scannet2label[nyu40_name]

        return raw2label

    def _get_unique_multiple_lookup(self):
        all_sem_labels = {}
        cache = {}
        for data in self.scanrefer:
            scene_id = data["scene_id"]
            object_id = data["object_id"]
            object_name = " ".join(data["object_name"].split("_"))
            ann_id = data["ann_id"]

            if scene_id not in all_sem_labels:
                all_sem_labels[scene_id] = []

            if scene_id not in cache:
                cache[scene_id] = {}

            if object_id not in cache[scene_id]:
                cache[scene_id][object_id] = {}
                try:
                    all_sem_labels[scene_id].append(self.raw2label[object_name])
                except KeyError:
                    all_sem_labels[scene_id].append(17)

        # convert to numpy array
        all_sem_labels = {scene_id: np.array(all_sem_labels[scene_id]) for scene_id in all_sem_labels.keys()}

        unique_multiple_lookup = {}
        for data in self.scanrefer:
            scene_id = data["scene_id"]
            object_id = data["object_id"]
            object_name = " ".join(data["object_name"].split("_"))
            ann_id = data["ann_id"]

            try:
                sem_label = self.raw2label[object_name]
            except KeyError:
                sem_label = 17

            unique_multiple = 0 if (all_sem_labels[scene_id] == sem_label).sum() == 1 else 1

            # store
            if scene_id not in unique_multiple_lookup:
                unique_multiple_lookup[scene_id] = {}

            if object_id not in unique_multiple_lookup[scene_id]:
                unique_multiple_lookup[scene_id][object_id] = {}

            if ann_id not in unique_multiple_lookup[scene_id][object_id]:
                unique_multiple_lookup[scene_id][object_id][ann_id] = None

            unique_multiple_lookup[scene_id][object_id][ann_id] = unique_multiple

        return unique_multiple_lookup

    def _tranform_des(self):
        lang = {}
        label = {}
        for data in self.scanrefer:
            scene_id = data["scene_id"]
            object_id = data["object_id"]
            ann_id = data["ann_id"]

            if object_id != "SYNTHETIC":

                if scene_id not in lang:
                    lang[scene_id] = {}
                    label[scene_id] = {}

                if object_id not in lang[scene_id]:
                    lang[scene_id][object_id] = {}
                    label[scene_id][object_id] = {}

                if ann_id not in lang[scene_id][object_id]:
                    lang[scene_id][object_id][ann_id] = {}
                    label[scene_id][object_id][ann_id] = {}

                # trim long descriptions
                tokens = data["token"][:CONF.TRAIN.MAX_DES_LEN]

                # tokenize the description
                tokens = [self.vocabulary["special_tokens"]["bos_token"]] + tokens + [self.vocabulary["special_tokens"]["eos_token"]]
                embeddings = np.zeros((CONF.TRAIN.MAX_DES_LEN + 2, 300))
                labels = np.zeros((CONF.TRAIN.MAX_DES_LEN + 2)) # start and end

                # embeddings = np.zeros((CONF.TRAIN.MAX_DES_LEN, 300))
                # labels = np.zeros((CONF.TRAIN.MAX_DES_LEN)) # start and end

                # load
                for token_id in range(len(tokens)):
                    token = tokens[token_id] 
                    if token not in self.vocabulary["word2idx"]: token = self.vocabulary["special_tokens"]["unk_token"]

                    if token_id < CONF.TRAIN.MAX_DES_LEN + 2:
                        labels[token_id] = self.vocabulary["word2idx"][token]

                    try:
                        glove_id = int(self.vocabulary["word2idx"][token])                    
                        embeddings[token_id] = self.glove[glove_id]
                    except KeyError:
                        glove_id = int(self.vocabulary["word2idx"][self.vocabulary["special_tokens"]["unk_token"]])                    
                        embeddings[token_id] = self.glove[glove_id]

                # store
                lang[scene_id][object_id][ann_id] = embeddings
                label[scene_id][object_id][ann_id] = labels

        return lang, label

    def _tranform_des_with_erase(self, lang_feat, lang_len, p=0.2):
        num_erase = int((lang_len - 2) * p)
        erase_ids = np.arange(1, lang_len - 2, 1).tolist()
        erase_ids = np.random.choice(erase_ids, num_erase, replace=False) # randomly pick indices of erased tokens
        
        unk_idx = int(self.vocabulary["word2idx"][self.vocabulary["special_tokens"]["unk_token"]])
        unk = self.glove[unk_idx] # 300
        unk_exp = unk.reshape((1, -1)).repeat(erase_ids.shape[0], axis=0)

        lang_feat[erase_ids] = unk_exp
        
        return lang_feat

    def _build_vocabulary(self):
        if self.use_bert_vocab:
            vocab_path = VOCAB.format("BERT")
            print("vocabulary path: {}".format(vocab_path))
            if os.path.exists(vocab_path):
                print("loading vocabulary...")
                vocabulary = json.load(open(vocab_path))
            else:
                if self.split == "train":
                    print("building vocabulary...")
                    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
                    word2idx = tokenizer.get_vocab() # str -> idx
                    idx2word = {v: k for k, v in word2idx.items()}
                    speical_tokens = {
                        "bos_token": "[CLS]",
                        "eos_token": "[SEP]",
                        "unk_token": "[UNK]",
                        "pad_token": "[PAD]"
                    }
                    vocabulary = {
                        "word2idx": word2idx,
                        "idx2word": idx2word,
                        "special_tokens": speical_tokens
                    }
                    json.dump(vocabulary, open(vocab_path, "w"), indent=4)

            emb_mat_path = GLOVE_PATH.format("BERT")
            if os.path.exists(emb_mat_path):
                embeddings = np.load(emb_mat_path)
            else:
                all_glove = pickle.load(open(GLOVE_PICKLE, "rb"))

                spw_mappings = {
                    "[CLS]": "sos",
                    "[SEP]": "eos",
                    "[UNK]": "unk",
                    "[PAD]": "pad_",
                }

                embeddings = np.zeros((len(vocabulary["word2idx"]), 300))
                for word, idx in vocabulary["word2idx"].items():
                    if word in vocabulary["special_tokens"].values():
                        if spw_mappings[word] != "pad_": 
                            emb = all_glove[spw_mappings[word]]
                        else:
                            emb = np.zeros((300,))
                    else:
                        try:
                            emb = all_glove[word]
                        except KeyError:
                            emb = all_glove["unk"]

                    embeddings[int(idx)] = emb

                np.save(emb_mat_path, embeddings)

        else:
            vocab_path = VOCAB.format(self.name)
            print("vocabulary path: {}".format(vocab_path))
            if os.path.exists(vocab_path):
                print("loading vocabulary...")
                vocabulary = json.load(open(vocab_path))
            else:
                if self.split == "train":
                    print("building vocabulary...")
                    glove = pickle.load(GLOVE_PICKLE)
                    train_data = [d for d in self.scanrefer if d["object_id"] != "SYNTHETIC"]
                    all_words = chain(*[data["token"][:CONF.TRAIN.MAX_DES_LEN] for data in train_data])
                    word_counter = Counter(all_words)
                    word_counter = sorted([(k, v) for k, v in word_counter.items() if k in glove], key=lambda x: x[1], reverse=True)
                    word_list = [k for k, _ in word_counter]

                    # build vocabulary
                    word2idx, idx2word = {}, {}
                    spw = ["pad_", "unk", "sos", "eos"] # NOTE distinguish padding token "pad_" and the actual word "pad"
                    for i, w in enumerate(word_list):
                        shifted_i = i + len(spw)
                        word2idx[w] = shifted_i
                        idx2word[shifted_i] = w

                    # add special words into vocabulary
                    for i, w in enumerate(spw):
                        word2idx[w] = i
                        idx2word[i] = w

                    speical_tokens = {
                        "bos_token": "sos",
                        "eos_token": "eos",
                        "unk_token": "unk",
                        "pad_token": "pad_"
                    }
                    vocabulary = {
                        "word2idx": word2idx,
                        "idx2word": idx2word,
                        "special_tokens": speical_tokens
                    }
                    json.dump(vocabulary, open(vocab_path, "w"), indent=4)


            emb_mat_path = GLOVE_PATH.format(self.name)
            if os.path.exists(emb_mat_path):
                embeddings = np.load(emb_mat_path)
            else:
                all_glove = pickle.load(open(GLOVE_PICKLE, "rb"))

                embeddings = np.zeros((len(vocabulary["word2idx"]), 300))
                for word, idx in vocabulary["word2idx"].items():
                    try:
                        emb = all_glove[word]
                    except KeyError:
                        emb = all_glove["unk"]
                        
                    embeddings[int(idx)] = emb

                np.save(emb_mat_path, embeddings)

        return vocabulary, embeddings

    def _load_data(self):
        print("loading data...")
        # load language features
        self.vocabulary, self.glove = self._build_vocabulary()
        self.lang, self.lang_ids = self._tranform_des()

        # add scannet data
        self.scene_list = sorted(list(set([data["scene_id"] for data in self.scanrefer])))

        # load scene data
        self.scene_data = {}
        for scene_id in self.scene_list:
            self.scene_data[scene_id] = {}
            # self.scene_data[scene_id]["mesh_vertices"] = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id)+"_vert.npy")
            self.scene_data[scene_id]["mesh_vertices"] = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id)+"_aligned_vert.npy") # axis-aligned
            self.scene_data[scene_id]["instance_labels"] = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id)+"_ins_label.npy")
            self.scene_data[scene_id]["semantic_labels"] = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id)+"_sem_label.npy")
            # self.scene_data[scene_id]["instance_bboxes"] = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id)+"_bbox.npy")
            self.scene_data[scene_id]["instance_bboxes"] = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id)+"_aligned_bbox.npy")

        # prepare class mapping
        lines = [line.rstrip() for line in open(SCANNET_V2_TSV)]
        lines = lines[1:]
        raw2nyuid = {}
        for i in range(len(lines)):
            elements = lines[i].split('\t')
            raw_name = elements[1]
            nyu40_name = int(elements[4])
            raw2nyuid[raw_name] = nyu40_name

        # store
        self.raw2nyuid = raw2nyuid
        self.raw2label = self._get_raw2label()
        self.unique_multiple_lookup = self._get_unique_multiple_lookup()
        

    def _translate(self, point_set, bbox):
        # unpack
        coords = point_set[:, :3]

        # translation factors
        x_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
        y_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
        z_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
        factor = [x_factor, y_factor, z_factor]
        
        # dump
        coords += factor
        point_set[:, :3] = coords
        bbox[:, :3] += factor

        return point_set, bbox, factor
