from lib.config import CONF
import os
import sys
import torch

import numpy as np
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

sys.path.append(os.path.join(os.getcwd())) # HACK add the lib folder

from data.scannet.model_util_scannet import ScannetDatasetConfig

from models.backbone_module import Pointnet2Backbone
from models.voting_module import VotingModule
from models.proposal_module import ProposalModule
from models.graph_module import GraphModule
from models.caption_module import TopDownSceneCaptionModule

from lib.loss_helper import get_loss
from lib.eval_helper import eval_caption_step, eval_caption_epoch

DC = ScannetDatasetConfig()

class SpeakerNet(pl.LightningModule):
    def __init__(self, dataset, root, num_class, num_heading_bin, num_size_cluster, mean_size_arr, 
    mode="normal", det_channel=128, use_contextual_aggregation=False,
    input_feature_dim=0, num_proposal=256, num_locals=-1, vote_factor=1, sampling="vote_fps",
    no_caption=False, use_topdown=False, query_mode="corner", 
    graph_mode="graph_conv", num_graph_steps=0, use_relation=False, graph_aggr="add",
    use_orientation=False, num_bins=6, use_distance=False, min_num_pts=0,
    emb_size=300, hidden_size=512, beam_opt={}):
        super().__init__()

        self.dataset = dataset
        self.root = root

        self.mode = mode # normal or gt
        self.det_channel = det_channel
        self.use_contextual_aggregation = use_contextual_aggregation

        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        assert(mean_size_arr.shape[0] == self.num_size_cluster)
        self.input_feature_dim = input_feature_dim
        self.num_proposal = 128 if mode == "gt" else num_proposal
        self.vote_factor = vote_factor
        self.sampling = sampling
        self.no_caption = no_caption
        self.num_graph_steps = num_graph_steps
        self.use_orientation = use_orientation
        self.use_distance = use_distance

        self.beam_opt = beam_opt

        # --------- PROPOSAL GENERATION ---------
        if self.mode != "gt": # if not using GT data
            # Backbone point feature learning
            self.backbone_net = Pointnet2Backbone(input_feature_dim=self.input_feature_dim)

            # Hough voting
            self.vgen = VotingModule(self.vote_factor, 256)

            # Vote aggregation and object proposal
            self.proposal = ProposalModule(num_class, num_heading_bin, num_size_cluster, mean_size_arr, self.num_proposal, sampling, 
                use_contextual_aggregation=use_contextual_aggregation)

        if use_relation: assert not no_caption # only enable use_relation in topdown captioning module

        if num_graph_steps > 0:
            self.graph = GraphModule(128, 128, num_graph_steps, self.num_proposal, 128, num_locals, 
                query_mode, graph_mode, return_edge=use_relation, graph_aggr=graph_aggr, 
                return_orientation=use_orientation, num_bins=num_bins, return_distance=use_distance)

        # Caption generation
        if not no_caption:
            self.caption = TopDownSceneCaptionModule(dataset.vocabulary, dataset.glove, emb_size, det_channel, 
                hidden_size, self.num_proposal, num_locals, query_mode, min_num_pts, use_relation, use_oracle=(mode=="gt"))

    def training_step(self, data_dict, idx):
        # forward pass
        data_dict = self.forward(data_dict)

        # loss
        loss, data_dict = get_loss(
            data_dict=data_dict,
            config=DC,
            detection=self.mode != "gt",
            caption=not self.no_caption,
            orientation=self.use_orientation,
            distance=self.use_distance
        )

        # unpack
        log_dict = {
            "loss": loss,

            "vote_loss": data_dict["vote_loss"],
            "objectness_loss": data_dict["objectness_loss"],
            "box_loss": data_dict["box_loss"],
            "sem_cls_loss": data_dict["sem_cls_loss"],

            "cap_loss": data_dict["cap_loss"],

            "ori_loss": data_dict["ori_loss"],

            "cap_acc": data_dict["cap_acc"],
            "ori_acc": data_dict["ori_acc"],

            "pred_ious": data_dict["pred_ious"],
        }

        # log
        in_prog_bar = ["cap_acc"]
        for key, value in log_dict.items():
            ctg = "loss" if "loss" in key else "score"
            self.log("train_{}/{}".format(ctg, key), value, prog_bar=(key in in_prog_bar), on_step=True, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, data_dict, idx):
        # forward pass
        data_dict = self.forward(data_dict, use_tf=False, is_eval=True, beam_opt=self.beam_opt)

        # loss
        loss, data_dict = get_loss(
            data_dict=data_dict,
            config=DC,
            detection=self.mode != "gt",
            caption=False,
            orientation=False,
            distance=False
        )

        # eval
        candidates = eval_caption_step(
            data_dict=data_dict,
            dataset=self.dataset,
            detection=(self.mode != "gt")
        )

        return candidates

    def validation_epoch_end(self, outputs):
        # aggregate captioning outputs
        candidates = {}
        for outs in outputs:
            for key, value in outs.items():
                if key not in candidates:
                    candidates[key] = value

        # evaluate captions
        bleu, cider, rouge, meteor = eval_caption_epoch(
            candidates=candidates,
            dataset=self.dataset,
            folder=self.root,
            device=self.device,
            phase="val"
        )

        log_dict = {
            "bleu-1": bleu[0][0],
            "bleu-2": bleu[0][1],
            "bleu-3": bleu[0][2],
            "bleu-4": bleu[0][3],
            "cider": cider[0],
            "meteor": meteor[0],
            "rouge": rouge[0],
        }

        # log
        for key, value in log_dict.items():
            self.log("eval/{}".format(key), value, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=CONF.lr, weight_decay=CONF.wd)

        # learning rate scheduler
        if self.no_caption:
            LR_DECAY_STEP = [80, 120, 160]
            LR_DECAY_RATE = 0.1
            
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, LR_DECAY_STEP, LR_DECAY_RATE)

            return [optimizer], [scheduler]
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100)

            return [optimizer], [scheduler]

    # NOTE direct access only during inference
    def forward(self, data_dict, use_tf=True, use_rl=False, is_eval=False, beam_opt={}):
        """ Forward pass of the network

        Args:
            data_dict: dict
                {
                    point_clouds, 
                    lang_feat
                }

                point_clouds: Variable(torch.cuda.FloatTensor)
                    (B, N, 3 + input_channels) tensor
                    Point cloud to run predicts on
                    Each point in the point-cloud MUST
                    be formated as (x, y, z, features...)
        Returns:
            end_points: dict
        """

        #######################################
        #                                     #
        #           DETECTION BRANCH          #
        #                                     #
        #######################################

        if self.mode != "gt":

            data_dict["point_clouds"] = data_dict["point_clouds"]
        
            # --------- HOUGH VOTING ---------
            data_dict = self.backbone_net(data_dict)
                    
            # --------- HOUGH VOTING ---------
            xyz = data_dict["fp2_xyz"]
            features = data_dict["fp2_features"]
            data_dict["seed_inds"] = data_dict["fp2_inds"]
            data_dict["seed_xyz"] = xyz
            data_dict["seed_features"] = features
            
            xyz, features = self.vgen(xyz, features)
            features_norm = torch.norm(features, p=2, dim=1)
            features = features.div(features_norm.unsqueeze(1))
            data_dict["vote_xyz"] = xyz
            data_dict["vote_features"] = features

            # --------- PROPOSAL GENERATION ---------
            data_dict = self.proposal(xyz, features, data_dict)
        else:
            data_dict["bbox_feature"] = data_dict["bbox_feature"]
            data_dict["bbox_corner"] = data_dict["bbox_corner"]
            data_dict["center"] = data_dict["bbox_center"]
            data_dict["bbox_mask"] = data_dict["bbox_mask"]

        #######################################
        #                                     #
        #           GRAPH ENHANCEMENT         #
        #                                     #
        #######################################

        if self.num_graph_steps > 0: 
            data_dict = self.graph(data_dict)

        #######################################
        #                                     #
        #            CAPTION BRANCH           #
        #                                     #
        #######################################

        # --------- CAPTION GENERATION ---------
        if not self.no_caption:
            data_dict = self.caption(data_dict, use_tf, use_rl, is_eval, beam_opt)

        return data_dict
