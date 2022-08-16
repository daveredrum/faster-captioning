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
from lib.ap_helper import APCalculator, parse_predictions, parse_groundtruths
from lib.pointnet2.pytorch_utils import BNMomentumScheduler

DC = ScannetDatasetConfig()
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
AP_CALCULATOR = APCalculator(0.5, DC.class2type)

class SpeakerNet(pl.LightningModule):
    def __init__(self, dataset, root, num_class, num_heading_bin, num_size_cluster, mean_size_arr, 
    mode="normal", det_channel=128, use_contextual_aggregation=False,
    input_feature_dim=0, num_proposal=256, num_locals=-1, vote_factor=1, sampling="vote_fps",
    no_caption=False, use_topdown=False, query_mode="corner", 
    graph_mode="graph_conv", num_graph_steps=0, use_relation=False, graph_aggr="add",
    use_orientation=False, num_bins=6, use_distance=False, min_num_pts=0,
    emb_size=300, hidden_size=512, beam_opt={},
    post_dict=POST_DICT, ap_calculator=AP_CALCULATOR):
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
        self.post_dict = post_dict
        self.ap_calculator = ap_calculator

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

    def on_train_epoch_start(self):
        # update lr scheduler
        if self.lr_scheduler:
            # print("current learning rate --> {}\n".format(self.lr_scheduler.get_last_lr()))
            self.lr_scheduler.step()

        # update bn scheduler
        if self.bn_scheduler:
            # print("current batch normalization momentum --> {}\n".format(self.bn_scheduler.lmbd(self.bn_scheduler.last_epoch)))
            self.bn_scheduler.step()

        return super().on_train_epoch_start()

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
        _, data_dict = get_loss(
            data_dict=data_dict,
            config=DC,
            detection=self.mode != "gt",
            caption=False,
            orientation=False,
            distance=False
        )

        # eval detection
        batch_pred_map_cls = parse_predictions(data_dict, self.post_dict) 
        batch_gt_map_cls = parse_groundtruths(data_dict, self.post_dict) 
        self.ap_calculator.step(batch_pred_map_cls, batch_gt_map_cls)

        # eval caption
        if self.no_caption:
            candidates = []
        else:
            candidates = eval_caption_step(
                data_dict=data_dict,
                dataset=self.dataset,
                detection=(self.mode != "gt")
            )

        return candidates

    def validation_epoch_end(self, outputs):
        if self.no_caption:
            log_dict = {
                "bleu-1": 0,
                "bleu-2": 0,
                "bleu-3": 0,
                "bleu-4": 0,
                "cider": 0,
                "meteor": 0,
                "rouge": 0
            }
        else:
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
                "rouge": rouge[0]
            }

        # aggregate detection results
        det_metrics = self.ap_calculator.compute_metrics()

        log_dict["mAP@0.5"] = det_metrics["mAP"]

        # log
        for key, value in log_dict.items():
            self.log("eval/{}".format(key), value, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)

        # NOTE self.ap_calculator must be re-intialized after every validation epoch
        self.ap_calculator = APCalculator(0.5, DC.class2type)

    def configure_optimizers(self):
        # optimizer = optim.Adam(self.parameters(), lr=CONF.lr, weight_decay=CONF.wd)
        # optimizer = optim.AdamW(self.parameters(), lr=CONF.lr, weight_decay=CONF.wd)

        backbone_params = self.backbone_net.parameters()
        vgen_params = self.vgen.parameters()

        adam_settings = [
            {"params": backbone_params, "lr": 5e-4},
            {"params": vgen_params, "lr": 5e-4},
        ]

        if self.use_contextual_aggregation:
            proposal_params = set(self.proposal.parameters())
            aggregation_params = set(self.proposal.contextual_aggregation.parameters())
            
            proposal_params = list(proposal_params - aggregation_params)
            aggregation_params = list(aggregation_params)

            adam_settings.append({"params": aggregation_params, "lr": 1e-4}) # make it smaller than other components
        else:
            proposal_params = self.proposal.parameters()

        adam_settings.append({"params": proposal_params, "lr": 5e-4})

        if not self.no_caption:
            graph_params = self.graph.parameters()
            caption_params = self.caption.parameters()

            adam_settings.append({"params": graph_params, "lr": 1e-3})
            adam_settings.append({"params": caption_params, "lr": 1e-3})

        # optimizer = optim.Adam(adam_settings, weight_decay=CONF.wd)
        optimizer = optim.AdamW(adam_settings, weight_decay=CONF.wd)

        # learning rate scheduler
        if self.no_caption:
            LR_DECAY_STEP = [80, 120, 160]
            LR_DECAY_RATE = 0.5
            
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, LR_DECAY_STEP, LR_DECAY_RATE)
        
             # bn scheduler
            START_EPOCH = 0
            BN_DECAY_STEP = 20
            BN_DECAY_RATE = 0.5
            BN_MOMENTUM_INIT = 0.5
            BN_MOMENTUM_MAX = 0.001
            bn_lbmd = lambda it: max(BN_MOMENTUM_INIT * BN_DECAY_RATE**(int(it / BN_DECAY_STEP)), BN_MOMENTUM_MAX)
            self.bn_scheduler = BNMomentumScheduler(self, bn_lambda=bn_lbmd, last_epoch=START_EPOCH-1)

        else:
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100)
            # self.lr_scheduler = None
            self.bn_scheduler = None

        return [optimizer]

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
