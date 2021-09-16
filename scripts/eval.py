# HACK ignore warnings
import warnings

from torch.utils import data
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import sys
import json
import torch
import argparse

import numpy as np

from tqdm import tqdm
from copy import deepcopy
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.getcwd())) # HACK add the root folder

import lib.capeval.bleu.bleu as capblue
import lib.capeval.cider.cider as capcider
import lib.capeval.rouge.rouge as caprouge

from data.scannet.model_util_scannet import ScannetDatasetConfig
from lib.dataset import ScannetReferenceDataset
from lib.config import CONF
from lib.ap_helper import APCalculator, parse_predictions, parse_groundtruths
from lib.loss_helper import get_loss
from models.speaker import SpeakerNet
from lib.eval_helper import eval_caption_step, eval_caption_epoch

# extracted ScanNet object rotations from Scan2CAD 
# NOTE some scenes are missing in this annotation!!!
SCAN2CAD_ROTATION = json.load(open(os.path.join(CONF.PATH.SCAN2CAD, "scannet_instance_rotations.json")))

# constants
DC = ScannetDatasetConfig()

def get_dataloader(args, scanrefer, all_scene_list, config, augment=False, scan2cad_rotation=SCAN2CAD_ROTATION):
    dataset = ScannetReferenceDataset(
        scanrefer=scanrefer, 
        scanrefer_all_scene=all_scene_list,  
        split="val",
        name=args.dataset,
        num_points=args.num_points, 
        use_height=(not args.no_height),
        use_color=args.use_color, 
        use_normal=args.use_normal, 
        use_multiview=args.use_multiview,
        augment=augment,
        scan2cad_rotation=scan2cad_rotation
    )
    # dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)

    return dataset, dataloader

def get_model(args, dataset, root):
    # initiate model
    input_channels = int(args.use_multiview) * 128 + int(args.use_normal) * 3 + int(args.use_color) * 3 + int(not args.no_height)

    model = SpeakerNet(
        dataset=dataset,
        root=root,
        num_class=DC.num_class,
        num_heading_bin=DC.num_heading_bin,
        num_size_cluster=DC.num_size_cluster,
        mean_size_arr=DC.mean_size_arr,
        input_feature_dim=input_channels,
        num_proposal=args.num_proposals,
        no_caption=args.no_caption,
        use_topdown=args.use_topdown,
        num_locals=args.num_locals,
        query_mode=args.query_mode,
        graph_mode=args.graph_mode,
        num_graph_steps=args.num_graph_steps,
        use_relation=args.use_relation,
        use_orientation=args.use_orientation,
        use_distance=args.use_distance,
        beam_opt={
            "train_beam_size": args.train_beam_size,
            "train_sample_topn": args.train_sample_topn,
            "eval_beam_size": args.eval_beam_size
        }
    )

    if args.eval_pretrained:
        # load pretrained model
        print("loading pretrained VoteNet...")
        pretrained_model = SpeakerNet(
            dataset=dataset,
            root=root,
            num_class=DC.num_class,
            num_heading_bin=DC.num_heading_bin,
            num_size_cluster=DC.num_size_cluster,
            mean_size_arr=DC.mean_size_arr,
            num_proposal=args.num_proposals,
            input_feature_dim=input_channels,
            no_caption=True
        )

        pretrained_name = "PRETRAIN_VOTENET_XYZ"
        if args.use_color: pretrained_name += "_COLOR"
        if args.use_multiview: pretrained_name += "_MULTIVIEW"
        if args.use_normal: pretrained_name += "_NORMAL"

        pretrained_path = os.path.join(CONF.PATH.PRETRAINED, pretrained_name, "model.pth")
        pretrained_model.load_state_dict(torch.load(pretrained_path), strict=False)

        # mount
        model.backbone_net = pretrained_model.backbone_net
        model.vgen = pretrained_model.vgen
        model.proposal = pretrained_model.proposal
    else:
        # load
        model_name = "model.ckpt"
        model_path = os.path.join(CONF.PATH.OUTPUT, root, model_name)
        # model.load_state_dict(torch.load(model_path), strict=False)
        model = model.load_from_checkpoint(
            model_path,
            dataset=dataset,
            root=root,
            num_class=DC.num_class,
            num_heading_bin=DC.num_heading_bin,
            num_size_cluster=DC.num_size_cluster,
            mean_size_arr=DC.mean_size_arr,
            input_feature_dim=input_channels,
            num_proposal=args.num_proposals,
            no_caption=args.no_caption,
            use_topdown=args.use_topdown,
            num_locals=args.num_locals,
            query_mode=args.query_mode,
            graph_mode=args.graph_mode,
            num_graph_steps=args.num_graph_steps,
            use_relation=args.use_relation,
            use_orientation=args.use_orientation,
            use_distance=args.use_distance,
            beam_opt={
                "train_beam_size": args.train_beam_size,
                "train_sample_topn": args.train_sample_topn,
                "eval_beam_size": args.eval_beam_size
            }
        )

    model.cuda()
    
    # set mode
    model.eval()

    return model

def get_scannet_scene_list(data):
    # scene_list = sorted([line.rstrip() for line in open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_{}.txt".format(split)))])
    scene_list = sorted(list(set([d["scene_id"] for d in data])))

    return scene_list

def get_eval_data(args):
    if args.dataset == "ScanRefer":
        scanrefer_train = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_train.json")))
        scanrefer_val = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_val.json")))
    elif args.dataset == "ReferIt3D":
        scanrefer_train = json.load(open(os.path.join(CONF.PATH.DATA, "nr3d_train.json")))
        scanrefer_val = json.load(open(os.path.join(CONF.PATH.DATA, "nr3d_val.json")))
    else:
        raise ValueError("Invalid dataset.")

    eval_scene_list = get_scannet_scene_list(scanrefer_train) if args.use_train else get_scannet_scene_list(scanrefer_val)
    scanrefer_eval = []
    for scene_id in eval_scene_list:
        data = deepcopy(scanrefer_val[0])
        data["scene_id"] = scene_id
        scanrefer_eval.append(data)

    print("eval on {} samples".format(len(scanrefer_eval)))

    return scanrefer_eval, eval_scene_list

def eval_caption(args, root=CONF.PATH.OUTPUT):
    print("evaluate captioning...")

    print("initializing...")
    folder = os.path.join(CONF.name.upper(), CONF.tag.upper())

    # get eval data
    scanrefer_eval, eval_scene_list = get_eval_data(args)

    # get dataloader
    dataset, dataloader = get_dataloader(args, scanrefer_eval, eval_scene_list, DC)

    # get model
    model = get_model(args, dataset, folder)

    # evaluate
    min_ious = CONF.min_ious.split(",")
    for min_iou in min_ious:
        print("evaluating for min IOU {}".format(min_iou))
        
        # check if predictions exist
        pred_path = os.path.join(CONF.PATH.OUTPUT, folder, "pred_{}_{}.json".format("val", str(model.device)))
        if min_iou == min_ious[0] or not os.path.exists(pred_path):
        # if not os.path.exists(pred_path):
            print("generating predictions for min IOU {}".format(min_iou))
            outputs = []
            for data_dict in tqdm(dataloader):
                # to device
                for key in data_dict:
                    data_dict[key] = data_dict[key].cuda()

                # feed
                with torch.no_grad():
                    data_dict = model.forward(
                        data_dict=data_dict,
                        use_tf=False,
                        is_eval=True,
                        beam_opt={
                            "train_beam_size": args.train_beam_size,
                            "train_sample_topn": args.train_sample_topn,
                            "eval_beam_size": args.eval_beam_size
                        }
                    )
                    # loss
                    _, data_dict = get_loss(
                        data_dict=data_dict,
                        config=DC,
                        detection=True,
                        caption=False,
                        orientation=False,
                        distance=False
                    )

                outs = eval_caption_step(
                    data_dict=data_dict,
                    dataset=dataset,
                    detection=True
                )

                # store
                outputs.append(outs)

            # aggregate
            candidates = {}
            for outs in outputs:
                for key, value in outs.items():
                    if key not in candidates:
                        candidates[key] = value

            # store
            with open(pred_path, "w") as f:
                json.dump(candidates, f, indent=4)

        else:
            print("loading predictions for min IoU {}".format(min_iou))
            with open(pred_path) as f:
                candidates = json.load(f)

        # evaluate
        print("computing scores...")
        bleu, cider, rouge, meteor = eval_caption_epoch(
            candidates=candidates,
            dataset=dataset,
            folder=folder,
            device=model.device,
            min_iou=float(min_iou)
        )

    # report
    print("\n----------------------Evaluation-----------------------")
    print("[BLEU-1] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(bleu[0][0], max(bleu[1][0]), min(bleu[1][0])))
    print("[BLEU-2] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(bleu[0][1], max(bleu[1][1]), min(bleu[1][1])))
    print("[BLEU-3] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(bleu[0][2], max(bleu[1][2]), min(bleu[1][2])))
    print("[BLEU-4] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(bleu[0][3], max(bleu[1][3]), min(bleu[1][3])))
    print("[CIDEr] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(cider[0], max(cider[1]), min(cider[1])))
    print("[ROUGE-L] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(rouge[0], max(rouge[1]), min(rouge[1])))
    print("[METEOR] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(meteor[0], max(meteor[1]), min(meteor[1])))
    print()

def eval_detection(args, root=CONF.PATH.OUTPUT):
    print("evaluate detection...")
    folder = os.path.join(root, CONF.name.upper(), CONF.tag.upper())
    
    # init training dataset
    print("preparing data...")
    # get eval data
    scanrefer_eval, eval_scene_list = get_eval_data(args)

    # get dataloader
    dataset, dataloader = get_dataloader(args, scanrefer_eval, eval_scene_list, DC)

    # model
    print("initializing...")
    model = get_model(args, dataset, folder)

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
    AP_IOU_THRESHOLDS = [0.25, 0.5]
    AP_CALCULATOR_LIST = [APCalculator(iou_thresh, DC.class2type) for iou_thresh in AP_IOU_THRESHOLDS]

    sem_acc = []
    for data_dict in tqdm(dataloader):
        for key in data_dict:
            data_dict[key] = data_dict[key].cuda()

        # feed
        with torch.no_grad():
            data_dict = model.forward(data_dict)
            _, data_dict = get_loss(
                data_dict=data_dict,
                config=DC,
                detection=True,
                caption=False,
                orientation=False,
                distance=False
            )

        batch_pred_map_cls = parse_predictions(data_dict, POST_DICT) 
        batch_gt_map_cls = parse_groundtruths(data_dict, POST_DICT) 
        for ap_calculator in AP_CALCULATOR_LIST:
            ap_calculator.step(batch_pred_map_cls, batch_gt_map_cls)

    # aggregate object detection results and report
    for i, ap_calculator in enumerate(AP_CALCULATOR_LIST):
        print()
        print("-"*10, "iou_thresh: %f"%(AP_IOU_THRESHOLDS[i]), "-"*10)
        metrics_dict = ap_calculator.compute_metrics()
        for key in metrics_dict:
            print("eval %s: %f"%(key, metrics_dict[key]))

if __name__ == "__main__":
    # setting
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # reproducibility
    torch.manual_seed(CONF.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(CONF.seed)

    # evaluate
    if CONF.eval_caption: eval_caption(CONF)
    if CONF.eval_detection: eval_detection(CONF)

