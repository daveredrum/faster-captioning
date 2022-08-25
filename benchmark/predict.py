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

from data.scannet.model_util_scannet import ScannetDatasetConfig
from lib.dataset import ScannetReferenceDataset
from lib.config import CONF
from lib.ap_helper import parse_predictions
from models.speaker import SpeakerNet

# extracted ScanNet object rotations from Scan2CAD 
# NOTE some scenes are missing in this annotation!!!
SCAN2CAD_ROTATION = json.load(open(os.path.join(CONF.PATH.SCAN2CAD, "scannet_instance_rotations.json")))

# constants
DC = ScannetDatasetConfig()

# post-process
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

def get_dataloader(args, scanrefer, all_scene_list, config, augment=False, scan2cad_rotation=SCAN2CAD_ROTATION):
    dataset = ScannetReferenceDataset(
        scanrefer=scanrefer, 
        scanrefer_all_scene=all_scene_list,  
        split="test",
          name=args.dataset,
        num_points=args.num_points, 
        use_height=(not args.no_height),
        use_color=args.use_color, 
        use_normal=args.use_normal, 
        use_multiview=args.use_multiview,
        augment=augment,
        scan2cad_rotation=scan2cad_rotation,
        use_bert_vocab=args.use_bert_vocab
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
        use_contextual_aggregation=args.use_contextual_aggregation,
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
            use_contextual_aggregation=args.use_contextual_aggregation,
            no_caption=True
        )

        pretrained_name = "PRETRAIN_VOTENET_XYZ"
        if args.use_color: pretrained_name += "_COLOR"
        if args.use_multiview: pretrained_name += "_MULTIVIEW"
        if args.use_normal: pretrained_name += "_NORMAL"

        # pretrained_path = os.path.join(CONF.PATH.PRETRAINED, pretrained_name, "model.pth")
        # pretrained_model.load_state_dict(torch.load(pretrained_path), strict=False)

        pretrained_path = os.path.join(CONF.PATH.PRETRAINED, pretrained_name, "model.ckpt")
        pretrained_model.load_from_checkpoint(
            pretrained_path,
            # strict=False,
            dataset=dataset,
            root=root,
            num_class=DC.num_class,
            num_heading_bin=DC.num_heading_bin,
            num_size_cluster=DC.num_size_cluster,
            mean_size_arr=DC.mean_size_arr,
            num_proposal=args.num_proposals,
            input_feature_dim=input_channels,
            no_caption=True,
            use_contextual_aggregation=args.use_contextual_aggregation
        )

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
            strict=False,
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
            use_contextual_aggregation=args.use_contextual_aggregation,
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

def get_scannet_scene_list(split):
    scene_list = sorted([line.rstrip() for line in open(os.path.join(CONF.PATH.SCANNET_META, "scannetv2_{}.txt".format(split)))])
    scene_list = [s for s in scene_list if s.split("_")[-1] == "00"]

    return scene_list

def get_eval_data(args):
    scanrefer_test = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_{}.json".format(args.test_split))))

    eval_scene_list = get_scannet_scene_list(args.test_split)
    scanrefer_eval = []
    for scene_id in eval_scene_list:
        data = deepcopy(scanrefer_test[0])
        data["scene_id"] = scene_id
        scanrefer_eval.append(data)

    print("test on {} samples".format(len(scanrefer_eval)))

    return scanrefer_eval, eval_scene_list

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

def predict_caption(args, root=CONF.PATH.OUTPUT):
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
    print("evaluating...")
    outputs = {}
    for data_dict in tqdm(dataloader):
        # to device
        for key in data_dict:
            if isinstance(data_dict[key], list): continue
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

        pred_captions = data_dict["lang_cap"]
        pred_boxes = data_dict["bbox_corner"]

        # nms mask
        _ = parse_predictions(data_dict, POST_DICT)
        nms_masks = torch.FloatTensor(data_dict["pred_mask"]).type_as(pred_boxes).long()

        # objectness mask
        obj_masks = torch.argmax(data_dict["objectness_scores"], 2).long()

        # final mask
        nms_masks = nms_masks * obj_masks

        # nms_masks = torch.ones(pred_boxes.shape[0], pred_boxes.shape[1]).type_as(pred_boxes)

        # for object detection
        pred_sem_prob = torch.softmax(data_dict['sem_cls_scores'], dim=-1) # B, num_proposal, num_cls
        pred_obj_prob = torch.softmax(data_dict['objectness_scores'], dim=-1) # B, num_proposal, 2

        for batch_id in range(pred_captions.shape[0]):
            scene_id = data_dict["scene_id"][batch_id]
            scene_outputs = []
            for object_id in range(pred_captions.shape[1]):
                if nms_masks[batch_id, object_id] == 1:
                    caption = decode_caption(pred_captions[batch_id, object_id], dataset.vocabulary["idx2word"], dataset.vocabulary["special_tokens"]) 
                    box = pred_boxes[batch_id, object_id].cpu().detach().numpy().tolist()

                    sem_prob = pred_sem_prob[batch_id, object_id].cpu().detach().numpy().tolist()
                    obj_prob = pred_obj_prob[batch_id, object_id].cpu().detach().numpy().tolist()

                    scene_outputs.append(
                        {
                            "caption": caption,
                            "box": box,
                            "sem_prob": sem_prob,
                            "obj_prob": obj_prob
                        }
                    )

            outputs[scene_id] = scene_outputs

    # dump
    save_path = os.path.join(CONF.PATH.OUTPUT, folder, "benchmark_{}.json".format(args.test_split))
    with open(save_path, "w") as f:
        json.dump(outputs, f, indent=4)

    print("done!")

if __name__ == "__main__":
    # setting
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # reproducibility
    torch.manual_seed(CONF.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(CONF.seed)

    # evaluate
    predict_caption(CONF)

