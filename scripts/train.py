import os
import sys
import json
import yaml
import torch

import multiprocessing as mp

import torch.optim as optim
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from copy import deepcopy

sys.path.append(os.path.join(os.getcwd())) # HACK add the root folder
from data.scannet.model_util_scannet import ScannetDatasetConfig
from lib.dataset import ScannetReferenceDataset
from lib.config import CONF
from models.speaker import SpeakerNet

# extracted ScanNet object rotations from Scan2CAD 
# NOTE some scenes are missing in this annotation!!!
SCAN2CAD_ROTATION = json.load(open(os.path.join(CONF.PATH.SCAN2CAD, "scannet_instance_rotations.json")))

# constants
DC = ScannetDatasetConfig()

def get_dataloader(args, scanrefer, all_scene_list, split, config, augment, scan2cad_rotation=SCAN2CAD_ROTATION):
    dataset = ScannetReferenceDataset(
        scanrefer=scanrefer, 
        scanrefer_all_scene=all_scene_list, 
        split=split, 
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
    # dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=(split == "train"), pin_memory=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=(split == "train"), num_workers=8, pin_memory=True)

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
        use_contextual_aggregation=args.use_contextual_aggregation,
        use_distance=args.use_distance,
        beam_opt={
            "train_beam_size": args.train_beam_size,
            "train_sample_topn": args.train_sample_topn,
            "eval_beam_size": args.eval_beam_size
        }
    )

    # trainable model
    if args.use_pretrained:
        # load model
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
            no_caption=True,
            use_contextual_aggregation=args.use_contextual_aggregation
        )

        # pretrained_name = "PRETRAIN_VOTENET"
        pretrained_name = "PRETRAIN_VOTENET"
        if args.use_contextual_aggregation: pretrained_name += "_TRANSFORMER"
        pretrained_name += "_XYZ"
        if args.use_color: pretrained_name += "_COLOR"
        if args.use_multiview: pretrained_name += "_MULTIVIEW"
        if args.use_normal: pretrained_name += "_NORMAL"
        
        pretrained_root = os.path.join(CONF.PATH.PRETRAINED, pretrained_name)
        model_name = os.listdir(pretrained_root)[0]
        pretrained_path = os.path.join(pretrained_root, model_name)
        
        if os.path.splitext(model_name)[-1] == ".ckpt":
            model_weights = torch.load(pretrained_path)["state_dict"]
        else:
            model_weights = torch.load(pretrained_path)

        # pretrained_model.load_state_dict(model_weights, strict=False)
        pretrained_model.load_state_dict(model_weights)

        # pretrained_path = os.path.join(CONF.PATH.PRETRAINED, pretrained_name, "model.ckpt")
        # pretrained_model.load_from_checkpoint(
        #     pretrained_path,
        #     # strict=False,
        #     dataset=dataset,
        #     root=root,
        #     num_class=DC.num_class,
        #     num_heading_bin=DC.num_heading_bin,
        #     num_size_cluster=DC.num_size_cluster,
        #     mean_size_arr=DC.mean_size_arr,
        #     num_proposal=args.num_proposals,
        #     input_feature_dim=input_channels,
        #     no_caption=True,
        #     use_contextual_aggregation=args.use_contextual_aggregation
        # )
        # pretrained_model.load_state_dict(torch.load(pretrained_path), strict=False)

        # mount
        model.backbone_net = pretrained_model.backbone_net
        model.vgen = pretrained_model.vgen
        model.proposal = pretrained_model.proposal

        if args.no_detection:
            # freeze pointnet++ backbone
            for param in model.backbone_net.parameters():
                param.requires_grad = False

            # freeze voting
            for param in model.vgen.parameters():
                param.requires_grad = False
            
            # freeze detector
            for param in model.proposal.parameters():
                param.requires_grad = False

    return model

def get_num_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_params = int(sum([np.prod(p.size()) for p in model_parameters]))

    return num_params

def save_info(args, root, num_params, train_dataset, val_dataset):
    info = {}
    for key, value in vars(args).items():
        info[key] = value

    info["num_train"] = len(train_dataset)
    info["num_val"] = len(val_dataset)
    info["num_train_scenes"] = len(train_dataset.scene_list)
    info["num_val_scenes"] = len(val_dataset.scene_list)
    info["num_params"] = num_params

    with open(os.path.join(root, "info.yaml"), "w") as f:
        # json.dump(info, f, indent=4)
        yaml.dump(info, f)

def get_scannet_scene_list(split):
    scene_list = sorted([line.rstrip() for line in open(os.path.join(CONF.PATH.SCANNET_META, "scannetv2_{}.txt".format(split)))])

    return scene_list

def get_scene_organized_data(raw_data):
    new_data = {}
    for data in raw_data:
        scene_id = data["scene_id"]

        if scene_id not in new_data: new_data[scene_id] = []

        new_data[scene_id].append(data)
    
    return new_data

def get_scanrefer(args):
    if args.dataset == "ScanRefer":
        scanrefer_train = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_train.json")))
        scanrefer_val = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_val.json")))
    elif args.dataset == "ReferIt3D":
        scanrefer_train = json.load(open(os.path.join(CONF.PATH.DATA, "nr3d_train.json")))
        scanrefer_val = json.load(open(os.path.join(CONF.PATH.DATA, "nr3d_val.json")))
    else:
        raise ValueError("Invalid dataset.")

    organized_train = get_scene_organized_data(scanrefer_train)
    organized_val = get_scene_organized_data(scanrefer_val)

    if args.no_caption:
        train_scene_list = get_scannet_scene_list("train")
        val_scene_list = get_scannet_scene_list("val")

        # if args.debug: 
        #     train_scene_list = [train_scene_list[0]]
        #     val_scene_list = [val_scene_list[0]]

        new_scanrefer_train = []
        for scene_id in train_scene_list:
            data = deepcopy(scanrefer_train[0])
            data["scene_id"] = scene_id
            new_scanrefer_train.append(data)

        new_scanrefer_val = []
        for scene_id in val_scene_list:
            data = deepcopy(scanrefer_val[0])
            data["scene_id"] = scene_id
            new_scanrefer_val.append(data)
    else:
        # get initial scene list
        train_scene_list = sorted(list(set([data["scene_id"] for data in scanrefer_train])))
        val_scene_list = sorted(list(set([data["scene_id"] for data in scanrefer_val])))

        # if args.debug: 
        #     train_scene_list = [train_scene_list[0]]
        #     val_scene_list = [val_scene_list[0]]

        # filter data in chosen scenes
        new_scanrefer_train = []
        for data in scanrefer_train:
            if data["scene_id"] in train_scene_list:
                new_scanrefer_train.append(data)
        
        new_scanrefer_val = []
        for scene_id in val_scene_list:
            data = organized_val[scene_id][0]
            new_scanrefer_val.append(data)

    # all scanrefer scene
    all_scene_list = train_scene_list + val_scene_list

    print("using {} dataset".format(args.dataset))
    print("train on {} samples from {} scenes".format(len(new_scanrefer_train), len(set(train_scene_list))))
    print("eval on {} samples from {} scenes".format(len(new_scanrefer_val), len(set(val_scene_list))))

    return new_scanrefer_train, new_scanrefer_val, all_scene_list

def get_trainer(args, monitor, logger):
    trainer = pl.Trainer(
        gpus=-1, # use all available GPUs 
        strategy="ddp_find_unused_parameters_false",
        # strategy="ddp",
        accelerator="gpu", # use multiple GPUs on the same machine
        max_epochs=args.epoch, 
        num_sanity_val_steps=args.num_sanity_val_steps, # validate on all val data before training 
        log_every_n_steps=args.verbose,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        callbacks=[monitor], # comment when debug
        logger=logger,
        reload_dataloaders_every_n_epochs=1
    )

    return trainer

def get_logger(args, root):
    logger = pl.loggers.TensorBoardLogger(root, name="logs")

    return logger

def get_monitor(args, root):
    monitor = pl.callbacks.ModelCheckpoint(
        monitor="eval/{}".format(args.criterion),
        mode="max",
        save_weights_only=True,
        dirpath=root,
        filename="model",
        save_last=True
    )

    return monitor

def set_up_root_dir(args):
    name = args.name.upper()
    tag = args.tag.upper()
    root = os.path.join(CONF.PATH.OUTPUT, name)
    subroot = os.path.join(root, tag)
    os.makedirs(root, exist_ok=True)
    os.makedirs(subroot, exist_ok=True)

    return root, subroot

def train(args):
    # init training dataset
    print("preparing data...")
    scanrefer_train, scanrefer_val, all_scene_list = get_scanrefer(args)

    # dataloader
    train_dataset, train_dataloader = get_dataloader(args, scanrefer_train, all_scene_list, "train", DC, True)
    val_dataset, val_dataloader = get_dataloader(args, scanrefer_val, all_scene_list, "val", DC, False)

    print("creating directory...")
    root, subroot = set_up_root_dir(args)

    print("initializing model...")
    pipeline = get_model(args, val_dataset, subroot) # NOTE properties in val_dataset is needed
    num_params = get_num_params(pipeline)

    print("initializing monitor...")
    monitor = get_monitor(args, subroot)

    print("initializing logger...")
    logger = get_logger(args, root) # logs are stored under root for cross-comparison

    print("initializing trainer...")
    trainer = get_trainer(args, monitor, logger)

    print("Start training...\n")
    save_info(args, subroot, num_params, train_dataset, val_dataset)
    trainer.fit(model=pipeline, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    # trainer.fit(model=pipeline)

if __name__ == "__main__":
    # setting
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    # mp.set_start_method("spawn")

    # reproducibility
    torch.manual_seed(CONF.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(CONF.seed)
    
    train(CONF)
