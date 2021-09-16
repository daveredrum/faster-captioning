import os
import sys
import yaml
import argparse

from datetime import datetime
from easydict import EasyDict

CONF = EasyDict()

# path
CONF.PATH = EasyDict()
CONF.PATH.BASE = "/rhome/dchen/SSL/faster-captioning/" # TODO: change this
CONF.PATH.NODE = "balrog" # TODO: change this
CONF.PATH.CLUSTER = "/cluster/{}/dchen/ScanRefer/".format(CONF.PATH.NODE) # where the data is stored TODO: change this
# CONF.PATH.BASE = "/home/davech2y/Scan2Cap/" # TODO: change this
# CONF.PATH.DATA = os.path.join(CONF.PATH.BASE, "data")
CONF.PATH.DATA = os.path.join(CONF.PATH.CLUSTER, "data")
CONF.PATH.SCANNET = os.path.join(CONF.PATH.DATA, "scannet")
CONF.PATH.LIB = os.path.join(CONF.PATH.BASE, "lib")
CONF.PATH.MODELS = os.path.join(CONF.PATH.BASE, "models")
CONF.PATH.UTILS = os.path.join(CONF.PATH.BASE, "utils")

# append to syspath
for _, path in CONF.PATH.items():
    sys.path.append(path)

# scannet data
CONF.PATH.SCANNET_SCANS = os.path.join(CONF.PATH.SCANNET, "scans")
CONF.PATH.SCANNET_META = os.path.join(CONF.PATH.SCANNET, "meta_data")
CONF.PATH.SCANNET_DATA = os.path.join(CONF.PATH.SCANNET, "scannet_data")

# Scan2CAD
CONF.PATH.SCAN2CAD = os.path.join(CONF.PATH.DATA, "Scan2CAD_dataset")

# data
CONF.SCANNET_DIR =  "/canis/Datasets/ScanNet/public/v2/scans" # TODO change this
CONF.SCANNET_FRAMES_ROOT = "/home/davech2y/frames_square/" # TODO change this
CONF.PROJECTION = "/home/davech2y/multiview_projection_scanrefer" # TODO change this
CONF.ENET_FEATURES_ROOT = "/home/davech2y/enet_features" # TODO change this
CONF.ENET_FEATURES_SUBROOT = os.path.join(CONF.ENET_FEATURES_ROOT, "{}") # scene_id
CONF.ENET_FEATURES_PATH = os.path.join(CONF.ENET_FEATURES_SUBROOT, "{}.npy") # frame_id
CONF.SCANNET_FRAMES = os.path.join(CONF.SCANNET_FRAMES_ROOT, "{}/{}") # scene_id, mode 
# CONF.SCENE_NAMES = sorted(os.listdir(CONF.SCANNET_DIR))
CONF.ENET_WEIGHTS = os.path.join(CONF.PATH.BASE, "data/scannetv2_enet.pth")
# CONF.MULTIVIEW = os.path.join(CONF.PATH.SCANNET_DATA, "enet_feats.hdf5")
CONF.MULTIVIEW = os.path.join(CONF.PATH.SCANNET_DATA, "enet_feats_maxpool.hdf5")
CONF.NYU40_LABELS = os.path.join(CONF.PATH.SCANNET_META, "nyu40_labels.csv")

# scannet
CONF.SCANNETV2_TRAIN = os.path.join(CONF.PATH.SCANNET_META, "scannetv2_train.txt")
CONF.SCANNETV2_VAL = os.path.join(CONF.PATH.SCANNET_META, "scannetv2_val.txt")
CONF.SCANNETV2_TEST = os.path.join(CONF.PATH.SCANNET_META, "scannetv2_test.txt")
CONF.SCANNETV2_LIST = os.path.join(CONF.PATH.SCANNET_META, "scannetv2.txt")

# output
CONF.PATH.OUTPUT = os.path.join(CONF.PATH.BASE, "outputs")

# pretrained
CONF.PATH.PRETRAINED = os.path.join(CONF.PATH.BASE, "pretrained")

# Pretrained features
CONF.PATH.GT_FEATURES = os.path.join(CONF.PATH.CLUSTER, "VoteNet_GT_features") # VoteNet grounding truth features
# CONF.PATH.GT_FEATURES = os.path.join(CONF.PATH.CLUSTER, "PointGroup_GT_features") # PointGroup grounding truth features
# CONF.PATH.VOTENET_FEATURES = os.path.join(CONF.PATH.CLUSTER, "votenet_features")
CONF.PATH.VOTENET_FEATURES = os.path.join(CONF.PATH.CLUSTER, "votenet_{}_predictions") # dataset

# train
CONF.TRAIN = EasyDict()
CONF.TRAIN.MAX_DES_LEN = 30
CONF.TRAIN.SEED = 42
CONF.TRAIN.OVERLAID_THRESHOLD = 0.5
CONF.TRAIN.MIN_IOU_THRESHOLD = 0.25
CONF.TRAIN.NUM_BINS = 6
CONF.TRAIN.BEAM_SIZE = 1
CONF.TRAIN.NUM_PRESET_EPOCHS = 100
CONF.TRAIN.NUM_DES_PER_SCENE = 8

# eval
CONF.EVAL = EasyDict()
CONF.EVAL.BEAM_SIZE = 1
CONF.EVAL.MAX_DES_LEN = 30
CONF.EVAL.MIN_IOU_THRESHOLD = 0.5

def get_parser(args):
    parser = argparse.ArgumentParser()
    
    # ---------- general settings -----------
    parser.add_argument("--config", type=str, default="config/train.yaml")

    # for eval
    parser.add_argument("--eval_caption", action="store_true", help="evaluate the reference localization results")
    parser.add_argument("--eval_reference", action="store_true", help="evaluate the object detection results")
    parser.add_argument("--eval_generated", action="store_true", help="evaluate on the generated descriptions")
    parser.add_argument("--eval_detection", action="store_true", help="evaluate detection")
    parser.add_argument("--eval_pretrained", action="store_true", help="evaluate the pretrained object detection results")
    
    parser.add_argument("--min_ious", type=str, default="0.25,0.5", help="Min IoU threshold for evaluation")

    parser.add_argument("--use_train", action="store_true", help="Use train split in evaluation.")
    
    parser.add_argument("--repeat", type=int, default=1, help="Repeat for N times")
    
    args_cfg = parser.parse_args()
    assert args_cfg.config is not None

    # load config
    with open(args_cfg.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

        for k, v in config.items():
            setattr(args, k, v)

    # override config
    for key, value in vars(args_cfg).items():
        setattr(args, key, value)

    # for eval mode
    if args_cfg.eval_reference: setattr(args, "use_rl", False)

    # add time stamp
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if "datetime" not in args: setattr(args, "datetime", stamp)
    
    # convert to upper case
    if "name" in args: setattr(args, "name", args.name.upper())
    if "tag" in args: setattr(args, "tag", args.tag.upper())

    return args

CONF = get_parser(CONF)
