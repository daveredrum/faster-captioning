import os
import sys
import json
import argparse

import numpy as np
import point_cloud_utils as pcu

sys.path.append(".")
from utils.box_util import write_bbox

DATA_ROOT = "/cluster/balrog/dchen/ScanRefer/data/"
SCANNET_ROOT = os.path.join(DATA_ROOT, "scannet")
SCANNET_DATA_ROOT = os.path.join(SCANNET_ROOT, "scannet_data")
OUTPUT_ROOT = "./outputs/"

def load_scanrefer(args):
    print("loading ScanRefer {}...".format(args.split))
    with open(os.path.join(DATA_ROOT, "ScanRefer_filtered_{}_gt_bbox.json".format(args.split))) as f:
        data = json.load(f)

    return data

def visualize(args, scanrefer):
    for data in scanrefer:
        scene_id = data["scene_id"]
        object_id = data["object_id"]
        bbox = data["bbox"]

        if scene_id != args.scene_id: continue

        scene_root = os.path.join(OUTPUT_ROOT, "GT", "vis", scene_id)
        os.makedirs(scene_root, exist_ok=True)

        pc_save_path = os.path.join(scene_root, "scene.ply")
        if not os.path.exists(pc_save_path):
            pc_path = os.path.join(SCANNET_DATA_ROOT, scene_id+"_aligned_vert.npy")
            pc_numpy = np.load(pc_path)
            pcu.save_mesh_vc(pc_save_path, v=pc_numpy[:, :3], c=pc_numpy[:, 3:6] / 255.)

        gt_box_path = os.path.join(scene_root, "{}.ply".format(object_id))
        write_bbox(np.array(bbox), [0, 255, 0], gt_box_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--scene_id", type=str, required=True)
    args = parser.parse_args()

    print("organizing predictions...")
    scanrefer = load_scanrefer(args)

    print("visualizing...")
    visualize(args, scanrefer)

    print("done!")
