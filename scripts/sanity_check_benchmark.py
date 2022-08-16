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

def load_predictions(args):
    print("loading predictions...")
    with open(args.path) as f:
        data = json.load(f)

    return data

def visualize(args, predictions):
    for key, data in predictions.items():
        scene_id, object_id = key.split("|")
        box = data["box"]
        gt_box = data["gt_box"]

        if scene_id != args.scene_id: continue

        scene_root = os.path.join(OUTPUT_ROOT, "benchmark", "vis", scene_id)
        os.makedirs(scene_root, exist_ok=True)

        pc_save_path = os.path.join(scene_root, "scene.ply")
        if not os.path.exists(pc_save_path):
            pc_path = os.path.join(SCANNET_DATA_ROOT, scene_id+"_aligned_vert.npy")
            pc_numpy = np.load(pc_path)
            pcu.save_mesh_vc(pc_save_path, v=pc_numpy[:, :3], c=pc_numpy[:, 3:6] / 255.)

        box_path = os.path.join(scene_root, "pred_{}.ply".format(object_id))
        write_bbox(np.array(box), [0, 0, 255], box_path)
        
        gt_box_path = os.path.join(scene_root, "gt_{}.ply".format(object_id))
        write_bbox(np.array(gt_box), [0, 255, 0], gt_box_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_id", type=str, required=True)
    parser.add_argument("--path", type=str, help="Path to the prediction json file")
    args = parser.parse_args()

    print("organizing predictions...")
    predictions = load_predictions(args)

    print("visualizing...")
    visualize(args, predictions)

    print("done!")
