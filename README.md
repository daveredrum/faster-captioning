# FasterCaptioning

## Introduction

A Faster Implementation of [Scan2Cap: Context-aware Dense Captioning in RGB-D Scans](https://github.com/daveredrum/Scan2Cap)

## Setup

Please refer to [Scan2Cap](https://github.com/daveredrum/Scan2Cap) for the data preparation and setup details.

## :star2: Benchmark Toolbox :star2:

For submission to the Scan2Cap benchmark, run the following script to generate predictions:

```shell
python benchmark/predict.py --config outputs/XYZ_MULTIVIEW_NORMAL/VOTENET_SCAN2CAP/info.yaml --test_split test
```

Please compress the `benchmark_test.json` as a .zip or .7z file and follow the [instructions](http://kaldir.vc.in.tum.de/scanrefer_benchmark/documentation) to upload your results.

### Local Benchmarking on Val Set

Before submitting the results on the test set to the official benchmark, you can also benchmark the performance on the val set. Run the following script to generate GTs for val set first:

```shell
python scripts/build_benchmark_gt.py --split val
```

> NOTE: don't forget to change the `DATA_ROOT` in `scripts/build_benchmark_gt.py`

Generate the predictions on val set:

```shell
python benchmark/predict.py --config outputs/XYZ_MULTIVIEW_NORMAL/VOTENET_SCAN2CAP/info.yaml --test_split val
```

Evaluate the predictions on the val set:

```shell
python benchmark/eval.py --split val --path <path to predictions> --verbose
```

## Usage

### End-to-End training for 3D dense captioning

Run the following script to start the end-to-end training of Scan2Cap model using the multiview features and normals. For more training options, please run `scripts/train.py -h`:

```shell
python scripts/train.py --config config/votenet_scan2cap.yaml
```

The trained model as well as the intermediate results will be dumped into `outputs/<output_folder>`. For evaluating the model (@0.5IoU), please run the following script and change the `<output_folder>` accordingly, and note that arguments must match the ones for training:

```shell
python scripts/eval.py --config outputs/XYZ_MULTIVIEW_NORMAL/VOTENET_SCAN2CAP/info.yaml --eval_caption
```

Evaluating the detection performance:

```shell
python scripts/eval.py --config outputs/XYZ_MULTIVIEW_NORMAL/VOTENET_SCAN2CAP/info.yaml --eval_detection
```

You can even evaluate the pretraiend object detection backbone:


## Citation
If you found our work helpful, please kindly cite our paper via:
```bibtex
@inproceedings{chen2021scan2cap,
  title={Scan2Cap: Context-aware Dense Captioning in RGB-D Scans},
  author={Chen, Zhenyu and Gholami, Ali and Nie{\ss}ner, Matthias and Chang, Angel X},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={3193--3203},
  year={2021}
}
```

## License
Scan2Cap is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License](LICENSE).

Copyright (c) 2021 Dave Zhenyu Chen, Ali Gholami, Matthias Nießner, Angel X. Chang
