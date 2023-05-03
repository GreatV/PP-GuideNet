# PP-GuideNet

A PaddlePaddle implementation for [Learning Guided Convolutional Network for Depth Completion](https://arxiv.org/abs/1908.01238) ([GudieNet](https://github.com/kakaxi314/GuideNet)).


## Install

1. Install [PaddlePaddle](https://www.paddlepaddle.org.cn/)

2. Build CUDA extension

```shell
# You may need to install additional packages.
sudo apt install libpython3.8-dev libnccl-dev

cd ext
python setup.py install
```

3. Install dependency

```bash
python -m pip install -r requirements.txt
```

## Dataset

Please download the KITTI [depth completion](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion) dataset and create a symlink to `datas/kitti`. The dataset structure is shown below:

```
└── datas
    └── kitti
        ├── data_depth_annotated
        │   ├── train
        │   └── val
        ├── data_depth_velodyne
        │   ├── train
        │   └── val
        ├── raw
        │   ├── 2011_09_26
        │   ├── 2011_09_28
        │   ├── 2011_09_29
        │   ├── 2011_09_30
        │   └── 2011_10_03
        ├── test_depth_completion_anonymous
        │   ├── image
        │   ├── intrinsics
        │   └── velodyne_raw
        └── val_selection_cropped
            ├── groundtruth_depth
            ├── image
            ├── intrinsics
            └── velodyne_raw
```

## Train

```bash
python -m paddle.distributed.launch train.py
```

## Test

```bash
python test.py
```

## Performance on the KITTI dataset `val_selection_cropped`

|RMSE|MAE|iRMSE|iMAE|Pre-trained model|
|:----:|:----:|:------:|:------:|:----:|
|776.48|225.29|2.52|1.05|[GNS_20230228](https://github.com/GreatV/PP-GuideNet/releases/download/v0.1/GNS_20230228.tar.xz)|
