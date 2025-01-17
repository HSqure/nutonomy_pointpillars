# PointPillars Pytorch

Welcome to PointPillars(This is origin from nuTonomy/second.pytorch ReadMe.txt).

This repo demonstrates how to reproduce the results from
[_PointPillars: Fast Encoders for Object Detection from Point Clouds_](https://arxiv.org/abs/1812.05784) (to be published at CVPR 2019) on the
[KITTI dataset](http://www.cvlibs.net/datasets/kitti/) by making the minimum required changes from the preexisting
open source codebase [SECOND](https://github.com/traveller59/second.pytorch). 

This is not an official nuTonomy codebase, but it can be used to match the published PointPillars results.

**WARNING: This code is not being actively maintained. This code can be used to reproduce the results in the first version of the paper, https://arxiv.org/abs/1812.05784v1. For an actively maintained repository that can also reproduce PointPillars results on nuScenes, we recommend using [SECOND](https://github.com/traveller59/second.pytorch). We are not the owners of the repository, but we have worked with the author and endorse his code.**


## Getting Started!

This is a fork of [SECOND for KITTI object detection](https://github.com/traveller59/second.pytorch) and the relevant
subset of the original README is reproduced here.

___





## 1. 环境配置

### 1.1 Clone本工程

```bash
git clone https://github.com/SmallMunich/nutonomy_pointpillars.git
```

### 1.2 安装所需 Python 拓展包

推荐使用**Anaconda package manager**。

**Anaconda:**

```bash
conda create -n pointpillars python=3.6 anaconda
source activate pointpillars
conda install shapely pybind11 protobuf scikit-image numba pillow
conda install pytorch torchvision -c pytorch
conda install google-sparsehash -c bioconda
```
**Pipe:**

```bash
pip install --upgrade pip
pip install fire tensorboardX
```

Additionally, you may need to install Boost geometry:

```bash
sudo apt-get install libboost-all-dev
```

### 1.3 numba CUDA设置

You need to add following environment variables for numba to `~/.bashrc`:

```bash
export NUMBAPRO_CUDA_DRIVER=/usr/lib/x86_64-linux-gnu/libcuda.so
export NUMBAPRO_NVVM=/usr/local/cuda/nvvm/lib64/libnvvm.so
export NUMBAPRO_LIBDEVICE=/usr/local/cuda/nvvm/libdevice
```

### 1.4 路径环境变量

Add `nutonomy_pointpillars/` to your `PYTHONPATH`.


```bash 
export PYTHONPATH=$PYTHONPATH:/your_root_path/nutonomy_pointpillars/
```
> 注: 如果有vitis-ai gpu docker环境，可直接在环境内运行:
> ```bash
> ./env_setup.sh
> ```


---




## 2. KITTI数据集准备

KITTI Dataset概览:
```plain
$KITTI_DATASET_ROOT
 ├── testing  <-- 7580 test data
 │   ├── calib
 │   ├── image_2  <-- for visualization
 │   ├── velodyne  <-- point cloud bin file
 │   └── velodyne_reduced  <-- empty directory
 └── training
     ├── calib  <-- 7481 train data
     ├── image_2
     ├── label_2
     ├── velodyne
     └── velodyne_reduced  <-- empty directory

```
其中**training**目录是有标注的数据（标注数据目录是**label_2**），工程中**training**和**evaluate**用的都是**training**目录下的数据；**testing**目录下面是无标注的数据，可以用来测测模型的检测效果（需要在可视化代码工程**second_pytorch**中查看最终的检测结果）。

将数据集根目录设置为`$KITTI_DATASET_ROOT`之后，在工程根目录下进入子目录`second`，通过下面几条命令创建info数据。

### 2.1 创建pkl参数文件

```bash
python create_data.py create_kitti_info_file --data_path=$KITTI_DATASET_ROOT
```

在`$KITTI_DATASET_ROOT`目录下创建:
`kitti_infos_train.pkl`、
`kitti_infos_val.pkl`、
`kitti_infos_trainval.pkl`、
`kitti_infos_test.pkl`
四个`.bin`文件，每个`.bin`文件包含了**图片的路径**、**calib中Camera和Lidar的标定参数**等。

### 2.2 生成裁减后的数据副本

```bash
python create_data.py create_reduced_point_cloud --data_path=$KITTI_DATASET_ROOT
```

在`$KITTI_DATASET_ROOT/training/velodyne_reduced`和`$KITTI_DATASET_ROOT/testing/velodyne_reduced`目录下分别创建它们同级velodyne目录中点云`.bin`文件的reduce版本，去掉了点云数据中一些冗余的背景等数据，可以认为是经过裁剪的点云数据。

### 2.3 生成GroundTruth数据

```bash
python create_data.py create_groundtruth_database --data_path=$KITTI_DATASET_ROOT
```

在`$KITTI_DATASET_ROOT`目录下创建`kitti_dbinfos_train.pkl`数据。在训练代码中的`input_cfg.database_sampler`变量中用到。

完成步骤后的dataset概览：
```plain
$KITTI_DATASET_ROOT
 ├── kitti_dbinfos_train.pkl  <-- step 03
 ├── kitti_infos_test.pkl  <-- step 01
 ├── kitti_infos_train.pkl  <-- step 01
 ├── kitti_infos_trainval.pkl  <-- step 01
 ├── kitti_infos_val.pkl  <-- step 01
 ├── gt_database  <-- step 03
 ├── testing
 │   ├── calib
 │   ├── image_2  
 │   ├── velodyne 
 │   └── velodyne_reduced  <-- step 02
 └── training
     ├── calib 
     ├── image_2
     ├── label_2
     ├── velodyne
     └── velodyne_reduced  <-- step 02

```

### 2.4 修改配置文件

#### 2.4.1 数据集包pkl位置

配置文件路径:`second/configs/pointpillars/livox/xyres_16_all.proto`

需在以下部份包含前面生成的所有的数据集`.pkl`文件。

```bash
train_input_reader: {
  ...
  database_sampler {
    database_info_path: "/path/to/kitti_dbinfos_train.pkl"
    ...
  }
  kitti_info_path: "/path/to/kitti_infos_train.pkl"
  kitti_root_path: "KITTI_DATASET_ROOT"
}
...
eval_input_reader: {
  ...
  kitti_info_path: "/path/to/kitti_infos_val.pkl"
  kitti_root_path: "KITTI_DATASET_ROOT"
}
```
#### 2.4.2 目标类别变动

如果类别发生了改变，则需要修改的地方有:

**(1). 配置文件路径:**`second/configs/pointpillars/livox/xyres_16_all.proto`

```bash
model: {
  second: {
    ...
    num_class: 7 
    ...
  }
}
...
train_input_reader: {
  class_names: ["car","bus","motorcycle","bicycle","pedestrians","truck","Traffic_cone"]
  ...
}
...
eval_input_reader: {
  class_names: ["car","bus","motorcycle","bicycle","pedestrians","truck","Traffic_cone"]
}
```
**(2). 解析工具路径:**`second/data/kitti_common.py`

`line:222`
```python
def get_class_to_label_map():
    class_to_label = {
        'car': 0,
        'bus': 1, 
        'motorcycle': 2, 
        'bicycle': 3, 
        'pedestrians': 4, 
        'truck': 5, 
        'Traffic_cone': 6,         
        'DontCare': -1
    }
    return class_to_label
```

**(3). 评估工具路径:**`second/utils/eval.py` 

`line:740`
`line:801`
`line:858`

```python
def get_official_eval_result_v1(gt_annos, dt_annos, current_class):
	...
    class_to_name = {
        0: 'car',
        1: 'bus',
        2: 'motorcycle',
        3: 'bicycle',
        4: 'pedestrians',
        5: 'truck',
        6: 'Traffic_cone',
    }
    ...
 
def get_official_eval_result(gt_annos, dt_annos, current_classes, difficultys=[0, 1, 2], return_data=False):
	...
    class_to_name = {
        0: 'car',
        1: 'bus',
        2: 'motorcycle',
        3: 'bicycle',
        4: 'pedestrians',
        5: 'truck',
        6: 'Traffic_cone'
    }
  	...
    
    
def get_coco_eval_result(gt_annos, dt_annos, current_classes):
    class_to_name = {
        0: 'car',
        1: 'bus',
        2: 'motorcycle',
        3: 'bicycle',
        4: 'pedestrians',
        5: 'truck',
        6: 'Traffic_cone'
    }
    class_to_range = {
        0: [0.5, 1.0, 0.05],
        1: [0.25, 0.75, 0.05],
        2: [0.25, 0.75, 0.05],
        3: [0.5, 1.0, 0.05],
        4: [0.25, 0.75, 0.05],
        5: [0.5, 1.0, 0.05],
        6: [0.5, 1.0, 0.05],
    }
    class_to_range = {
        0: [0.5, 0.95, 10],
        1: [0.25, 0.7, 10],
        2: [0.25, 0.7, 10],
        3: [0.5, 0.95, 10],
        4: [0.25, 0.7, 10],
        5: [0.5, 0.95, 10],
        6: [0.5, 0.95, 10],

    }
```





## 3. 模型训练与评估



### 3.1 训练 Training

进入路径`second/`

```bash
python ./pytorch/train.py train --config_path=./configs/pointpillars/livox/xyres_16_all.proto --model_dir=./run
```

* If you want to train a new model, make sure `/path/to/model_dir`doesn't exist.

* If`/path/to/model_dir`does exist, training will be resumed from the last checkpoint.

* Training only supports a single GPU. 

* Training uses a `batchsize=2` which should fit in memory on most standard GPUs.

* On a single 1080Ti, training `xyres_16` requires approximately `20 hours` for `160 epochs`.

    

### 3.2 评估 Evaluate

进入路径`second/`


```bash
python pytorch/train.py evaluate --config_path=./configs/pointpillars/livox/xyres_16_all.proto --model_dir=./run
```

* Detection result will saved in model_dir/eval_results/step_xxx.
* By default, results are stored as a result.pkl file. To save as official KITTI label format use --pickle_result=False.



---





## 4. 模型量化



**方法1：量化并导出.xmodel**

* 运行:

```shell
./run_quant.sh
```



进入路径`second/`

```bash
python pytorch/finetune.py evaluate --config_path=./configs/pointpillars/livox/xyres_16_all.proto --model_dir=./run
```



**2.方法2：QAT(Quantization-Aware-Training)**

* 运行QAT:

```shell
./run_qat.sh
```

* 导出xmodel，运行:

```shell
./convert_test_qat.sh
```












### 参考

>* More Details will be update on my chinese blog:
> * export from pytorch to onnx IR blog : https://blog.csdn.net/Small_Munich/article/details/101559424  
> * onnx compare blog : https://blog.csdn.net/Small_Munich/article/details/102073540
> * tensorrt compare blog : https://blog.csdn.net/Small_Munich/article/details/102489147
> * wait for update & best wishes.

