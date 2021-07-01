## KITTI Dataset概览
```
$KITTI_DATASET_ROOT
 ├── testing # 7580 test data
 │   ├── calib
 │   ├── image_2  # for visualization
 │   ├── velodyne # point cloud bin file
 │   └── velodyne_reduced # empty directory
 └── training
     ├── calib  # 7481 train data
     ├── image_2
     ├── label_2
     ├── velodyne
     └── velodyne_reduced # empty directory

```
其中training目录是有标注的数据（标注数据目录是label_2），工程中training和evaluate用的都是training目录下的数据；testing目录下面都是没有标注的数据，可以用来测测模型的检测效果（需要自己写可视化代码看看最终的检测结果）。


## 创建info数据

在工程根目录下进入子目录`second`，通过下面几条命令创建info数据。

---
01.**创建.pkl参数文件**

```
python create_data.py create_kitti_info_file --data_path=$KITTI_DATASET_ROOT
```

在`$KITTI_DATASET_ROOT`目录下创建:
`kitti_infos_train.pkl`、
`kitti_infos_val.pkl`、
`kitti_infos_trainval.pkl`、
`kitti_infos_test.pkl`
四个`.bin`文件，每个`.bin`文件包含了**图片的路径**、**calib中Camera和Lidar的标定参数**等。

---
02.**生成裁减后的数据副本**

```
python create_data.py create_reduced_point_cloud --data_path=$KITTI_DATASET_ROOT
```

在`$KITTI_DATASET_ROOT/training/velodyne_reduced`和`$KITTI_DATASET_ROOT/testing/velodyne_reduced`目录下分别创建它们同级velodyne目录中点云`.bin`文件的reduce版本，去掉了点云数据中一些冗余的背景等数据，可以认为是经过裁剪的点云数据。

---
03.**生成GroundTruth数据**

```
python create_data.py create_groundtruth_database --data_path=$KITTI_DATASET_ROOT
```

在`$KITTI_DATASET_ROOT`目录下创建`kitti_dbinfos_train.pkl`数据。在训练代码中的`input_cfg.database_sampler`变量中用到。

---

完成步骤后的dataset概览：
```
$KITTI_DATASET_ROOT
 ├── kitti_dbinfos_train.pkl  # step 03
 ├── kitti_infos_test.pkl # step 01
 ├── kitti_infos_train.pkl # step 01
 ├── kitti_infos_trainval.pkl # step 01
 ├── kitti_infos_val.pkl # step 01
 ├── gt_database # step 03
 ├── testing
 │   ├── calib
 │   ├── image_2  
 │   ├── velodyne 
 │   └── velodyne_reduced # step 02
 └── training
     ├── calib 
     ├── image_2
     ├── label_2
     ├── velodyne
     └── velodyne_reduced # step 02

```

---

