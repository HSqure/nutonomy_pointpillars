# PointPillar for Vitis AI使用说明



**Framework版本**

`Vitis AI V1.4.1`

`Vitis AI Library V1.4.1`

`vitis 2020.2`



**DPU配置 for xmodel**

`Dual B4096`

`RAM Usage LOW`

`DSP48 Usage HIGH`



**命令参数** 

可执行应用程序：

`./test_bin_pointpillars`

`./test_performance_pointpillars`

`./test_accuracy_pointpillars`

后接4个命令参数：

- **Model:**

    后接参数必须带上两个`model`文件。后接第一个参数的`model`名为 **PointNet**, 第二个参数的 `model` 为 **RPN**：

    **PointNet:**  `pointpillars_kitti_12000_0_pt`
    **RPN:**       `pointpillars_kitti_12000_1_pt`

- **Data file: **

    **点云文件:**  `.bin `

    **图像文件:** `.jpg` `.png` `...`



---



## 1. Host端交叉编译



### 1.1.初始环境设置

如系统中无相关环境，需要按命令运行脚本，安装如下组件：

- **Yocto sdk:** MPSoC中ARM端交叉编译工具

- **Vitis-AI 库:** 编译代码所需库文件

```shell
source setup/host_cross_compiler_setup_2020.2.sh
```



### 1.2.编译

输入如下命令开始编译：

```shell
bash build.sh
```



---



## 2. Edge端运行



### 1.1.初始环境设置

将`setup/vitis-ai-runtime-1.4.0`拷贝至目标版系统中，并进入目录`vitis-ai-runtime-1.4.0/2020.2/aarch64/centos`执行：

```
./setup.sh
```



### 2.1.运行程序

将可执行应用程序拷贝至开发板上，执行程序。



#### test_bin_pointpillars

```shell
env XLNX_POINTPILLARS_PRE_MT=1 ./test_bin_pointpillars  pointpillars_kitti_12000_0_pt pointpillars_kitti_12000_1_pt sample_pointpillars.bin  sample_pointpillars.png 
```

输出结果：

```shell
0       18.465866 3.999999 -1.708367 1.703191 4.350764 1.465484 1.679375     0.880797
0       10.917599 4.705865 -1.622433 1.650789 4.350764 1.634866 1.632500     0.867036
0       34.531731 1.571731 -1.563948 1.503061 3.495937 1.420396 1.726250     0.851953
1       21.338514 -2.400001 -1.681677 0.600000 1.963422 1.784916 4.742843     0.777300
0       57.891731 -4.188268 -1.536627 1.575194 3.780010 1.512004 2.007500     0.679179
```



#### test_accuracy_pointpillars

```shell
env XLNX_POINTPILLARS_PRE_MT=1 ./test_accuracy_pointpillars pointpillars_kitti_12000_0_pt  pointpillars_kitti_12000_1_pt  test_accuracy_pointpillars_bin.list test_accuracy_pointpillars_rgb.list test_accuracy_pointpillars_calib.list result
```

每个数据对应的检测结果文件路径保存在`test_accuracy_pointpillars_bin.list`中。



#### test_performance_pointpillars

```shell
env XLNX_POINTPILLARS_PRE_MT=1 ./test_performance_pointpillars pointpillars_kitti_12000_0_pt  pointpillars_kitti_12000_1_pt  -t 1 -s 30 test_performance_pointpillars.list
```

**注意：**

环境变量`XLNX_POINTPILLARS_PRE_MT`指定了进程内部的线程数。(范围：`1 - 4`，默认为 `2`)



```shell
note: env variable  XLNX_POINTPILLARS_PRE_MT means the inner threads num for preprocess. default value=2 (range 1 to 4).
  if it >1, it may cause some random to the result ( usually it becomes better)
  for accuracy test, it's better to set it to 1 to eliminate the random.
  for performance test, the total threads num will be inner_threads_num*test_threads_num,
  if this value is too big, too many threads cause the performance will be dropping instead of raising. 
  so if test threads num is little, XLNX_POINTPILLARS_PRE_MT can be set a little bigger;
  if test threads num is big, XLNX_POINTPILLARS_PRE_MT can be set to 1.
  it's better to do more test with different combination to choose the best value. 
```



💡 **Tips**：如何设置参数 `DISPLAY_PARAM`

参数`DISPLAY_PARAM`中有3个变量: **P2**, **rect**, **Trv2c**. 其中每一项都是`4x4`矩阵 `Float` 数据，矩阵最后一行都为 [0,0,0,1]。

- **Trv2c:**
        in calib_velo_to_cam.txt, there are R (3x3 ) and T (3x1). 
        R is rotation matrix, T is translation vector.
        R|T takes a point in Velodyne coordinates and transforms it into the
        coordinate system of the left video camera. Likewise it serves as a
        representation of the Velodyne coordinate frame in camera coordinates.
        Reshape R to 3x3 and make R|T be 3x4 matrix. This is the upper 3 rows of Trv2c

- **P2:**
        in calib_cam_to_cam.txt, P_rect_02 ( normally, _02 is used)
        reshape P_rect_02 to 3x4 Matrix, this is the upper 3 rows of P2.

- **rect:**
        in calib_cam_to_cam.txt, reshape R_rect_00 to 3x3 Matrix (called r).
        then reshape r|0 to 3x4 Matrix ( add 0 to each row  as last element )
        this is the upper 3 rows of rect.



