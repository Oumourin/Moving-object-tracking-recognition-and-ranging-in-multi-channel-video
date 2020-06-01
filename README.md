# Moving-object-tracking-recognition-and-ranging-in-multi-channel-video

多路视频中的运动物体跟踪、识别和测距

##  项目简介

本项目是对于多个视频监控系统综合应用的探索，它由以下已有功能和可能有的功能组成

- [x] 运动目标的实时跟踪、检测

- [x] 相似三角形的单目测距

- [x] SURF的不同视频流中相同物体检测

- [x] SURF的视频拼接算法

  ~~以下是🕊🕊🕊~~

- [ ] GUI前端

- [ ] Docker一键部署

- [ ] CMAKE跨平台

- [ ] 更加合理的接口

##  项目结构

项目结构如目录树所示

```powershell
│  .gitignore
│  .gitmodules
│  config.json
│  LICENSE
│  main.py
│  README.md
│  TestCamera.py
│  tree.txt
│  
├─bin
│      SURF_DLL.dll
│      
├─object_detection
│  │  object_detection.py
│  │  __init__.py
│  │  
│  ├─cfg
│  │      .gitkeep
│  │      yolov3-320.cfg
│  │      yolov3-tiny.cfg
│  │      yolov3.cfg
│  │      
│  ├─classes
│  │      .gitkeep
│  │      object_detection_classes_yolov3.txt
│  │      
│  ├─weights
│  │      .gitkeep
│  │      yolov3-tiny.weights
│  │      yolov3.weights
│  │      
│  └─__pycache__
│          object_detection.cpython-38.pyc
│          __init__.cpython-38.pyc
│          
├─object_distance
│  │  object_distance.py
│  │  __init__.py
│  │  
│  ├─Calibration Picture
│  │      LOGITECH.jpg
│  │      paper-distance1.jpg
│  │      paper-distance2.jpg
│  │      Paper1.jpg
│  │      Paper2.jpg
│  │      Paper3.jpg
│  │      Paper4.jpg
│  │      Picture.jpg
│  │      Picture1.jpg
│  │      Picture2.jpg
│  │      Picture3.jpg
│  │      Token1.jpg
│  │      Token2.jpg
│  │      Token3.jpg
│  │      Token4.jpg
│  │      Token5.jpg
│  │      Token6.jpg
│  │      
│  └─__pycache__
│          object_distance.cpython-38.pyc
│          __init__.cpython-38.pyc
│          
├─src
│      dllmain.cpp
│      dll_surf.cpp
│      framework.h
│      pch.cpp
│      pch.h
│      Source.def
│      
└─SURF
    │  image_match.py
    │  image_sitiching.py
    │  load_surf.py
    │  surf.py
    │  __init__.py
    │  
    ├─image
    │      1.jpg
    │      2.jpg
    │      aim_image.jpg
    │      aim_image2.jpg
    │      left.JPG
    │      left.PNG
    │      NUC_left.JPG
    │      NUC_left1.JPG
    │      NUC_left2.JPG
    │      NUC_right.JPG
    │      NUC_right1.JPG
    │      NUC_right2.JPG
    │      right.JPG
    │      right.PNG
    │      source_image.jpg
    │      
    ├─Video
    │      left.mp4
    │      left1.mp4
    │      right.mp4
    │      right1.mp4
    │      VID_20200513_150229.mp4
    │      WIN_20200513_13_13_38_Pro.mp4
    │      WIN_20200513_13_14_28_Pro.mp4
    │      WIN_20200513_13_29_20_Pro.mp4
    │      WIN_20200513_13_29_57_Pro.mp4
    │      WIN_20200513_14_47_21_Pro.mp4
    │      WIN_20200513_14_47_22_Pro.mp4
    │      WIN_20200513_15_02_33_Pro.mp4
    │      
    └─__pycache__
            load_surf.cpython-38.pyc
            surf.cpython-38.pyc
            __init__.cpython-38.pyc
```

*  bin为C++描述的SURF算法动态调用库生成位置
*  object_detection包含了物体跟踪与检测相关算法，同时提供了相关模型文件和权重文件
*  object_distance包含了目标测距算法，校准用图片存在Calibration Picture文件夹中
*  src包含了C++描述的SURF算法源代码文件
*  SURF包含了Python描述的SURF算法脚本和加载C++生成的SURF算法动态调用库的函数接口，同时包含了相关的测试文件
*  主文件夹下
   *  config.json用于记录相机焦距内参、待检测物体实际高度、检测模型文件位置
   *  TestCamera.py用于检测相机的连接情况
   *  main.py 程序启动

##  项目运行环境

项目依赖以下运行环境：

*  OpenCV 4.X with nonfree module and cuda module
*  CUDA
*  cuDNN
*  Python3.x

运行需要自行对OpenCV源码进行编译，编译需要包含nonfree和CUDA相关功能，否则无法正常运行，同时需要编译Python3前端

附上OpenCV430源码仓库和OpenCV430-contrib仓库

```shell
wget https://github.com/opencv/opencv/archive/4.3.0.zip
wget https://github.com/opencv/opencv_contrib/releases/tag/4.3.0
```

##  目标跟踪与检测模块训练相关

程序本质是利用OpenCV提供的Darknet接口或ONNX接口加载我们迁移训练好的模型文件，故可以使用YOLO原生的DarkNet框架进行训练，用训练好的weights文件路径修改config内的模型路径即可。

本项目使用的Pytorch进行训练，这里提供Pytorch版本的YOLOV3代码仓库地址

```shell
git clone https://github.com/ultralytics/yolov3.git
```

Pytorch训练好的pt权重文件我们需要进行权重文件格式转换，转换为可以加载的ONNX格式。

本项目使用的数据集为VOC2012 类别只选择了车辆 巴士 人

具体训练细节可参考官方文档