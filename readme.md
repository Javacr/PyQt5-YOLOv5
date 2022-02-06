### 友情提示：官网的yolov5版本会持续更新，比如最近新增了SPPF类。为了避免不兼容的问题，建议使用本仓库的yolov5。如果想兼容最新版本的yolov5，自行更改对应的代码即可，改动不大。
本仓库的yolov5版本为v5.0，由于是直接从yolov5仓库下拉取下来的源码，本仓库也支持训练。模型下载地址：https://github.com/ultralytics/yolov5/releases/tag/v5.0
翻到最下面有链接下载.

2021/10/8: 所有代码已上传，直接clone后，运行*yolo_win.py*即可开启界面。

2021/9/29：加入置信度选择
![置信度](https://github.com/Javacr/PyQt5-YOLOv5/blob/master/imgs/20210929_134634%2000_00_00-00_00_30.gif)

界面是在*ultralytics*的[yolov5](https://github.com/ultralytics/yolov5)基础上建立的，界面使用*pyqt5*实现，内容较简单，娱乐而已。

**功能：**

1. 模型选择
2. 本地文件选择(视频图片均可)
3. 开关摄像头
4. 运行/终止
5. 统计检测结果

![界面](https://github.com/Javacr/PyQt5-YOLOv5/blob/master/imgs/%E7%95%8C%E9%9D%A2.jpg)
默认模型为*yolov5s.pt*，默认输入文件为电脑摄像头视频

使用视频：
[https://www.bilibili.com/video/BV1sQ4y1C7Vk?spm_id_from=333.999.0.0](https://www.bilibili.com/video/BV1sQ4y1C7Vk?spm_id_from=333.999.0.0)

csdn:
[https://blog.csdn.net/weixin_41735859/article/details/120507779?spm=1001.2014.3001.5501](https://blog.csdn.net/weixin_41735859/article/details/120507779?spm=1001.2014.3001.5501)

摄像头检测画面：

![摄像头](https://github.com/Javacr/PyQt5-YOLOv5/blob/master/imgs/%E6%91%84%E5%83%8F%E5%A4%B4.jpg)

本地视频检测画面：
![本地](https://github.com/Javacr/PyQt5-YOLOv5/blob/master/imgs/video.gif)

本地图片检测画面：
![本地](https://github.com/Javacr/PyQt5-YOLOv5/blob/master/imgs/%E5%9B%BE%E7%89%87.png)

## 使用

运行*yolo_win.py*即可开启检测界面。

存在的一个小问题，切换模型或者文件**过于频繁**，可能会卡住，重启一下即可。

这种情况很少出现，问题不大。

