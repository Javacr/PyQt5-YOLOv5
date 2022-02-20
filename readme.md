### 友情提示：由于官方给出的yolov5版本会持续更新。为了避免不兼容的问题，建议使用本仓库的yolov5。如果想兼容最新版本的yolov5，自行更改对应的代码即可，改动不大。
本仓库的yolov5版本为**v5.0**，是直接从官方仓库拉取的，支持训练。

本仓库依赖模型有yolov5s.pt、yolov5m.pt、yolov5l.pt、yolov5x.pt,下载地址：https://github.com/ultralytics/yolov5/releases/tag/v5.0
点击地址后翻到最下面有下载链接，将下载好的模型放在pt文件夹下。

如果模型下载太慢，可以用百度网盘下载，链接：https://pan.baidu.com/s/1sFFWVyidFZZKi76CsKhf6Q?pwd=6666 
提取码：6666

### 更新日期：2022/2/6

**界面**

![界面](https://github.com/Javacr/PyQt5-YOLOv5/blob/v3.0/imgs/%E7%95%8C%E9%9D%A2.jpg)

**本地图片检测画面：**

![本地图片](https://github.com/Javacr/PyQt5-YOLOv5/blob/v3.0/imgs/%E5%9B%BE%E7%89%87.png)

**本地视频检测画面：**

![本地视频](https://github.com/Javacr/PyQt5-YOLOv5/blob/v3.0/imgs/%E8%A7%86%E9%A2%91.png)

演示视频：
[https://www.bilibili.com/video/BV1sQ4y1C7Vk?spm_id_from=333.999.0.0](https://www.bilibili.com/video/BV1sQ4y1C7Vk?spm_id_from=333.999.0.0)

csdn:
[https://blog.csdn.net/weixin_41735859/article/details/120507779?spm=1001.2014.3001.5501](https://blog.csdn.net/weixin_41735859/article/details/120507779?spm=1001.2014.3001.5501)

**功能：**

1. 模型选择
2. 输入选择(本地文件、摄像头、RTSP)；在检测RTSP视频流的时候，尽量不要启用帧间延时，否则会出现很高的延时，用yolo5x模型时，rtsp会很卡，建议抽帧检测, 把main.py中的133-135行注释取消
```python
                    # jump_count += 1
                    # if jump_count % 5 != 0:
                    #     continue
```

3. IoU调整
4. 置信度调整
5. 帧间延时调整
6. 播放/暂停/结束
7. 统计检测结果


**使用：**

运行*main.py*即可开启检测界面。ui文件也已上传，可以按照自己的想法更改ui界面。

**使用过程中如果遇到问题，欢迎issue**

**问题汇总**

**Q:** 把模型替换为最新版本的模型后，界面右下角有错误提示，但是没有报错，请问怎么解决？<br />
**A:** 最新版本的界面，用最新的yolov5模型不会报错或者闪退，这是因为加了异常捕获，避免闪退。如果你想看错误报告，可以把DetThread类中的异常捕获取消。
<br /><br />
**Q:** 点击摄像头按钮后，再检测，为什么还是检测上一次的文件？<br />
**A:** 点击摄像头后，会自动检测电脑连接了几个摄像头（除了电脑自带摄像头，有些人还会连接usb摄像头），检测完成后，摄像头下方会出现数字序号，你需要手动点击出现的数字选择摄像头。
<br /><br />
**Q:** 请问怎么更改背景图片、背景颜色？<br />
**A:** 自行搜索：pyqt5+qss。
<br /><br />
**Q:** 你是怎么学习PyQt5的？<br />
**A:** 买了本《PyQt5快速开发与实战》，书上没有的就查官方文档+CSDN+StackOverflow
