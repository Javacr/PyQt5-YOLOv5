### 友情提示：由于官方给出的yolov5版本会持续更新。为了避免不兼容的问题，建议使用本仓库的yolov5。如果想兼容最新版本的yolov5，自行更改对应的代码即可。
本仓库的yolov5版本为**v5.0**，是直接从官方仓库拉取的，支持训练。

本仓库依赖模型有yolov5s.pt、yolov5m.pt、yolov5l.pt、yolov5x.pt,下载地址：https://github.com/ultralytics/yolov5/releases/tag/v5.0
点击地址后翻到最下面有下载链接，将下载好的模型放在pt文件夹下，运行界面时，**会自动检测已有模型**。

如果模型下载太慢，可以用百度网盘下载，链接：https://pan.baidu.com/s/1sFFWVyidFZZKi76CsKhf6Q?pwd=6666 
提取码：6666

### 更新日期：2022/5/29
新增：检测完成后，自动保存

**界面**

![界面](https://github.com/Javacr/PyQt5-YOLOv5/blob/master/imgs/%E7%95%8C%E9%9D%A2.png)

**运行效果：**

![运行效果](https://github.com/Javacr/PyQt5-YOLOv5/blob/master/imgs/%E8%BF%90%E8%A1%8C.png)

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
7. 统计检测结果（显示边框时，支持中文标签）
8. 检测完成后，自动保存检测结果


**使用：**
```bash
# conda创建python虚拟环境
conda create -n yolov5_pyqt5 python=3.8
# 激活环境
conda activate yolov5_pyqt5
# 到项目根目录下
cd ./
# 安装依赖文件
pip install -r requirements.txt
# 将下载好的模型放在pt文件夹下
# 运行main.py
python main.py
```
运行*main.py*开启检测界面后**会自动检测已有模型**。ui文件也已上传，可以按照自己的想法更改ui界面。

**使用过程中如果遇到问题，欢迎Issue，同时也欢迎Pull request**

**问题汇总**

**Q:** 把模型替换为yolov5最新版本的模型后，界面左下角有错误提示，但是没有报错，请问怎么解决？<br />
**A:** 使用v5.0之后的版本会报错"The size of tensor a (80) must match the size the size of tensor b(52) at non-singleton dimension 3"或者"AttributeError: Can't get attribute 'SPPF' on <module 'models.common' from 'D:\\Yolo_detect\\pyqt+yolo\\PyQt5-YOLOv5-master\\models\\common.py'>"。模型版本不同则无法直接使用，建议用本仓库训练，或者修改代码以匹配你的模型。如果你想看详细错误报告，可以把DetThread类中的异常捕获取消，或者调用cgit模块（自行搜索cgit的用法，很简单）。
<br /><br />
**Q:** 遇到错误：Permission denied：'pt'<br />
**A:** 可能是pt文件在下载时有损坏，重新下载，或者是没有将pt文件放到pt文件夹中。
<br /><br />
**Q:** ‘Upsample’ object has no attribute 'recompute_scale_factor'<br />
**A:** torch版本太高，下降一下版本吧，我的是torch=1.7.1、torchvision=0.8.2。torch==1.9.0、torchvision==0.10.0也可以。
<br /><br />
**Q:** 如何打包成exe文件？<br />
**A:** 打包过一次，但是启动后打不开，打包的文件1G左右，不建议打包。如果有人打包成功了，可以留个言。如果你只是不想使用命令行启动文件，windows用户可以建一个bat文件快速启动。
<br /><br />
**Q:** 点击摄像头按钮后，再检测，为什么还是检测上一次的文件？<br />
**A:** 点击摄像头按钮后，会自动检测电脑连接了几个摄像头（除了电脑自带摄像头，有些人还会连接usb摄像头），检测完成后，摄像头按钮下方会出现数字序号，你需要手动点击出现的数字选择摄像头。
<br /><br />
**Q:** 请问怎么更改背景图片、背景颜色？<br />
**A:** 自行搜索：pyqt5+qss。
<br /><br />
**Q:** 你是怎么学习PyQt5的？<br />
**A:** 买了本《PyQt5快速开发与实战》，书上没有的就查官方文档+CSDN+StackOverflow+github
<br /><br />
**Q:** 我将摄像头的默认分辨率640x480修改为1920x1080后，画面就很卡顿，FPS从30变为了7，我在网上也搜了相关的解决办法，有说需要重新编译opencv的，但是我单独用代码调用高分辨率摄像头不会卡顿，不知道是不是opencv的问题，请博主给一些指导。谢谢（修改帧率也试过了没用）<br />
**A:** 为了快速启动摄像头，代码中opencv是使用direct show模式打开摄像头的，这种模式下摄像头捕获分辨率调高之后，帧率可能会下降。改一下datasets.py中LoadWebcam类，大概242行，把括号中“, cv2.CAP_DSHOW”删除，括号里就留一个“eval(pipe)”就行了。但是启动摄像头需要的时间会比之前长一些。<br />或者，依旧使用direct show模式，将244行的`self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)`替换为以下代码
```python
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # set buffer size
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
```
第二种方法可能对有些电脑不适用。
