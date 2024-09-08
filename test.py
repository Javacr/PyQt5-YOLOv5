# -*- coding: utf-8 -*-
# @Time : 2020/4/18 19:36
# @Author : Zhao HL
# @File : 14_multThread.py
import sys
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *


class Example(QWidget):
    def __init__(self):
        super(Example, self).__init__()
        self.v_layout = QVBoxLayout()
        self.setLayout(self.v_layout)

        self.bl1 = QLabel('0', self)
        self.bl1.setAlignment(Qt.AlignCenter)
        self.bl2 = QLabel('0', self)
        self.bl2.setAlignment(Qt.AlignCenter)

        self.btn_start1 = QPushButton('start1', clicked=self.start1)
        self.btn_start2 = QPushButton('start2', clicked=self.start2)
        self.btn_stop1 = QPushButton('stop1', clicked=self.stop1)
        self.btn_stop2 = QPushButton('stop2', clicked=self.stop2)

        self.hbox1 = QHBoxLayout()
        self.hbox2 = QHBoxLayout()

        self.hbox1.addWidget(self.btn_start1)
        self.hbox1.addWidget(self.btn_stop1)
        self.hbox2.addWidget(self.btn_start2)
        self.hbox2.addWidget(self.btn_stop2)

        self.v_layout.addWidget(self.bl1)
        self.v_layout.addWidget(self.bl2)
        self.v_layout.addLayout(self.hbox1)
        self.v_layout.addLayout(self.hbox2)

        self.thd1 = MyThread(1)
        self.thd1.signal.connect(self.process1)
        self.thd2 = MyThread(2)
        self.thd2.signal.connect(self.process2)
        self.show()

    def process1(self, content):
        self.bl1.setText(content)

    def process2(self, content):
        self.bl2.setText(content)

    def start1(self):
        self.thd1.start()
        self.thd1.thd_on = True

    def start2(self):
        self.thd2.start()
        self.thd2.thd_on = True

    def stop1(self):
        self.thd1.thd_on = False
        self.thd1.exit()

    def stop2(self):
        self.thd2.thd_on = False
        self.thd2.exit()


class MyThread(QThread):  # 线程类
    signal = pyqtSignal(str)

    def __init__(self, gap):
        super(MyThread, self).__init__()
        self.count = 0
        self.thd_on = True
        self.gap = gap

    def run(self):  # 线程执行函数
        while self.thd_on:
            print(self.count)
            self.count += self.gap
            self.signal.emit(str(self.count))
            self.sleep(1)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    example = Example()
    sys.exit(app.exec_())