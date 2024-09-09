from ui.newlogin_ui import Ui_MainWindow
import sys
from PyQt5.QtWidgets import *
from datetime import datetime
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from utils.id_utils import get_id_info, sava_id_info, sava_remember_password, get_remember_password# 账号信息工具函数
from lib.share import shareInfo # 公共变量名
from ui.newregiste_ui import Ui_Dialog
from maintest import MainWindow
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMenu, QAction
from PyQt5.QtCore import Qt, QPoint, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QPainter, QIcon
import warnings
import json
warnings.filterwarnings("ignore", category=DeprecationWarning)

class win_Login(QMainWindow):
    def __init__(self, parent = None):
        super(win_Login, self).__init__(parent)
        self.ui_login = Ui_MainWindow()
        self.ui_login.setupUi(self)
        self.init_slots()
        self.hidden_pwd()
        self.m_flag = False
        self.ui_login.checkBox.clicked.connect(self.save_password)
        REMEMBER_PWD, count = get_remember_password()
        print(REMEMBER_PWD)
        keys = REMEMBER_PWD.keys()
        key_list = list(keys)
        last_key = key_list[-1]
        last_value = REMEMBER_PWD[last_key]
        self.ui_login.edit_username.setText(last_key)
        config_file = 'config/fold.json'
        config = json.load(open(config_file, 'r', encoding='utf-8'))
        if config['choose_save'] == "Yes":
            self.ui_login.edit_password.setText(last_value)
            self.ui_login.checkBox.setChecked(True)
        else:
            print(2)

    def save_password(self):
        if self.ui_login.checkBox.isChecked():
            print("Moon")
            config_file = 'config/fold.json'
            config = json.load(open(config_file, 'r', encoding='utf-8'))
            config['choose_save'] = "Yes"
            config_json = json.dumps(config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(config_json)
        else:
            print("DuChen")
            config_file = 'config/fold.json'
            config = json.load(open(config_file, 'r', encoding='utf-8'))
            config['choose_save'] = "No"
            config_json = json.dumps(config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(config_json)

    def mousePressEvent(self, event):
        self.m_Position = event.pos()
        if event.button() == Qt.LeftButton:
            if 0 < self.m_Position.x() < self.ui_login.label_2.pos().x() + self.ui_login.label_2.width() and \
                    0 < self.m_Position.y() < self.ui_login.label_2.pos().y() + self.ui_login.label_2.height():
                self.m_flag = True

    def mouseMoveEvent(self, QMouseEvent):
        if Qt.LeftButton and self.m_flag:
            self.move(QMouseEvent.globalPos() - self.m_Position)

    def mouseReleaseEvent(self, QMouseEvent):
        self.m_flag = False
    # 密码输入框隐藏
    def hidden_pwd(self):
        self.ui_login.edit_password.setEchoMode(QLineEdit.Password)

    # 绑定信号槽
    def init_slots(self):
        self.ui_login.btn_login.clicked.connect(self.onSignIn) # 点击按钮登录
        self.ui_login.edit_password.returnPressed.connect(self.onSignIn) # 按下回车登录
        self.ui_login.btn_register.clicked.connect(self.create_id)

    # 跳转到注册界面
    def create_id(self):
        shareInfo.createWin = win_Register()
        shareInfo.createWin.show()

    # 保存登录日志
    def save_login_log(self, username):
        with open('login_log.txt', 'a', encoding='utf-8') as f:
            f.write(username + '\t log in at' + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\r')

    # 登录
    def onSignIn(self):
        print("You pressed sign in")
        # 从登陆界面获得输入账户名与密码
        username = self.ui_login.edit_username.text().strip()
        print(username)
        password = self.ui_login.edit_password.text().strip()
        print(password)
        # 获得账号信息
        USER_PWD = get_id_info()
        # print(USER_PWD)

        if username not in USER_PWD.keys():
            replay = QMessageBox.warning(self, "登陆失败!", "账号或密码输入错误", QMessageBox.Yes)
        else:
            # 若登陆成功，则跳转主界面
            if USER_PWD.get(username) == password:
                sava_remember_password(username, password)
                self.save_login_log(username)
                print(3)
                print("Jump to main window")
                # # 实例化新窗口
                # # 写法1：
                # self.ui_new = win_Main()
                # # 显示新窗口
                # self.ui_new.show()

                # 写法2：
                # 不用self.ui_new,因为这个子窗口不是从属于当前窗口,写法不好
                # 所以使用公用变量名
                shareInfo.mainWin = MainWindow()
                shareInfo.mainWin.show()
                # 关闭当前窗口
                self.close()
            else:
                replay = QMessageBox.warning(self, "!", "账号或密码输入错误", QMessageBox.Yes)

class win_Register(QDialog):
    def __init__(self, parent = None):
        super(win_Register, self).__init__(parent)
        self.ui_register = Ui_Dialog()
        self.ui_register.setupUi(self)
        self.init_slots()

    def mousePressEvent(self, event):
        self.m_Position = event.pos()
        if event.button() == Qt.LeftButton:
            if 0 < self.m_Position.x() < self.ui_register.label_2.pos().x() + self.ui_register.label_2.width() and \
                    0 < self.m_Position.y() < self.ui_register.label_2.pos().y() + self.ui_register.label_2.height():
                self.m_flag = True

    def mouseMoveEvent(self, QMouseEvent):
        if Qt.LeftButton and self.m_flag:
            self.move(QMouseEvent.globalPos() - self.m_Position)

    def mouseReleaseEvent(self, QMouseEvent):
        self.m_flag = False
    # 绑定槽信号
    def init_slots(self):
        self.ui_register.pushButton_regiser.clicked.connect(self.new_account)
        self.ui_register.pushButton_cancer.clicked.connect(self.cancel)

    # 创建新账户
    def new_account(self):
        print("Create new account")
        USER_PWD = get_id_info()
        # print(USER_PWD)
        new_username = self.ui_register.edit_username.text().strip()
        new_password = self.ui_register.edit_password.text().strip()
        # 判断用户名是否为空
        if new_username == "":
            replay = QMessageBox.warning(self, "!", "账号不准为空", QMessageBox.Yes)
        else:
            # 判断账号是否存在
            if new_username in USER_PWD.keys():
                replay = QMessageBox.warning(self, "!", "账号已存在", QMessageBox.Yes)
            else:
                # 判断密码是否为空
                if new_password == "":
                    replay = QMessageBox.warning(self, "!", "密码不能为空", QMessageBox.Yes)
                else:
                    # 注册成功
                    print("Successful!")
                    sava_id_info(new_username, new_password)
                    replay = QMessageBox.warning(self,  "!", "注册成功！", QMessageBox.Yes)
                    # 关闭界面
                    self.close()
    # 取消注册
    def cancel(self):
        self.close() # 关闭当前界面

if __name__ == "__main__":
    app = QApplication(sys.argv)
    # 利用共享变量名来实例化对象
    shareInfo.loginWin = win_Login()  # 登录界面作为主界面
    shareInfo.loginWin.show()
    sys.exit(app.exec_())

