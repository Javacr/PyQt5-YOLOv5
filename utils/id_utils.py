# -*- coding: utf-8 -*-
# @Modified by: Ruihao
# @ProjectName:yolov5-pyqt5

'''
存放公用的账户读写函数
'''
import csv

# 写入账户信息到csv文件
def sava_id_info(user, pwd):
    headers = ['name', 'key']
    values = [{'name': user, 'key': pwd}]
    with open('userInfo.csv', 'a', encoding='utf-8', newline='') as fp:
        writer = csv.DictWriter(fp, headers)
        writer.writerows(values)

# 读取csv文件获得账户信息
def get_id_info():
    USER_PWD = {}
    with open('userInfo.csv', 'r') as csvfile: # 此目录即是当前项目根目录
        spamreader = csv.reader(csvfile)
        # 逐行遍历csv文件,按照字典存储用户名与密码
        for row in spamreader:
            USER_PWD[row[0]] = row[1]
    return USER_PWD

def sava_remember_password(user, pwd):

    with open('remember_password.csv', 'w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([])

    headers = ['name', 'key']
    values = [{'name': user, 'key': pwd}]
    with open('remember_password.csv', 'a', encoding='utf-8', newline='') as fp:
        writer = csv.DictWriter(fp, headers)
        writer.writerows(values)


def get_remember_password():
    REMEMBER_PWD = {}
    count = 0
    with open('remember_password.csv', 'r') as csvfile: # 此目录即是当前项目根目录
        spamreader = csv.reader(csvfile)
        next(spamreader)
        # 逐行遍历csv文件,按照字典存储用户名与密码
        for row in spamreader:
            REMEMBER_PWD[row[0]] = row[1]
            count += 1

    return REMEMBER_PWD, count