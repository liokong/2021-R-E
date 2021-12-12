import os
from PyQt5.QtGui import *
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
import sys
from PyQt5 import QtCore, QtGui, QtWidgets
import paramiko
import time
class Ui_MainWindow(object):
    text=''
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(835, 932)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label_img_1 = QtWidgets.QLabel(self.centralwidget)
        self.label_img_1.setGeometry(QtCore.QRect(10, 50, 351, 351))
        self.label_img_1.setMaximumSize(QtCore.QSize(351, 351))
        self.label_img_1.setObjectName("label_img_1")
        self.textBrowser_2 = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser_2.setGeometry(QtCore.QRect(390, 450, 331, 351))
        self.textBrowser_2.setObjectName("textBrowser_2")
        self.textBrowser = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser.setGeometry(QtCore.QRect(10, 410, 721, 31))
        self.textBrowser.setObjectName("textBrowser")
        self.label_img_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_img_2.setGeometry(QtCore.QRect(380, 50, 351, 351))
        self.label_img_2.setMaximumSize(QtCore.QSize(351, 351))
        self.label_img_2.setObjectName("label_img_2")
        self.label_img_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_img_3.setGeometry(QtCore.QRect(10, 450, 351, 351))
        self.label_img_3.setMaximumSize(QtCore.QSize(351, 351))
        self.label_img_3.setObjectName("label_img_3")
        
        self.textEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.textEdit.setGeometry(QtCore.QRect(10, 10, 661, 31))
        self.textEdit.setObjectName("textEdit")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(680, 10, 56, 31))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(390, 810, 331, 31))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_2.clicked.connect(self.start_2)
        #self.textBrowser_3 = QtWidgets.QTextBrowser(self.centralwidget)
        #self.textBrowser_3.setGeometry(QtCore.QRect(10, 810, 351, 31))
        #self.textBrowser_3.setObjectName("textBrowser_3")
        self.pushButton.clicked.connect(self.start)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_img_1.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\">원본 이미지</p></body></html>"))
        self.label_img_2.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\">원본 이미지</p></body></html>"))
        self.label_img_3.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\">원본 이미지</p></body></html>"))
        self.pushButton.setText(_translate("MainWindow", "입력"))
    def start(self):
        text=self.textEdit.text()
        mode = 0

        if text[-3:]=='npy':
            IMG_WIDTH = 256
            IMG_HEIGHT = 256
            IMG_CHANNELS = 1            

            data_path3 = "files/test256npy"         

            X_test = np.zeros((len(os.listdir(data_path3)), IMG_HEIGHT, IMG_WIDTH, 1))        

            count = 0           

            for fh in sorted(os.listdir(data_path3), key=lambda x:int(x[:-4])):
                t = os.path.join(data_path3, fh)
                
                img = np.load(t)            

                X_test[count] = img
                count += 1
            X_test = X_test[:100]

            if len(X_test.shape) == 4:
                mode = 1
                ix=np.random.randint(0, X_test.shape[0], size=4)
            else:
                mode = 0
                X_test = X_test.reshape(1, 256, 256,1)
                ix=np.zeros(1)
            input_ = X_test.copy()
            maxd = 240
            mind = -160
            input_ = np.where(input_>240, maxd,input_)
            input_ = np.where(input_<-160, mind,input_)
            input_ += 160
            input_ = input_ / 400.

        if text[-3:]=='jpg' or text[-3:]=='png':
            mode = 0
            X_test=cv2.imread(text,cv2.IMREAD_GRAYSCALE).reshape(1,256,256, 1)/255
            input_ = X_test.copy()
        if mode:
            img = np.zeros((4,128,128))
            for i in range(4):
                img[i] = cv2.resize(input_[ix[i]], dsize=(128,128), interpolation=cv2.INTER_AREA)
            img1 = np.concatenate((img[0],img[1]), axis=1)
            img2 = np.concatenate((img[2],img[3]), axis=1)
            img = np.concatenate((img1,img2), axis=0)

        cv2.imwrite('files/origin.jpg',img.reshape(256,256)*255)
        self.label_img_1.setPixmap(QtGui.QPixmap("files/origin.jpg"))

        #np.save('files/input1.npy',input_)
        #access_Server('input1.npy', 'spine.npy', 1)
        preds = np.load('files/spine.npy')

        if mode:
            img = np.zeros((4,256,256))
            for i in range(4):
                img[i] = preds[ix[i]].reshape(256,256)
            img1 = np.concatenate((img[0],img[1]), axis=1)
            img2 = np.concatenate((img[2],img[3]), axis=1)
            img = np.concatenate((img1,img2), axis=0)
            img = cv2.resize(img, dsize=(256,256), interpolation=cv2.INTER_AREA)
        else:
            img = preds.copy().reshape(256,256)

        cv2.imwrite('mask.jpg',img*255)
        self.label_img_2.setPixmap(QtGui.QPixmap("mask.jpg"))
        if mode:
            self.textBrowser.append('사진번호:'+ '-'.join(list(map(str,(ix+np.array([1,1,1,1]).tolist())))))

        X_test += 500
        X_test = X_test / 1800.
        input_ = preds*X_test
        
        if mode:
            img = np.zeros((4,256,256))
            for i in range(4):
                img[i] = input_[ix[i]].reshape(256,256)
            img1 = np.concatenate((img[0],img[1]), axis=1)
            img2 = np.concatenate((img[2],img[3]), axis=1)
            img = np.concatenate((img1,img2), axis=0)
            img = cv2.resize(img, dsize=(256,256), interpolation=cv2.INTER_AREA)
        else:
            img = input_.copy()

        cv2.imwrite('files/spine.jpg',img.reshape(256,256)*255)
        self.label_img_3.setPixmap(QtGui.QPixmap("files/spine.jpg"))

        #np.save('files/input2.npy',input_)
        #access_Server('input2.npy', 'result.npy', 0)
        preds = np.load('files/result.npy')
        print(preds)
        if mode:
            index = np.where(preds==1)[0]

            for i in index.tolist():
                self.textBrowser_2.append(f'{int(i)+1}번 이미지에서 골다공증이 관측되었습니다!\n')
        else:
            if preds.astype(np.uint8) == 1:
                self.textBrowser_2.append('골다공증이 관측되었습니다!\n')
            else:
                self.textBrowser_2.append('정상입니다.\n')
        
        
    def start_2(self):
        pass

def access_Server(input_, fileName, mode):
    print(1)
    host = '164.125.37.79'
    username = 'kmj'
    pwd='1407'
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(host, port=522, username=username, password=pwd)
    paramiko.util.log_to_file("paramiko.log")
    sftp = ssh.open_sftp()

    filePath = "./files/" + input_
    sftp.put(filePath, input_)
    print(2)
    stdin, stdout, stderr = ssh.exec_command('source torch-env-3.6/bin/activate')
    lines = stdout.readlines() # 실행한 명령어에 대한 결과 텍스트
    if mode:
        stdin, stdout, stderr = ssh.exec_command('python extract_spine.py')
    else:
        stdin, stdout, stderr = ssh.exec_command('python predict.py')
    lines = stdout.readlines() # 실행한 명령어에 대한 결과 텍스트
    
    time.sleep(1)
    sftp.get(fileName, 'files/' + fileName)

    if ssh: ssh.close()
    

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())


