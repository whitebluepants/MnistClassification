import sys
import matplotlib.pyplot as plt
from PIL import Image, ImageQt
from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5.QtGui import QPixmap, QColor
from PyQt5.QtCore import QSize

from Mnist.paintboard import PaintBoard
from Mnist.MnistClassification_layout import Ui_MainWindow
from Mnist import Mnist_Classification
import mxnet as mx
import numpy as np
from mxnet import nd
import random
import cv2

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        # 初始化UI
        self.setupUi(self)

        # 初始化画板
        self.paintBoard = PaintBoard(self, Size=QSize(224, 224), Fill=QColor(0, 0, 0, 0))
        self.paintBoard.setPenColor(QColor(0, 0, 0, 0))
        self.dArea_Layout.addWidget(self.paintBoard)

        self.net = None
        self.image = None
        self.flag1 = 0  # 判断是否有训练模型
        self.flag2 = 0  # 判断是否有加载图片
        self.mode = 0
        self.file_dir = None
        self.ctx = mx.gpu()
        self.clearDataArea()

    # 清除数据待输入区
    def clearDataArea(self):
        self.paintBoard.Clear()
        self.lbDataArea.clear()

    def TrainButton_Click(self):
        self.net = Mnist_Classification.train(load_params=False)
        self.flag1 = 1

    def LoadParamsButton_Click(self):
        self.net = Mnist_Classification.train(load_params=True)
        self.flag1 = 1
        print("模型已加载!!\n模型已加载!!\n模型已加载!!\n模型已加载!!\n模型已加载!!\n")

    def TestButton_Click(self):
        if self.flag1 == 0:
            print("请先训练好模型!\n请先训练好模型!\n请先训练好模型!\n请先训练好模型!\n请先训练好模型!")
        else:
            Mnist_Classification.test(self.net)

    def RandomGetImageButton_Click(self):
        self.clearDataArea()
        self.paintBoard.setBoardFill(QColor(0, 0, 0, 0))
        self.paintBoard.setPenColor(QColor(0, 0, 0, 0))

        index = random.randint(0, 9999)
        self.file_dir = 'Mnist_test/' + str(index) + '.png'
        pix = QPixmap(self.file_dir)
        pix = pix.scaled(224, 224)
        self.lbDataArea.setPixmap(pix)
        self.mode = 1

    def WriteButton_Click(self):
        self.clearDataArea()
        self.paintBoard.setBoardFill(QColor(0, 0, 0, 255))
        self.paintBoard.setPenColor(QColor(255, 255, 255, 255))
        self.mode = 2

    def PredictButton_Click(self):
        image = None
        if self.mode == 0:
            print("请先加载图片\n请先加载图片\n请先加载图片\n请先加载图片\n请先加载图片\n")
        elif self.mode == 1:
            image = cv2.imread(self.file_dir, cv2.IMREAD_GRAYSCALE)
            image = image.reshape(28, 28, 1) / 255.0
        elif self.mode == 2:
            img = self.paintBoard.getContentAsQImage()
            qimage = ImageQt.fromqimage(img)
            qimage = qimage.resize((28, 28), Image.ANTIALIAS)
            image = np.array(qimage.convert('L')).reshape(28, 28, 1) / 255.0

        image = np.transpose(image, [2, 0, 1])
        nparray = np.zeros(shape=(1, 1, 28, 28))
        nparray[0] = image
        nda = nd.array(nparray)
        img = nda.as_in_context(self.ctx)

        y_pred = Mnist_Classification.Predict(self.net, img)
        self.ResultLabel.setText(str(int(y_pred)))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    Gui = MainWindow()
    Gui.show()

    sys.exit(app.exec_())