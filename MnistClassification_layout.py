# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'MnistClassification_GUI.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(612, 516)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("icon.jpg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        self.lbDataArea = QtWidgets.QLabel(MainWindow)
        self.lbDataArea.setGeometry(QtCore.QRect(370, 200, 224, 224))
        self.lbDataArea.setMouseTracking(False)
        self.lbDataArea.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.lbDataArea.setFrameShape(QtWidgets.QFrame.Box)
        self.lbDataArea.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.lbDataArea.setLineWidth(4)
        self.lbDataArea.setMidLineWidth(0)
        self.lbDataArea.setText("")
        self.lbDataArea.setObjectName("lbDataArea")
        self.verticalLayoutWidget = QtWidgets.QWidget(MainWindow)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(370, 200, 221, 221))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.dArea_Layout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.dArea_Layout.setContentsMargins(0, 0, 0, 0)
        self.dArea_Layout.setSpacing(0)
        self.dArea_Layout.setObjectName("dArea_Layout")
        self.PredictButton = QtWidgets.QPushButton(MainWindow)
        self.PredictButton.setGeometry(QtCore.QRect(270, 610, 91, 51))
        self.PredictButton.setObjectName("PredictButton")
        self.WriteButton = QtWidgets.QPushButton(MainWindow)
        self.WriteButton.setGeometry(QtCore.QRect(190, 320, 141, 51))
        self.WriteButton.setObjectName("WriteButton")
        self.LoadParamsButton = QtWidgets.QPushButton(MainWindow)
        self.LoadParamsButton.setGeometry(QtCore.QRect(30, 190, 81, 41))
        self.LoadParamsButton.setObjectName("LoadParamsButton")
        self.PredictButton = QtWidgets.QPushButton(MainWindow)
        self.PredictButton.setGeometry(QtCore.QRect(130, 390, 91, 51))
        self.PredictButton.setObjectName("PredictButton_2")
        self.TestButton = QtWidgets.QPushButton(MainWindow)
        self.TestButton.setGeometry(QtCore.QRect(30, 250, 71, 41))
        self.TestButton.setObjectName("TestButton")
        self.ResultLabel = QtWidgets.QLabel(MainWindow)
        self.ResultLabel.setGeometry(QtCore.QRect(220, 150, 111, 111))
        font = QtGui.QFont()
        font.setFamily("Bahnschrift SemiLight")
        font.setPointSize(40)
        self.ResultLabel.setFont(font)
        self.ResultLabel.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.ResultLabel.setObjectName("ResultLabel")
        self.RandomGetImageButton = QtWidgets.QPushButton(MainWindow)
        self.RandomGetImageButton.setGeometry(QtCore.QRect(30, 320, 141, 51))
        self.RandomGetImageButton.setObjectName("RandomGetImageButton")
        self.TrainButton = QtWidgets.QPushButton(MainWindow)
        self.TrainButton.setGeometry(QtCore.QRect(30, 130, 71, 41))
        self.TrainButton.setObjectName("TrainButton")
        self.label = QtWidgets.QLabel(MainWindow)
        self.label.setGeometry(QtCore.QRect(150, 200, 51, 31))
        self.label.setObjectName("label")
        self.Title = QtWidgets.QLabel(MainWindow)
        self.Title.setGeometry(QtCore.QRect(200, 30, 201, 31))
        self.Title.setObjectName("Title")

        self.retranslateUi(MainWindow)

        self.TrainButton.clicked.connect(MainWindow.TrainButton_Click)
        self.LoadParamsButton.clicked.connect(MainWindow.LoadParamsButton_Click)
        self.TestButton.clicked.connect(MainWindow.TestButton_Click)
        self.RandomGetImageButton.clicked.connect(MainWindow.RandomGetImageButton_Click)
        self.WriteButton.clicked.connect(MainWindow.WriteButton_Click)
        self.PredictButton.clicked.connect(MainWindow.PredictButton_Click)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)



    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MnistClassification v2.0 --by Chino"))
        self.PredictButton.setText(_translate("MainWindow", "预测"))
        self.WriteButton.setText(_translate("MainWindow", "手写模式"))
        self.LoadParamsButton.setText(_translate("MainWindow", "加载已有模型"))
        self.PredictButton.setText(_translate("MainWindow", "预测"))
        self.TestButton.setText(_translate("MainWindow", "测试模型"))
        self.ResultLabel.setText(_translate("MainWindow", "nan"))
        self.RandomGetImageButton.setText(_translate("MainWindow", "从测试集中随机抽取图片"))
        self.TrainButton.setText(_translate("MainWindow", "训练模型"))
        self.label.setText(_translate("MainWindow", "识别结果"))
        self.Title.setText(_translate("MainWindow", "基于深度学习的Mnist手写体识别"))
