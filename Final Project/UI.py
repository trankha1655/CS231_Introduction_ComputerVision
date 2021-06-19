from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QFileDialog
import cv2
import process
class Ui_MainWindow(object):
    def __init__(self):
        self.img = None
        self.path_img = None

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(443, 529)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(0, 70, 461, 61))
        self.label.setObjectName("label")
        self.capture_image = QtWidgets.QPushButton(self.centralwidget)
        self.capture_image.setGeometry(QtCore.QRect(80, 220, 151, 28))
        self.capture_image.setObjectName("capture_image")
        self.select_image = QtWidgets.QPushButton(self.centralwidget)
        self.select_image.setGeometry(QtCore.QRect(80, 300, 151, 28))
        self.select_image.setObjectName("select_image")
        self.rotate = QtWidgets.QPushButton(self.centralwidget)
        self.rotate.setGeometry(QtCore.QRect(290, 370, 93, 28))
        self.rotate.setObjectName("rotate")
        self.path = QtWidgets.QLineEdit(self.centralwidget)
        self.path.setGeometry(QtCore.QRect(82, 340, 301, 22))
        self.path.setObjectName("path")
        self.preview = QtWidgets.QPushButton(self.centralwidget)
        self.preview.setGeometry(QtCore.QRect(80, 370, 93, 28))
        self.preview.setObjectName("preview")
        self.solve = QtWidgets.QPushButton(self.centralwidget)
        self.solve.setGeometry(QtCore.QRect(180, 420, 93, 28))
        self.solve.setObjectName("solve")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 443, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.new_solve = QtWidgets.QPushButton(self.centralwidget)
        self.new_solve.setGeometry(QtCore.QRect(180, 150, 93, 28))
        self.new_solve.setObjectName("new_solve")

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.select_image.clicked.connect(self.select_button)
        self.preview.clicked.connect(self.preview_image)
        self.rotate.clicked.connect(self.rotate_image)
        self.solve.clicked.connect(self.solve_sudoku)
        self.capture_image.clicked.connect(self.capture)
        self.new_solve.clicked.connect(self.new)


    def select_button(self):
        title = 'Open image'
        path = "D:"
        filter = "Image(*.jpeg *.png *.jpg)"
        fileName = QFileDialog.getOpenFileName(None, title, path, filter)
        self.path_img = fileName[0]
        self.path.setText(self.path_img)
        self.img = cv2.imread(self.path.text())

    def preview_image(self):
        if self.path_img != None:
            cv2.imshow('Preview Image', self.img)
            cv2.waitKey(0)

    def rotate_image(self):
        if self.path_img != None:
            self.img = cv2.rotate(self.img,cv2.cv2.ROTATE_90_CLOCKWISE)
            cv2.imshow('Preview Image', self.img)
            cv2.waitKey(0)

    def solve_sudoku(self):
        if self.path_img != None:
            image = process.result(self.img)
            image.main()

    def capture(self):
        cam = cv2.VideoCapture(0)
        cv2.namedWindow("Camera")
        while True:
            ret, frame = cam.read()
            cv2.imshow("Camera", frame)
            k = cv2.waitKey(1)
            if k % 256 == 27:
                break
            if k % 256 == 32:
                # SPACE pressed
                self.img = frame
                break
        cam.release()
        cv2.destroyAllWindows()
        cv2.imshow('Sudoku', frame)
        image = process.result(self.img)
        image.main()


    def new(self):
        self.path.setText("")
        self.img = None
        self.path_img = None
        cv2.destroyAllWindows()

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:20pt;\">AI Sudoku Solver</span></p></body></html>"))
        self.capture_image.setText(_translate("MainWindow", "Capture image"))
        self.select_image.setText(_translate("MainWindow", "Select image"))
        self.rotate.setText(_translate("MainWindow", "Rotate"))
        self.preview.setText(_translate("MainWindow", "Preview"))
        self.solve.setText(_translate("MainWindow", "Solve"))
        self.new_solve.setText(_translate("MainWindow", "New"))


