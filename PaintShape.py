# Python app for painting simple shapes
# Adapted from https://www.geeksforgeeks.org/pyqt5-create-paint-application/
# WNGJIA001, June 2021

# importing libraries
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from PIL import Image, ImageQt
import cv2
import math

# output labels
LABELS = ['circle', 'square', 'diamond', 'star', 'triangle']

# neural network
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        # data has a single input channel, 28x28 images
        # create 6, 5x5 kernels
        # Pytorch does valid padding by default.
        self.conv1 = nn.Conv2d(1, 6, 5)
        # output 6, 24x24 feature maps
        # 2x2 max-pooling
        self.pool = nn.MaxPool2d(2, 2)
        # 6, 12x12 feature maps going out of the pooling stage
        self.conv2 = nn.Conv2d(6, 16, 5)
        # output 16, 8x8 feature maps
        # there will be another pooling stage in the forward pass before fc1
        # output 16, 4x4 feature maps
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load('shapeclassifier.pth')
model.eval()
# transform
test_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),])

def process_image(img):
    # remove row and column at the sides of the image which are blank
    while np.sum(img[0]) == 0:
        img = img[1:]
    while np.sum(img[:,0]) == 0:
        img = np.delete(img, 0, 1)
    while np.sum(img[-1]) == 0:
        img = img[:-1]
    while np.sum(img[:,-1]) == 0:
        img = np.delete(img, -1, 1)
    rows, cols = img.shape
    # resize outer box to fit it into a 90x90 box
    if rows > cols:
        factor = 90.0/rows
        rows = 90
        cols = int(round(cols*factor))
        img = cv2.resize(img, (cols,rows))
    else:
        factor = 90.0/cols
        cols = 90
        rows = int(round(rows*factor))
        img = cv2.resize(img, (cols, rows))

    colsPadding = (int(math.ceil((100-cols)/2.0)), int(math.floor((100-cols)/2.0)))
    rowsPadding = (int(math.ceil((100-rows)/2.0)), int(math.floor((100-rows)/2.0)))
    img = np.lib.pad(img, (rowsPadding,colsPadding), 'constant')

    return img

def predict_image(image):
    image_tensor = test_transforms(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input = Variable(image_tensor)
    input = input.to(device).cpu()
    output = model(input)
    index = output.data.cpu().numpy().argmax()
    return LABELS[index]

# window class
class Window(QMainWindow):
    def __init__(self):
        super().__init__()

        # setting title
        self.setWindowTitle("Simple Shape Classifier")
        # setting geometry to main window
        self.setGeometry(100, 100, 280, 280)
        self.setFixedSize(280, 280)
        # creating image object
        self.image = QImage(self.size(), QImage.Format_RGB32)
        # making image color to white
        self.image.fill(Qt.white)

        # variables
        # drawing flag
        self.drawing = False
        # QPoint object to tract the point
        self.lastPoint = QPoint()

        # creating menu bar
        mainMenu = self.menuBar()
        # creating file menu for save and clear action
        fileMenu = mainMenu.addMenu("File")

        # creating predict action
        predictAction = QAction("Predict", self)
        predictAction.setShortcut("Ctrl + P")
        fileMenu.addAction(predictAction)
        predictAction.triggered.connect(self.predict)

        # creating save action
        saveAction = QAction("Save", self)
        saveAction.setShortcut("Ctrl + S")
        fileMenu.addAction(saveAction)
        saveAction.triggered.connect(self.save)

        # creating clear action
        clearAction = QAction("Clear", self)
        clearAction.setShortcut("Ctrl + C")
        fileMenu.addAction(clearAction)
        clearAction.triggered.connect(self.clear)

        # creating push buttons
        predictButton = QPushButton("Predict", self)
        predictButton.setShortcut("Ctrl + P")
        predictButton.clicked.connect(self.predict)
        predictButton.move(140, 250)

        # creating push buttons
        clearButton = QPushButton("Clear", self)
        clearButton.setShortcut("Ctrl + C")
        clearButton.clicked.connect(self.clear)
        clearButton.move(40, 250)

    # method for checking mouse cicks
    def mousePressEvent(self, event):
        # if left mouse button is pressed
        if event.button() == Qt.LeftButton:
            # make drawing flag true
            self.drawing = True
            # make last point to the point of cursor
            self.lastPoint = event.pos()

    # method for tracking mouse activity
    def mouseMoveEvent(self, event):
        # checking if left button is pressed and drawing flag is true
        if (event.buttons() & Qt.LeftButton) & self.drawing:
            # creating painter object
            painter = QPainter(self.image)
            # set the pen of the painter
            painter.setPen(QPen(Qt.black, 16, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            # draw line from the last point of cursor to the current point
            # this will draw only one step
            painter.drawLine(self.lastPoint, event.pos())
            # change the last point
            self.lastPoint = event.pos()
            # update
            self.update()

    # method for mouse left button release
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            # make drawing flag false
            self.drawing = False

    # paint event
    def paintEvent(self, event):
        # create a canvas
        canvasPainter = QPainter(self)
        # draw rectangle on the canvas
        canvasPainter.drawImage(self.rect(), self.image, self.image.rect())

    # method for predicting canvas
    def predict(self):
        # preprocess
        image = ImageQt.fromqimage(self.image).convert('L')
        image = np.array(image)
        resized = cv2.resize(255-image, (140, 140))
        resized = process_image(resized)
        resized = cv2.resize(resized, (50, 50))
        resized = cv2.resize(255-resized, (28, 28))
        cv2.imshow('image', resized)
        # predict
        print("Classifier:", predict_image(resized))

    # method for saving canvas
    def save(self):
        filePath, _ = QFileDialog.getSaveFileName(self, "Save Image", "",
                          "PNG(*.png);;JPEG(*.jpg *.jpeg);;All Files(*.*) ")
        if filePath == "":
            return
        self.image.save(filePath)

    # method for clearing every thing on canvas
    def clear(self):
        # make the whole canvas white
        self.image.fill(Qt.white)
        # update
        self.update()

# main method
if __name__ == "__main__":
    # create pyqt5 app
    App = QApplication(sys.argv)
    # create the instance of our Window
    window = Window()
    # showing the wwindow
    window.show()

    sys.exit(App.exec())
