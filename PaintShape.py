# Python app for painting simple shapes
# Adapted from https://www.geeksforgeeks.org/pyqt5-create-paint-application/
# WNGJIA001, June 2021

# importing libraries
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import sys

# window class
class Window(QMainWindow):
    def __init__(self):
        super().__init__()

        # setting title
        self.setWindowTitle("Simple Shape Classifier")
        # setting geometry to main window
        self.setGeometry(100, 100, 560, 560)
        self.setFixedSize(560, 560)
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
        # adding short cut for predict action
        predictAction.setShortcut("Ctrl + P")
        # adding predict to the file menu
        fileMenu.addAction(predictAction)
        # adding action to the save
        predictAction.triggered.connect(self.predict)

        # creating save action
        saveAction = QAction("Save", self)
        # adding short cut for save action
        saveAction.setShortcut("Ctrl + S")
        # adding save to the file menu
        fileMenu.addAction(saveAction)
        # adding action to the save
        saveAction.triggered.connect(self.save)

        # creating clear action
        clearAction = QAction("Clear", self)
        # adding short cut to the clear action
        clearAction.setShortcut("Ctrl + C")
        # adding clear to the file menu
        fileMenu.addAction(clearAction)
        # adding action to the clear
        clearAction.triggered.connect(self.clear)

        # creating push buttons
        predictButton = QPushButton("Predict", self)
        predictButton.setShortcut("Ctrl + P")
        predictButton.clicked.connect(self.predict)
        predictButton.move(300, 500)

        # creating push buttons
        clearButton = QPushButton("Clear", self)
        clearButton.setShortcut("Ctrl + C")
        clearButton.clicked.connect(self.clear)
        clearButton.move(150, 500)

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
            painter.setPen(QPen(Qt.black, 4, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
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
        # load model
        # load canvas for preprocessing
        # predict
        # show output
        pass

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
    # start the app
    sys.exit(App.exec())
