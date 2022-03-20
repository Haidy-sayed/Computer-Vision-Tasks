





from PyQt5 import QtCore, QtWidgets
from PyQt5 import QtGui
import sys
import cv2
from PyQt5.QtWidgets import QFileDialog , QLabel
from PyQt5.QtGui import QPixmap
import pyqtgraph
from pyqtgraph import *
import matplotlib
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
import numpy as np
from PIL import Image
from Task1GUIEdited import Ui_MainWindow




matplotlib.use('QT5Agg')

class MatplotlibCanvas(FigureCanvasQTAgg):
	def __init__(self,parent=None, dpi = 120):
		fig = Figure(dpi = dpi)
		self.axes = fig.add_subplot(111)
		super(MatplotlibCanvas,self).__init__(fig)
		fig.tight_layout()


class mainApp(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(mainApp, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.ui.actionBrowse_an_image.triggered.connect(lambda: self.browseAnImg())
        self.ui.actionHistogram_Equalization.triggered.connect(lambda: self.histogramRun())
        self.ui.actionSave_Histogram.triggered.connect(lambda: self.saveImag("Histogram"))



        self.canvHistogram = MatplotlibCanvas(self)
        self.loadimgcanv=MatplotlibCanvas(self)
        self.canvEqualized =MatplotlibCanvas(self)
        self.canvNewImage =MatplotlibCanvas(self)
        self.ui.verticalLayout_5.addWidget(self.loadimgcanv)
        self.ui.verticalLayout_6.addWidget(self.canvHistogram)
        self.ui.verticalLayout_8.addWidget(self.canvEqualized)
        self.ui.verticalLayout_7.addWidget(self.canvNewImage)


        self.loadimgcanv.axes.axis('off')
        self.canvNewImage.axes.axis('off')

        self.logHistory = []
        self.ImageXsize = 364

    def logging(self, text):
        f=open("Task1CVLog.txt","w+")
        self.logHistory.append(text)
        for i in self.logHistory:
            f.write("=> %s\r\n" %(i))
        f.close()

    def browseAnImg(self):
        self.logging("browseAnImg function was called")
        image=QFileDialog.getOpenFileName()
        self.logging("Image path was chosen from the dialog box")
        self.imagePath = image[0]
        print(self.imagePath)
        self.logging("image path is set to "+self.imagePath)
        self.image = cv2.imread(self.imagePath, 0)
        self.loadimgcanv.axes.imshow(self.image, cmap=plt.get_cmap('gray'))
        self.loadimgcanv.draw()
        # pixmap = QPixmap(self.imagePath)
        # self.ui.label.setPixmap(QPixmap(pixmap).scaledToWidth(self.ImageXsize))
        # self.ui.label.setScaledContents(True)
        # print(self.ImageXsize)
        # print(pixmap.size())
        # self.ui.label.show()

    def make_histogram(self, image):

        # Take a flattened greyscale image and create a historgram from it
        self.imageasArray = np.array(image).flatten()
        self.histogram = np.zeros(256, dtype=int)
        for i in range(image.size):
            self.histogram[self.imageasArray[i]] += 1
        return self.histogram



    def histogramRun(self):
        self.image = cv2.imread(self.imagePath, 0)
        IMG_H, IMG_W = self.image.shape
        self.HistogramResult=self.make_histogram(self.image)


        # Create an array that represents the cumulative sum of the histogram
        CDF = np.zeros(256, dtype=int)
        CDF[0] = self.histogram[0]
        for i in range(1, self.histogram.size):
            CDF[i] = CDF[i - 1] + self.histogram[i]

        #  Create a mapping
        #  each old colour value is mapped to a new one between 0 and 255.
        #  Mapping is created using: M(i) =  cumulative sum /(h * w)) * (grey_levels)) - 1
        #  where g_levels is the number of grey levels in the image

        mapping = np.zeros(256, dtype=int)
        grey_levels = 256
        for i in range(grey_levels):
            mapping[i] = ((CDF[i] / (IMG_H * IMG_W)) * (grey_levels)) - 1

        #Apply the mapping to our image
        self.new_image = np.zeros(self.imageasArray.size, dtype=int)
        for i in range(self.imageasArray.size):
            self.new_image[i] = mapping[self.imageasArray[i]]

        self.HistogramEqualized=self.make_histogram(self.new_image)

        output_image = Image.fromarray(np.uint8(self.new_image.reshape((IMG_H, IMG_W))))


        x_axis = np.arange(256)
        self.canvHistogram.axes.cla()
        self.canvEqualized.axes.cla()
        self.canvNewImage.axes.cla()
        self.canvHistogram.axes.bar(x_axis, self.HistogramResult)
        self.canvEqualized.axes.bar(x_axis, self.HistogramEqualized)
        self.canvNewImage.axes.imshow(output_image, cmap=plt.get_cmap('gray'))
        self.canvHistogram.draw()
        self.canvEqualized.draw()
        self.canvNewImage.draw()













if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main = mainApp()
    main.show()
    sys.exit(app.exec_())