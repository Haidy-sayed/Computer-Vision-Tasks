





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
from Task1GUIEdited import Ui_MainWindow
import pyqtgraph.exporters
import cv2 as cv
from matplotlib import pyplot as plt
from math import sqrt
from PIL import Image
import pyqtgraph.exporters

import cv2
import numpy as np




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

        self.ui.actionExit.triggered.connect(lambda: self.exit() )
        self.ui.actionBrowse_an_image.triggered.connect(lambda: self.browseAnImg())
        self.ui.actionHistogram_Equalization.triggered.connect(lambda: self.histogramRun())
        self.ui.actionFrom_ch_1.triggered.connect(lambda: self.saveImag("Histogram"))
        self.ui.actionFreq_filtered.triggered.connect(lambda : self.saveImag("FreqFilter"))
        self.ui.actionFrom_ch_2.triggered.connect(lambda : self.saveImag("SpatialFilter"))
        self.ui.actionSave_equalized_histogram.triggered.connect(lambda: self.saveImag("EqHistogram"))
        self.ui.actionReversed_image_from_Eq_histo.triggered.connect(lambda: self.saveImag("RevImage"))
        self.ui.actionSpatial_Domain.triggered.connect(lambda: self.setDomain("S"))
        self.ui.actionFrequency_Domain.triggered.connect(lambda: self.setDomain("F"))
        self.ui.actionLow_pass.triggered.connect(lambda: self.filterSelection("LO"))
        self.ui.actionHigh_pass.triggered.connect(lambda: self.filterSelection("HI"))
        self.ui.actionMedium_pass.triggered.connect(lambda: self.filterSelection("MED"))
        self.ui.actionLa_placian.triggered.connect(lambda: self.filterSelection("PLA"))
        self.ui.actionHistogram_Equalization.triggered.connect(lambda: self.histogramRun())
        

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

    #Declaration of any global variables 
        self.logHistory=[] #A list created in order to log every action on the gui from its start till its closed by the user 
        self.ImageXsize=364
        
    #Start of functions 
    def saveLocation(self):
        filePath, _ = QFileDialog.getSaveFileName(self, "Save Image", "",
						"PNG(*.png);;JPEG(*.jpg *.jpeg);;All Files(*.*) ")

        return filePath
    #Save image function
    def saveImag(self, whichScreenText):
        self.logging("The save image function was called")
        path= self.saveLocation()
        if whichScreenText=="Histogram":
            self.logging("The GUI is now saving the histogram as a png image")
            img=self.canvHistogram
            img.savefig(path) 
        elif whichScreenText=="FreqFilter":
            self.logging("The GUI is now saving the Image from the frequency domian as a png image")
            img=self.canvEqualized
            img.save(path) 
        elif whichScreenText=="SpatialFilter":
            self.logging("The GUI is now saving the image from the spatial domain as a png image")
            img=pyqtgraph.exporters.ImageExporter(self.ui.FilterInSpatialDomainWidget.scene())
            img.export(path) 
        elif whichScreenText=="EqHistogram":
            self.logging("The GUI is now saving the image from the equalized histogram as a png image")
            img=pyqtgraph.exporters.ImageExporter(self.ui.verticalLayout_8.scene())
            img.export('The_Image_from_the_spatial_domain.png')
        elif whichScreenText=="RevImage":
            self.logging("The GUI is now saving the returned image from the equalized histogram as a png image")
            img=pyqtgraph.exporters.ImageExporter(self.ui.verticalLayout_7.scene())
            img.export('The_Image_from_the_spatial_domain.png') 
        else:
            self.logging("There must be an error occured here because the function was called without one of the specified texts, check for any bugs here")
            

 

    #The Exit function 
    def exit(self):
        self.logging("Exit function was called")
        sys.exit()

    #The logging function 
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

    #Selection of filter function 
    def filterSelection(self, filterTypeText):
        if self.domain=="Frequency":
            img = cv.imread(self.imagePath,0)
            rows,cols = img.shape
            
            crow,ccol = rows//2 , cols//2
            if filterTypeText=="HI":
                self.frequencydomain("HI",self.imagePath)
                self.fourier_tranf_shift[crow-30:crow+31, ccol-30:ccol+31] = 0

                f_ishift = np.fft.ifftshift(self.fourier_tranf_shift)
                img_back = np.fft.ifft2(f_ishift)
                img_back = np.real(img_back)
                print(type(img_back))
                print(img_back.shape)
                self.setpixmapspatial(img_back)

            elif filterTypeText=="LO":
                self.frequencydomain("LO",self.imagePath)

                self.mask = np.zeros((rows,cols,2),np.uint8)
                self.mask[crow-50:crow+51, ccol-50:ccol+51] = 1
                self.fourier_tranf_shift = self.mask[:,:,0] *self.fourier_tranf_shift
                f_ishift = np.fft.ifftshift(self.fourier_tranf_shift)
                img_back = np.fft.ifft2(f_ishift)
                img_back = np.real(img_back)
                print(type(img_back))
                print(img_back.shape)
                self.setpixmapspatial(img_back)

                

                # self.dft = cv.dft(np.float32(img),flags = cv.DFT_COMPLEX_OUTPUT)
                # self.dft_shift = np.fft.fftshift(self.dft)
                # self.magnitude_spectrum = 20*np.log(cv.magnitude(self.dft_shift[:,:,0],self.dft_shift[:,:,1]))
                # self.setpixmapfourier(self.mask)
                # fshift = self.dft_shift*self.mask
                # f_ishift = np.fft.ifftshift(fshift)
                # img_back = cv2.idft(f_ishift)
                # img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
                # print(type(img_back))
                # print(img_back.shape)
                # self.setpixmapspatial(img_back)
 
                
        elif self.domain=="Spatial":
            if filterTypeText=="HI":
                print("it's not valid")
            elif filterTypeText=="LO":
                print("it's not valid")
            elif filterTypeText=="MED":
                self.final = cv2.medianBlur(self.image, 5)
                self.setpixmap(self.final)
                self.frequencydomain("MED","filteredimage.jpg") 
            elif filterTypeText=="PLA":
                self.final = cv2.Laplacian(self.image, cv2.CV_16S, ksize=3)
                self.abs_final = cv2.convertScaleAbs(self.final)
                self.setpixmap(self.abs_final)
                self.frequencydomain("PLA","filteredimage.jpg")

    def setpixmap(self,image):
        data = Image.fromarray(image)
        data.save('filteredimage.jpg')
        self.ui.filterInSDomianLabel.setPixmap(QPixmap("filteredimage.jpg").scaledToWidth(self.ImageXsize))
        self.ui.filterInSDomianLabel.setScaledContents(True)

    def frequencydomain(self,filter,image):
        self.read_img=cv.imread(image,0)
        self.fourier_tranf = np.fft.fft2(self.read_img)
        self.fourier_tranf_shift = np.fft.fftshift(self.fourier_tranf)
        magnitude_spectrum = 20*np.log(np.abs(self.fourier_tranf_shift))
        self.setpixmapfourier(magnitude_spectrum)
        
    def setpixmapfourier(self,image):
        data = Image.fromarray(image)
        new_p = data.convert("L")
        new_p.save('filteredimage2.png')
        self.ui.FilterInFreqDomainWidget.setPixmap(QPixmap("filteredimage2.png").scaledToWidth(self.ImageXsize))
        self.ui.FilterInFreqDomainWidget.setScaledContents(True)

    def setpixmapspatial(self,image):
        data = Image.fromarray(image)
        new_p = data.convert("L")
        new_p.save('medfilterimage2.png')
        self.ui.FilterInSpatialDomainWidget.setPixmap(QPixmap("medfilterimage2.png").scaledToWidth(self.ImageXsize))
        self.ui.FilterInSpatialDomainWidget.setScaledContents(True)

    #Selecting the domain each time a filter is chosen to allow for different domain selection each time
    def setDomain(self, domianIdentifierChar):
        if domianIdentifierChar=='F':
            self.domain="Frequency"
        else:
            self.domain="Spatial"
    









if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main = mainApp()
    main.show()
    sys.exit(app.exec_())