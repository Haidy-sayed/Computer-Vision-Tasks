from PyQt5 import QtCore, QtWidgets
from PyQt5 import QtGui
import sys
import cv2
from PyQt5.QtWidgets import QFileDialog, QLabel
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
import cv2

matplotlib.use('QT5Agg')


class MatplotlibCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, dpi=120):
        fig = Figure(dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MatplotlibCanvas, self).__init__(fig)
        fig.tight_layout()


class mainApp(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(mainApp, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.ui.actionBrowse_an_image.triggered.connect(lambda: self.browseAnImg())
        self.ui.actionHistogram_Equalization.triggered.connect(lambda: self.histogramRun())
        self.ui.actionSave_Histogram.triggered.connect(lambda: self.saveImag("Histogram"))
        self.ui.actionSpatial_Domain.triggered.connect(lambda: self.setDomain("S"))
        self.ui.actionFrequency_Domain.triggered.connect(lambda: self.setDomain("F"))
        self.ui.actionLow_pass.triggered.connect(lambda: self.filterSelection("LO"))
        self.ui.actionHigh_pass.triggered.connect(lambda: self.filterSelection("HI"))
        self.ui.actionMedium_pass.triggered.connect(lambda: self.filterSelection("MED"))
        self.ui.actionLa_placian.triggered.connect(lambda: self.filterSelection("PLA"))
        self.ui.actionExit.triggered.connect(lambda: self.exit())
        self.ui.actionFrom_ch_1.triggered.connect(lambda: self.saveImag("Histogram"))
        self.ui.actionFreq_filtered.triggered.connect(lambda: self.saveImag("FreqFilter"))
        self.ui.actionFrom_ch_2.triggered.connect(lambda: self.saveImag("SpatialFilter"))
        self.ui.actionSave_equalized_histogram.triggered.connect(lambda: self.saveImag("EqHistogram"))
        self.ui.actionReversed_image_from_Eq_histo.triggered.connect(lambda: self.saveImag("RevImage"))

        self.canvHistogram = MatplotlibCanvas(self)
        self.loadimgcanv = MatplotlibCanvas(self)
        self.canvEqualized = MatplotlibCanvas(self)
        self.canvNewImage = MatplotlibCanvas(self)
        self.canvfilter = MatplotlibCanvas(self)
        self.canvSDomain = MatplotlibCanvas(self)
        self.ui.verticalLayout_5.addWidget(self.loadimgcanv)
        self.ui.verticalLayout_6.addWidget(self.canvHistogram)
        self.ui.verticalLayout_8.addWidget(self.canvEqualized)
        self.ui.verticalLayout_7.addWidget(self.canvNewImage)
        self.ui.verticalLayout_10.addWidget(self.canvfilter)
        self.ui.verticalLayout_9.addWidget(self.canvSDomain)

        self.loadimgcanv.axes.axis('off')
        self.canvNewImage.axes.axis('off')
        self.canvfilter.axes.axis('off')
        self.canvSDomain.axes.axis('off')

        self.logHistory = []
        self.ImageXsize = 364

        # FIX: initialize domain, imagePath and image so filters/histogram
        # don't crash if clicked before loading an image or setting a domain
        self.domain = "Spatial"
        self.imagePath = None
        self.image = None

    def logging(self, text):
        f = open("Task1CVLog.txt", "w+")
        self.logHistory.append(text)
        for i in self.logHistory:
            f.write("=> %s\r\n" % (i))
        f.close()

    def browseAnImg(self):
        self.logging("browseAnImg function was called")
        image = QFileDialog.getOpenFileName()
        self.logging("Image path was chosen from the dialog box")
        self.imagePath = image[0]
        print(self.imagePath)
        self.logging("image path is set to " + self.imagePath)
        self.image = cv2.imread(self.imagePath, 0)
        self.loadimgcanv.axes.imshow(self.image, cmap=plt.get_cmap('gray'))
        self.loadimgcanv.draw()

    def make_histogram(self, image):
        # Take a flattened greyscale image and create a histogram from it
        self.imageasArray = np.array(image).flatten()
        self.histogram = np.zeros(256, dtype=int)
        for i in range(image.size):
            self.histogram[self.imageasArray[i]] += 1
        return self.histogram

    def histogramRun(self):
        # FIX: guard against running before an image is loaded
        if self.imagePath is None:
            print("Please load an image first")
            return

        self.image = cv2.imread(self.imagePath, 0)
        IMG_H, IMG_W = self.image.shape

        # FIX: store result explicitly so self.histogram and self.HistogramResult
        # are clearly the same object — no silent side-effect dependency
        self.HistogramResult = self.make_histogram(self.image)

        # Create cumulative distribution function
        CDF = np.zeros(256, dtype=int)
        CDF[0] = self.HistogramResult[0]
        for i in range(1, self.HistogramResult.size):
            CDF[i] = CDF[i - 1] + self.HistogramResult[i]

        # Create mapping: M(i) = (CDF(i) / (h * w)) * grey_levels - 1
        mapping = np.zeros(256, dtype=int)
        grey_levels = 256
        for i in range(grey_levels):
            mapping[i] = ((CDF[i] / (IMG_H * IMG_W)) * grey_levels) - 1

        # Apply mapping to image
        self.new_image = np.zeros(self.imageasArray.size, dtype=int)
        for i in range(self.imageasArray.size):
            self.new_image[i] = mapping[self.imageasArray[i]]

        self.HistogramEqualized = self.make_histogram(self.new_image)
        output_image = Image.fromarray(np.uint8(self.new_image.reshape((IMG_H, IMG_W))))

        x_axis = np.arange(256)
        self.canvHistogram.axes.cla()
        self.canvEqualized.axes.cla()
        self.canvNewImage.axes.cla()
        self.canvNewImage.axes.axis('off')
        self.canvHistogram.axes.bar(x_axis, self.HistogramResult)
        self.canvEqualized.axes.bar(x_axis, self.HistogramEqualized)
        self.canvNewImage.axes.imshow(output_image, cmap=plt.get_cmap('gray'))
        self.canvHistogram.draw()
        self.canvEqualized.draw()
        self.canvNewImage.draw()

    def filterSelection(self, filterTypeText):
        # FIX: guard against running before an image is loaded
        if self.imagePath is None:
            print("Please load an image first")
            return

        if self.domain == "Frequency":
            img = cv2.imread(self.imagePath, 0)
            rows, cols = img.shape
            crow, ccol = rows // 2, cols // 2

            if filterTypeText == "HI":
                # FIX: zero the centre BEFORE displaying so the spectrum
                # shown matches the filter actually applied
                self.frequencydomain("HI", self.imagePath)
                self.fourier_tranf_shift[crow - 30:crow + 31, ccol - 30:ccol + 31] = 0
                self.setpixmapfourier(
                    20 * np.log(np.abs(self.fourier_tranf_shift) + 1e-10)
                )
                f_ishift = np.fft.ifftshift(self.fourier_tranf_shift)
                img_back = np.fft.ifft2(f_ishift)
                img_back = np.real(img_back)
                self.setpixmapspatial(img_back)

            elif filterTypeText == "LO":
                self.frequencydomain("LO", self.imagePath)

                # FIX: mask should be 2D to match the 2D complex fourier array
                self.mask = np.zeros((rows, cols), np.uint8)
                self.mask[crow - 50:crow + 51, ccol - 50:ccol + 51] = 1
                self.fourier_tranf_shift = self.mask * self.fourier_tranf_shift

                f_ishift = np.fft.ifftshift(self.fourier_tranf_shift)
                img_back = np.fft.ifft2(f_ishift)
                img_back = np.real(img_back)
                self.setpixmapspatial(img_back)

        elif self.domain == "Spatial":
            if filterTypeText == "HI":
                print("High pass is not valid in Spatial domain")
            elif filterTypeText == "LO":
                print("Low pass is not valid in Spatial domain")
            elif filterTypeText == "MED":
                self.final = cv2.medianBlur(self.image, 5)
                self.setpixmap(self.final)
                # FIX: pass the array directly instead of a disk round-trip
                self.frequencydomain("MED", self.final)
            elif filterTypeText == "PLA":
                self.final = cv2.Laplacian(self.image, cv2.CV_16S, ksize=3)
                self.abs_final = cv2.convertScaleAbs(self.final)
                self.setpixmap(self.abs_final)
                # FIX: pass the array directly instead of a disk round-trip
                self.frequencydomain("PLA", self.abs_final)

    def setDomain(self, domainIdentifierChar):
        if domainIdentifierChar == 'F':
            self.domain = "Frequency"
        else:
            self.domain = "Spatial"

    def frequencydomain(self, filter, image):
        # FIX: accept either a numpy array or a file path
        if isinstance(image, np.ndarray):
            self.read_img = image.astype(np.uint8)
        else:
            self.read_img = cv2.imread(image, 0)

        self.fourier_tranf = np.fft.fft2(self.read_img)
        self.fourier_tranf_shift = np.fft.fftshift(self.fourier_tranf)
        magnitude_spectrum = 20 * np.log(np.abs(self.fourier_tranf_shift) + 1e-10)
        self.setpixmapfourier(magnitude_spectrum)

    def setpixmapfourier(self, image):
        data = Image.fromarray(image)
        new_p = data.convert("L")
        new_p.save('filteredimage2.png')

        if self.domain == "Frequency":
            self.canvSDomain.axes.cla()
            self.canvSDomain.axes.imshow(new_p, cmap=plt.get_cmap('gray'))
            self.canvSDomain.draw()
            self.canvfilter.axes.cla()
            self.canvfilter.axes.imshow(new_p, cmap=plt.get_cmap('gray'))
            self.canvfilter.draw()

        if self.domain == "Spatial":
            self.canvfilter.axes.cla()
            self.canvfilter.axes.axis('off')
            self.canvfilter.axes.imshow(new_p, cmap=plt.get_cmap('gray'))
            self.canvfilter.draw()

    def setpixmapspatial(self, image):
        data = Image.fromarray(image)
        new_p = data.convert("L")
        new_p.save('medfilterimage2.png')

        self.canvSDomain.axes.cla()
        self.canvSDomain.axes.axis('off')
        self.canvSDomain.axes.imshow(new_p, cmap=plt.get_cmap('gray'))
        self.canvSDomain.draw()

    def setpixmap(self, image):
        data = Image.fromarray(image)
        data.save('filteredimage.jpg')

        self.canvSDomain.axes.cla()
        self.canvSDomain.axes.axis('off')
        self.canvSDomain.axes.imshow(data, cmap=plt.get_cmap('gray'))
        self.canvSDomain.draw()

    # FIX: saveImag was connected to multiple menu actions but never defined
    def saveImag(self, imageType):
        savePath = QFileDialog.getSaveFileName(self, "Save Image", "", "PNG (*.png);;JPEG (*.jpg)")[0]
        if not savePath:
            return

        if imageType == "Histogram":
            self.canvHistogram.figure.savefig(savePath)
        elif imageType == "EqHistogram":
            self.canvEqualized.figure.savefig(savePath)
        elif imageType == "RevImage":
            self.canvNewImage.figure.savefig(savePath)
        elif imageType == "FreqFilter":
            self.canvfilter.figure.savefig(savePath)
        elif imageType == "SpatialFilter":
            self.canvSDomain.figure.savefig(savePath)
        else:
            print(f"Unknown image type: {imageType}")

        self.logging(f"Saved image of type '{imageType}' to {savePath}")

    def exit(self):
        self.logging("Exit function was called")
        sys.exit()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main = mainApp()
    main.show()
    sys.exit(app.exec_())