import cv2 as cv
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import*
from PyQt5.uic import loadUi
from matplotlib.backends.backend_qt5agg import(NavigationToolbar2QT as NavigationToolbar)
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import sys


class Display(QMainWindow):
    def __init__(self):

        QMainWindow.__init__(self)
        loadUi("skripsi.ui", self)

        self.setWindowTitle("Skripsi")
        self.pushButton_mode1.clicked.connect(self.mode_1)
        self.pushButton_mode2.clicked.connect(self.mode_2)
        self.pushButton_openImage.clicked.connect(self.loadimage)

        self.addToolBar(NavigationToolbar(self.widgetHistogram.canvas.self))

    def mode_1(self):
        pass

    def mode_2(self):
        pass

    def loadimage(self):
        self.filename = QFileDialog.getOpenFileName(filter="Image (*.*)")[0]
        image = cv.imread(self.filename)
        self.display_histogram(image)
        ori_img = self.convert_cv_qt(image)
        mask_hot = self.convert_maskingHotspot(image)
        mask_panel = self.convert_maskingPanel(image)
        final_img = self.convert_final(image)
        max_area = self.maxArea(image)
        self.label_imageOri.setPixmap(ori_img)
        self.label_maskingHotspot.setPixmap(mask_hot)
        self.label_maskingPanel.setPixmap(mask_panel)
        self.label_finalImage.setPixmap(final_img)

    def display_histogram(self,image):
        self.widgetHistogram.canvas.axes1.clear()
        read_img = cv.imread(image, cv.IMREAD_GRAYSCALE)
        histr = cv.calcHist([read_img], [0], None, [256], [0, 256])
        self.widgetHistogram.canvas.axes1.plot(histr, color = 1 , linewidth=3.0)
        self.widgetHistogram.canvas.axes1.set_ylabel('Frequency', color = "white")
        self.widgetHistogram.canvas.axes1.set_xlabel("Intensity", color = 'white')
        self.widgetHistogram.canvas.axes1.set_title('Histogram')
        self.widgetHistogram.canvas.axes1.set_facecolor('xkcd:black')
        self.widgetHistogram.canvas.axes1.grid()
        self.widgetHistogram.canvas.draw()

    def convert_cv_qt(self, image):
        gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        h, w = gray_image.shape
        #bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(gray_image.data, w, h , QtGui.QImage.Format_Grayscale8)
        p = convert_to_Qt_format.scaled(336, 256, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def convert_maskingHotspot(self, image):
        gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        mean, std = cv.meanStdDev(gray_image)
        gray_image = cv.medianBlur(gray_image, 5)
        gray_image = cv.GaussianBlur(gray_image, (5,5), 2)
        thres1 = (mean + 2*std)

        _, th1 = cv.threshold(gray_image, thres1, 255, cv.THRESH_BINARY)
        kernal = np.ones((4,4), "uint8")
        hot = cv.morphologyEx(th1,cv.MORPH_OPEN,kernal)
        hot = cv.morphologyEx(hot,cv.MORPH_ELLIPSE,kernal)
        hot = cv.erode(hot,cv.getStructuringElement(cv.MORPH_ELLIPSE, (8,8)))
        hot = cv.dilate(hot,cv.getStructuringElement(cv.MORPH_RECT, (14,14)))
        hot = cv.dilate(hot,cv.getStructuringElement(cv.MORPH_RECT, (14,14)))
        hot = cv.erode(hot,cv.getStructuringElement(cv.MORPH_ERODE, (3,3)))
        
        h, w = th1.shape
        #bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(hot.data, w, h , QtGui.QImage.Format_Grayscale8)
        p = convert_to_Qt_format.scaled(336, 256, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def convert_maskingPanel(self, image):
        gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        mean, std = cv.meanStdDev(gray_image)
        gray_image = cv.medianBlur(gray_image, 5)
        gray_image = cv.GaussianBlur(gray_image, (5,5), 2)
        thres2 = (mean + 0.59*std)

        _, th4 = cv.threshold(gray_image, thres2, 255, cv.THRESH_BINARY)
        kernal = np.ones((4,4), "uint8")
        Solarpanel = cv.morphologyEx(th4,cv.MORPH_CLOSE,kernal)
        Erode1 = cv.erode(Solarpanel, cv.getStructuringElement(cv.MORPH_ELLIPSE, (12,12)))
        Dilate = cv.dilate(Erode1, cv.getStructuringElement(cv.MORPH_ELLIPSE, (14,14)))
        Erode2 = cv.erode(Dilate, cv.getStructuringElement(cv.MORPH_ERODE, (4,4)))
        Panel = cv.dilate(Erode2, cv.getStructuringElement(cv.MORPH_RECT, (7,7)))
                
        h, w = Panel.shape
        #bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(Panel.data, w, h , QtGui.QImage.Format_Grayscale8)
        p = convert_to_Qt_format.scaled(336, 256, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def convert_final(self, image):
        cntp, areap, Mp, cxp, cyp, pembatasp, kotakp = ([] for i in range(7))
        CXpanel, CYpanel, areaPanel, CX, CY, areaCell = ([] for b in range(6)) 
        cnt, area, M, cx, cy, pembatas, kotak = ([] for t in range(7))

        thres_cell = 346.0
        thres_panel = 14025.3
        luas_panel = luas_hotspot = 0

        gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        finalImg = cv.cvtColor(gray_image, cv.COLOR_GRAY2BGR)
        mean, std = cv.meanStdDev(gray_image)
        gray_image = cv.medianBlur(gray_image, 5)
        gray_image = cv.GaussianBlur(gray_image, (5,5), 2)
        thres1 = (mean + 2*std)
        thres2 = (mean + 0.59*std)

        _, th1 = cv.threshold(gray_image, thres1, 255, cv.THRESH_BINARY)
        _, th4 = cv.threshold(gray_image, thres2, 255, cv.THRESH_BINARY)
        kernal = np.ones((4,4), "uint8")
        hot = cv.morphologyEx(th1,cv.MORPH_ELLIPSE,kernal)
        hot = cv.dilate(hot,cv.getStructuringElement(cv.MORPH_ELLIPSE, (10,10)))
        hot = cv.dilate(hot,cv.getStructuringElement(cv.MORPH_RECT, (10,10)))
        hot = cv.erode(hot,cv.getStructuringElement(cv.MORPH_ERODE, (4,4)))

        contours, hierarchy = cv.findContours(hot, 1, 2)

        for n in range(len(contours)):
            cnt.append(contours[n])
            area.append(cv.contourArea(cnt[n]))
            if area[n] > 0.3*thres_cell:
                M.append(cv.moments(cnt[n]))
                cx.append(int(M[n]['m10']/M[n]['m00']))
                cy.append(int(M[n]['m01']/M[n]['m00']))
                pembatas.append(cv.minAreaRect(cnt[n]))
                kotak = cv.boxPoints(pembatas[n])
                kotak = np.int32(kotak)
                cv.drawContours(finalImg, [kotak], -1, (255, 0, 0), 1)
                luas_hotspot = luas_hotspot + area[n]
            areaCell = [x for x in area if x > 0.3*thres_cell]
        

        Solarpanel = cv.morphologyEx(th4,cv.MORPH_CLOSE,kernal)
        Erode1 = cv.erode(Solarpanel, cv.getStructuringElement(cv.MORPH_ELLIPSE, (7,7)))
        Dilate = cv.dilate(Erode1, cv.getStructuringElement(cv.MORPH_ELLIPSE, (15,15)))
        Erode2 = cv.erode(Dilate, cv.getStructuringElement(cv.MORPH_ERODE, (5,5)))
        Panel = cv.dilate(Erode2, cv.getStructuringElement(cv.MORPH_RECT, (3,3)))

        transformasiJarak = cv.distanceTransform(Panel, cv.DIST_L2, cv.DIST_MASK_5)
        ambang, latarDepan = cv.threshold(transformasiJarak, 0.6*transformasiJarak.max(), 255, cv.THRESH_BINARY)
        latarDepan = np.uint8(latarDepan)
        daerahTakBertuan = cv.subtract(Panel, latarDepan)
        
        jumlahPanel, penanda = cv.connectedComponents(latarDepan)
        #print("Jumlahg Panel: ", jumlahPanel - 1)

        penanda = penanda +1

        penanda[daerahTakBertuan == 255] = 0

        penanda = cv.watershed(finalImg, penanda)

        image[penanda == -1] = [255, 0, 0]

        tinggi, lebar, channel = image.shape[:3]
        cadar = np.zeros((tinggi, lebar, 1), np.uint8)
        bisa = np.zeros((tinggi, lebar, 1), np.uint8)
        bisa2 = cv.rectangle(bisa, (0,0), (lebar, tinggi), 255, -1)
        for indeks in range(1, jumlahPanel):
            cadar[penanda == indeks + 1] = 1
            ayo = bisa2 * cadar[:,:]

        #cv.imshow('ayo', ayo)

        contourspanel, hierarchy = cv.findContours(ayo, 1, 2)

        for n in range(len(contourspanel)):
            cntp.append(contourspanel[n])
            areap.append(cv.contourArea(cntp[n]))
            if areap[n] > 0.5*thres_panel:
                for m in range (len(areap)): 
                    if m > n :
                        break
                    Mp.append(cv.moments(cntp[n]))
                    cxp.append(int(Mp[m]['m10']/Mp[m]['m00']))
                    cyp.append(int(Mp[m]['m01']/Mp[m]['m00']))
                    pembatasp.append(cv.minAreaRect(cntp[n]))
                    kotakp = cv.boxPoints(pembatasp[m])
                    kotakp = np.int32(kotakp)
                    cv.drawContours(finalImg, [kotakp], -1, (0, 0, 255), 1)
                luas_panel = luas_panel + areap[n]
            [CXpanel.append(x) for x in cxp if x not in CXpanel]
            [CYpanel.append(x) for x in cyp if x not in CYpanel]
            areaPanel = [x for x in areap if x > 0.5*thres_panel]
        
        h, w = Panel.shape
        #bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(finalImg.data, w, h , QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(336, 256, Qt.KeepAspectRatio)
        defect = (luas_hotspot / luas_panel) * 100
        print("%Defect: ", defect)
        print("Luas Panel: ", areaPanel)
        print("Luas HotSpot: ", areaCell)
        return QPixmap.fromImage(p)

    def maxArea(self, image):
            cntp, areap, Mp, cxp, cyp, pembatasp, kotakp = ([] for i in range(7))
            CXpanel, CYpanel, areaPanel, CX, CY, areaCell = ([] for b in range(6)) 
            cnt, area, M, cx, cy, pembatas, kotak = ([] for t in range(7))

            thres_cell = 346.0
            thres_panel = 14025.3
            luas_panel = luas_hotspot = 0

            gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            finalImg = cv.cvtColor(gray_image, cv.COLOR_GRAY2BGR)
            clahe = cv.createCLAHE(clipLimit= 1.0)
            gray_image = clahe.apply(gray_image)
            mean, std = cv.meanStdDev(gray_image)
            gray_image = cv.medianBlur(gray_image, 5)
            gray_image = cv.GaussianBlur(gray_image, (5,5), 2)
            thres1 = (mean + 1.8*std)
            thres2 = (mean + 0.5*std)

            _, th1 = cv.threshold(gray_image, thres1, 255, cv.THRESH_BINARY)
            _, th4 = cv.threshold(gray_image, thres2, 255, cv.THRESH_BINARY)
            kernal = np.ones((4,4), "uint8")
            hot = cv.morphologyEx(th1,cv.MORPH_ELLIPSE,kernal)
            hot = cv.dilate(hot,cv.getStructuringElement(cv.MORPH_ELLIPSE, (10,10)))
            hot = cv.dilate(hot,cv.getStructuringElement(cv.MORPH_RECT, (10,10)))
            hot = cv.erode(hot,cv.getStructuringElement(cv.MORPH_ERODE, (4,4)))

            contours, hierarchy = cv.findContours(hot, 1, 2)

            #cv.imshow('HOT',hot)

            for n in range(len(contours)):
                cnt.append(contours[n])
                area.append(cv.contourArea(cnt[n]))
                if area[n] > 0.3*thres_cell:
                    M.append(cv.moments(cnt[n]))
                    cx.append(int(M[n]['m10']/M[n]['m00']))
                    cy.append(int(M[n]['m01']/M[n]['m00']))
                    pembatas.append(cv.minAreaRect(cnt[n]))
                    kotak = cv.boxPoints(pembatas[n])
                    kotak = np.int32(kotak)
                    cv.drawContours(finalImg, [kotak], -1, (0, 0, 255), 1)
                    luas_hotspot = luas_hotspot + area[n]
                areaCell = [x for x in area if x > 0.3* thres_cell]
            

            Solarpanel = cv.morphologyEx(th4,cv.MORPH_CLOSE,kernal)
            Erode1 = cv.erode(Solarpanel, cv.getStructuringElement(cv.MORPH_ELLIPSE, (10,10)))
            Dilate = cv.dilate(Erode1, cv.getStructuringElement(cv.MORPH_ELLIPSE, (15,15)))
            Erode2 = cv.erode(Dilate, cv.getStructuringElement(cv.MORPH_ERODE, (7,7)))
            Panel = cv.dilate(Erode2, cv.getStructuringElement(cv.MORPH_RECT, (3,3)))

            contourspanel, hierarchy = cv.findContours(Panel, 1, 2)

            for n in range(len(contourspanel)):
                cntp.append(contourspanel[n])
                areap.append(cv.contourArea(cntp[n]))
                if areap[n] > 0.3*thres_panel:
                    for m in range (len(areap)): 
                        if m > n :
                            break
                        Mp.append(cv.moments(cntp[n]))
                        cxp.append(int(Mp[m]['m10']/Mp[m]['m00']))
                        cyp.append(int(Mp[m]['m01']/Mp[m]['m00']))
                        pembatasp.append(cv.minAreaRect(cntp[n]))
                        kotakp = cv.boxPoints(pembatasp[m])
                        kotakp = np.int32(kotakp)
                        cv.drawContours(finalImg, [kotakp], -1, (255, 0, 0), 1)
                    luas_panel = luas_panel + areap[n]
                [CXpanel.append(x) for x in cxp if x not in CXpanel]
                [CYpanel.append(x) for x in cyp if x not in CYpanel]
                areaPanel = [x for x in areap if x > 0.3*thres_panel]

            defect = (luas_hotspot / luas_panel) * 100
            print("%Defect (contrast): ", defect)
            print("Luas Panel2 (contrast): ", areaPanel)
            print("Luas HotSpot2 (contrast): ", areaCell)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Display()
    ex.show()
    sys.exit(app.exec_())
        