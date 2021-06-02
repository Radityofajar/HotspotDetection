import cv2 as cv
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import*
from PyQt5.uic import loadUi
from matplotlib.backends.backend_qt5agg import(NavigationToolbar2QT as NavigationToolbar)
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import sys
import os
from scipy.spatial import distance as dist

from numpy.core.fromnumeric import resize


class Display(QMainWindow):
    def __init__(self):

        QMainWindow.__init__(self)
        loadUi("skripsi.ui", self)

        self.setWindowTitle("Skripsi")
        self.pushButton_mode1.clicked.connect(self.mode_1)
        self.pushButton_mode2.clicked.connect(self.mode_2)
        self.pushButton_mode3.clicked.connect(self.mode_3)
        self.pushButton_openImage.clicked.connect(self.loadimage)

        self.addToolBar(NavigationToolbar(self.widgetHistogram.canvas, self))

    def mode_1(self):
        pass

    def mode_2(self):
        self.cams = Window1()
        self.cams.show()
        self.close()
    
    def mode_3(self):
        self.cams = VideoWindow()
        self.cams.show()
        self.close()

    def loadimage(self):
        self.filename = QFileDialog.getOpenFileName(filter="Image (*.*)")[0]
        image = cv.imread(self.filename)
        image2 = cv.imread(self.filename)
        self.display_histogram(image)
        ori_img = self.convert_cv_qt(image)
        mask_hot = self.convert_maskingHotspot(image)
        mask_panel = self.convert_maskingPanel(image)
        final_img = self.convert_final(image)
        max_Area = self.maxArea(image2)
        min_Area = self.minArea(image2)
        self.label_imageOri.setPixmap(ori_img)
        self.label_maskingHotspot.setPixmap(mask_hot)
        self.label_maskingPanel.setPixmap(mask_panel)
        self.label_finalImage.setPixmap(final_img)

    def display_histogram(self,image):
        self.widgetHistogram.canvas.axes1.clear()
        read_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        histr = cv.calcHist([read_img], [0], None, [256], [0, 256])
        self.widgetHistogram.canvas.axes1.plot(histr, color = "yellow" , linewidth=1.0)
        self.widgetHistogram.canvas.axes1.set_ylabel('Frequency', color = "white")
        self.widgetHistogram.canvas.axes1.set_xlabel("Intensity", color = 'white')
        self.widgetHistogram.canvas.axes1.set_title('Histogram', color = 'white')
        self.widgetHistogram.canvas.axes1.tick_params(axis='x', colors='white')
        self.widgetHistogram.canvas.axes1.tick_params(axis='y', colors='white')
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
        thres1 = (mean + 1.75*std)

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
        thres1 = (mean + 1.75*std)
        thres2 = (mean + 0.59*std)

        _, th1 = cv.threshold(gray_image, thres1, 255, cv.THRESH_BINARY)
        _, th4 = cv.threshold(gray_image, thres2, 255, cv.THRESH_BINARY)
        kernal = np.ones((4,4), "uint8")
        hot = cv.morphologyEx(th1,cv.MORPH_ELLIPSE,kernal)
        hot = cv.dilate(hot,cv.getStructuringElement(cv.MORPH_ELLIPSE, (6,6)))
        hot = cv.dilate(hot,cv.getStructuringElement(cv.MORPH_RECT, (7,7)))
        hot = cv.erode(hot,cv.getStructuringElement(cv.MORPH_ERODE, (4,4)))

        contours, hierarchy = cv.findContours(hot, 1, 2)

        for n in range(len(contours)):
            cnt.append(contours[n])
            area.append(cv.contourArea(cnt[n]))
            if area[n] > 0.5*thres_cell:
                M.append(cv.moments(cnt[n]))
                cx.append(int(M[n]['m10']/M[n]['m00']))
                cy.append(int(M[n]['m01']/M[n]['m00']))
                pembatas.append(cv.minAreaRect(cnt[n]))
                kotak = cv.boxPoints(pembatas[n])
                kotak = np.int32(kotak)
                cv.drawContours(finalImg, [kotak], -1, (255, 0, 0), 1)
                luas_hotspot = luas_hotspot + area[n]
            [CX.append(x) for x in cx if x not in CX]
            [CY.append(x) for x in cy if x not in CY]
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
                    Mp.append(cv.moments(cntp[n]))
                    pembatasp.append(cv.minAreaRect(cntp[n]))
                    x, y, w, h = cv.boundingRect(cntp[n])
                    kotakp = cv.boxPoints(pembatasp[m])
                    kotakp = np.int0(kotakp)
                    cxp.append(int(Mp[m]['m10']/Mp[m]['m00']))
                    cyp.append(int(Mp[m]['m01']/Mp[m]['m00']))
                    #cv.circle(finalImg, (np.int(cxp), np.int(cyp)), 2, (0,255,255), -1)
                    cv.drawContours(finalImg, [kotakp], -1, (0, 0, 255), 1)

                    for (x,y) in kotakp:
                        #cv.circle(finalImg, (int(x), int(y)), -1, (0, 0, 255), 1)
                        (tl,tr,br,bl) = kotakp
                        (tltrX, tltrY) = ((tl[0] + tr[0]) *0.5, (tl[1] + tr[1])*0.5)
                        (blbrX, blbrY) = ((bl[0] + br[0]) *0.5, (bl[1] + br[1])*0.5)
                        (tlblX, tlblY) = ((tl[0] + bl[0]) *0.5, (tl[1] + bl[1])*0.5)
                        (trbrX, trbrY) = ((tr[0] + br[0]) *0.5, (tr[1] + br[1])*0.5)

                        dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
                        dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

                        dimA = dA * 6
                        dimB = dB * 6

                        cv.putText(finalImg, "{:.1f}mm".format(dimA), (int(tltrX+30), int(tltrY-80)),
                            cv.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)
                        #cv.putText(finalImg, "lebar: ", (int(tltrX), int(tltrY-100)),
                        #    cv.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)
                        cv.putText(finalImg, "{:.1f}mm".format(dimB), (int(tltrX-40), int(tltrY)),
                            cv.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)
                        #cv.putText(finalImg, "panjang: ", (int(tltrX-100), int(tltrY)),
                        #    cv.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)

                luas_panel = luas_panel + areap[n]
            [CXpanel.append(x) for x in cxp if x not in CXpanel]
            [CYpanel.append(x) for x in cyp if x not in CYpanel]
            areaPanel = [x for x in areap if x > 0.5*thres_panel]

        finalImg = cv.resize(finalImg, (504,384))
        defect = (luas_hotspot / luas_panel) * 100
        print("%Defect: ", defect)
        print("Luas Panel: ", areaPanel)
        print("Luas HotSpot: ", areaCell)
        print("CX Panel: ", CXpanel)
        print("CY Panel: ", CYpanel)
        print("CX: ", CX)
        print("CY: ", CY)
        
        cv.putText(finalImg, 'Mean: ', (5,10), cv.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,255), 1, cv.LINE_AA)
        cv.putText(finalImg, 'Std: ', (5,30), cv.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,255), 1, cv.LINE_AA)
        cv.putText(finalImg, 'Thres1: ', (5,50), cv.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,255), 1, cv.LINE_AA)
        cv.putText(finalImg, 'Thres2: ', (5,70), cv.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,255), 1, cv.LINE_AA)
        cv.putText(finalImg, 'Panel: ', (5,90), cv.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,255), 1, cv.LINE_AA)
        cv.putText(finalImg, 'Hotspot: ', (5,110), cv.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,255), 1, cv.LINE_AA)
        cv.putText(finalImg, '%Defect: ', (5,130), cv.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,255), 1, cv.LINE_AA)
        cv.putText(finalImg, str(mean) , (50,10), cv.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,255), 1, cv.LINE_AA)
        cv.putText(finalImg, str(std) , (50,30), cv.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,255), 1, cv.LINE_AA)
        cv.putText(finalImg, str(thres1) , (50,50), cv.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,255), 1, cv.LINE_AA)
        cv.putText(finalImg, str(thres2) , (50,70), cv.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,255), 1, cv.LINE_AA)
        cv.putText(finalImg, str(areaPanel) , (50,90), cv.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,255), 1, cv.LINE_AA)
        cv.putText(finalImg, str(areaCell) , (50,110), cv.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,255), 1, cv.LINE_AA)
        cv.putText(finalImg, str(defect) , (50,130), cv.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,255), 1, cv.LINE_AA)

        h, w = Panel.shape
        #bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(finalImg.data, 504, 384 , QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(504, 384, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def maxArea(self, image2):
        cntp, areap, Mp, cxp, cyp, pembatasp, kotakp = ([] for i in range(7))
        CXpanel, CYpanel, areaPanel, CX, CY, areaCell = ([] for b in range(6)) 
        cnt, area, M, cx, cy, pembatas, kotak = ([] for t in range(7))

        thres_cell = 346.0
        thres_panel = 14025.3
        luas_panel = luas_hotspot = 0

        gray_image = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)
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
        hot = cv.dilate(hot,cv.getStructuringElement(cv.MORPH_ELLIPSE, (6,6)))
        hot = cv.dilate(hot,cv.getStructuringElement(cv.MORPH_RECT, (7,7)))
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

        image2[penanda == -1] = [255, 0, 0]

        tinggi, lebar, channel = image2.shape[:3]
        cadar = np.zeros((tinggi, lebar, 1), np.uint8)
        bisa = np.zeros((tinggi, lebar, 1), np.uint8)
        bisa2 = cv.rectangle(bisa, (0,0), (lebar, tinggi), 255, -1)
        for indeks in range(1, jumlahPanel):
            cadar[penanda == indeks + 1] = 1
            ayo = bisa2 * cadar[:,:]

        contourspanel, hierarchy = cv.findContours(ayo, 1, 2)

        for n in range(len(contourspanel)):
            cntp.append(contourspanel[n])
            areap.append(cv.contourArea(cntp[n]))
            if areap[n] > 0.5*thres_panel:
                for m in range (len(areap)): 
                    Mp.append(cv.moments(cntp[n]))
                    pembatasp.append(cv.minAreaRect(cntp[n]))
                    x, y, w, h = cv.boundingRect(cntp[n])
                    kotakp = cv.boxPoints(pembatasp[m])
                    kotakp = np.int0(kotakp)
                    cxp.append(int(Mp[m]['m10']/Mp[m]['m00']))
                    cyp.append(int(Mp[m]['m01']/Mp[m]['m00']))
                    #cv.circle(finalImg, (np.int(cxp), np.int(cyp)), 2, (0,255,255), -1)
                    cv.drawContours(finalImg, [kotakp], -1, (0, 0, 255), 1)

        finalImg = cv.cvtColor(finalImg, cv.COLOR_BGR2RGB)
        cv.imwrite("apapun.jpg", finalImg)

    def minArea(self, image2):
        cntp, areap, Mp, cxp, cyp, pembatasp, kotakp = ([] for i in range(7))
        CXpanel, CYpanel, areaPanel, CX, CY, areaCell = ([] for b in range(6)) 
        cnt, area, M, cx, cy, pembatas, kotak = ([] for t in range(7))

        thres_cell = 340.0
        thres_panel = 14025.3
        luas_panel = luas_hotspot = 0

        gray_image = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)
        finalImg = cv.cvtColor(gray_image, cv.COLOR_GRAY2BGR)
        mean, std = cv.meanStdDev(gray_image)
        thres1 = (mean + 1.75*std)
        thres2 = (mean + 0.59*std)

        _, th1 = cv.threshold(gray_image, thres1, 255, cv.THRESH_BINARY)
        _, th4 = cv.threshold(gray_image, thres2, 255, cv.THRESH_BINARY)

        contours, hierarchy = cv.findContours(th1, 1, 2)

        for n in range(len(contours)):
            cnt.append(contours[n])
            area.append(cv.contourArea(cnt[n]))
            if area[n] > 0.1*thres_cell:
                for m in range(len(area)):
                    M.append(cv.moments(cnt[n]))
                    cx.append(int(M[m]['m10']/M[m]['m00']))
                    cy.append(int(M[m]['m01']/M[m]['m00']))
                    pembatas.append(cv.minAreaRect(cnt[n]))
                    kotak = cv.boxPoints(pembatas[m])
                    kotak = np.int0(kotak)
                    cv.drawContours(finalImg, [kotak], -1, (255, 0, 0), 1)
                [CX.append(x) for x in cx if x not in CX]
                [CY.append(x) for x in cy if x not in CY]

        contourspanel, hierarchy = cv.findContours(th4, 1, 2)

        for n in range(len(contourspanel)):
            cntp.append(contourspanel[n])
            areap.append(cv.contourArea(cntp[n]))
            if areap[n] > 0.5*thres_panel:
                for m in range (len(areap)): 
                    Mp.append(cv.moments(cntp[n]))
                    cxp.append(int(Mp[m]['m10']/Mp[m]['m00']))
                    cyp.append(int(Mp[m]['m01']/Mp[m]['m00']))
                    pembatasp.append(cv.minAreaRect(cntp[n]))
                    x, y, w, h = cv.boundingRect(cntp[n])
                    kotakp = cv.boxPoints(pembatasp[m])
                    kotakp = np.int0(kotakp)
                    cv.drawContours(finalImg, [kotakp], -1, (0, 0, 255), 1)
                [CXpanel.append(x) for x in cxp if x not in CXpanel]
                [CYpanel.append(x) for x in cyp if x not in CYpanel]
        finalImg = cv.cvtColor(finalImg, cv.COLOR_BGR2RGB)
        cv.imwrite("apapun2.jpg", finalImg)
        print("CX1 Panel: ", CXpanel)
        print("CY1 Panel: ", CYpanel)
        print("CX1: ", CX)
        print("CY1: ", CY)

class Window1(QWidget):
    def __init__(self):
        super().__init__()
        self.title = "Skrips1"
        self.top = 0
        self.left = 0
        self.width = 1360
        self.height = 768
        self.InitUI()
        self.show()

    def InitUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.top, self.left, self.width, self.height)

        pushButton_mode1 = QPushButton('Mode1', self)
        pushButton_mode1.setGeometry(10, 20, 111, 81)
        pushButton_mode1.clicked.connect(self.mode_1)

        pushButton_mode2 = QPushButton('Mode2', self)
        pushButton_mode2.setGeometry(10, 110, 111, 81)
        pushButton_mode2.clicked.connect(self.mode_2)

        pushButton_mode3 = QPushButton('Mode3', self)
        pushButton_mode3.setGeometry(10, 200, 111, 81)
        pushButton_mode3.clicked.connect(self.mode_3)

        pushButton_openImage = QPushButton('Open', self)
        pushButton_openImage.setGeometry(10, 410, 111, 81)
        pushButton_openImage.clicked.connect(self.loadimage)

        pushButton_save = QPushButton('Save', self)
        pushButton_save.setGeometry(10, 500, 111, 81)
        pushButton_save.clicked.connect(self.loadimage)

        self.label_imageOri = QLabel('Image Ori', self)
        self.label_imageOri.setGeometry(150, 20, 336, 256)
        self.label_imageOri.setFrameShape(QFrame.Panel)
        self.label_imageOri.setObjectName('label_imageOri')

        self.label_gray = QLabel('Gray', self)
        self.label_gray.setGeometry(500, 20, 336, 256)
        self.label_gray.setFrameShape(QFrame.Panel)
        self.label_gray.setObjectName('label_gray')

        self.label_panel = QLabel('Panel', self)
        self.label_panel.setGeometry(150, 300, 336, 256)
        self.label_panel.setFrameShape(QFrame.Panel)
        self.label_panel.setObjectName('label_panel')

        self.label_testing = QLabel('Testing', self)
        self.label_testing.setGeometry(500, 300, 336, 256)
        self.label_testing.setFrameShape(QFrame.Panel)
        self.label_testing.setObjectName('label_testing') 

        self.label_finalImage = QLabel('Final Image', self)
        self.label_finalImage.setGeometry(850, 20, 504, 384)
        self.label_finalImage.setFrameShape(QFrame.Panel)
        self.label_finalImage.setObjectName('label_finalImage') 

        self.label = QLabel('Logo', self)
        self.label.setGeometry(860,420,500,175)
        self.label.setObjectName('label') 
        pixmap = QPixmap('logoUI.png')
        self.label.setPixmap(pixmap)

        line = QFrame(self)
        line.setGeometry(122, 0, 31, 760)
        line.setLineWidth(5)
        line.setFrameShape(QFrame.VLine)
        line.setFrameShadow(QFrame.Plain)

    def mode_1(self):
        self.cams = Display() 
        self.cams.show()
        self.close()

    def mode_2(self):
        pass

    def mode_3(self):
        self.cams = VideoWindow() 
        self.cams.show()
        self.close()

    def loadimage(self):
        self.filename = QFileDialog.getOpenFileName(filter="Image (*.*)")[0]
        image = cv.imread(self.filename)
        image = cv.resize(image, (336,256), interpolation=cv.INTER_AREA)
        image2 = cv.resize(image, (800,450), interpolation=cv.INTER_AREA)
        ori_img = self.convert_ori(image)
        gray_img = self.convert_gray(image)
        panel_img = self.convert_panel(image)
        testing_img = self.convert_testingimage(image)
        final_img = self.convert_final(image2)
        self.label_imageOri.setPixmap(ori_img)
        self.label_gray.setPixmap(gray_img)
        self.label_panel.setPixmap(panel_img)
        self.label_testing.setPixmap(testing_img)
        self.label_finalImage.setPixmap(final_img)

    def convert_ori(self, image):
        rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, 336, 256 , QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(336, 256, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def convert_gray(self, image):
        kernalClose = np.ones((5,5),"uint8")
        kernalOpen = np.ones((7,7),"uint8")
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
        kernelC = cv.getStructuringElement(cv.MORPH_ELLIPSE,(15,15))

        blur = cv.GaussianBlur(image, (7,7), 3)
        b,g,r = cv.split(blur)
        ret, thresh = cv.threshold(b, 100, 255, cv.THRESH_BINARY)
        ret1, thresh1 = cv.threshold(r, 125, 255, cv.THRESH_BINARY)
        bitwise_xor = cv.bitwise_xor(thresh1, thresh)
        bitwise_xor1 = cv.bitwise_and(thresh, bitwise_xor)
        opening = cv.morphologyEx(bitwise_xor1, cv.MORPH_OPEN, kernel)
        if len(opening.shape) != 2:
            gray = cv.cvtColor(opening, cv.COLOR_BGR2GRAY)
        else:
            gray = opening
        convert_to_Qt_format = QtGui.QImage(gray.data, 336, 256 , QtGui.QImage.Format_Grayscale8)
        p = convert_to_Qt_format.scaled(336, 256, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def convert_panel(self, image):
        kernalClose = np.ones((5,5),"uint8")
        kernalOpen = np.ones((7,7),"uint8")
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
        kernelC = cv.getStructuringElement(cv.MORPH_ELLIPSE,(15,15))

        blur = cv.GaussianBlur(image, (7,7), 3)
        b,g,r = cv.split(blur)
        ret, thresh = cv.threshold(b, 100, 255, cv.THRESH_BINARY)
        ret1, thresh1 = cv.threshold(r, 125, 255, cv.THRESH_BINARY)
        bitwise_xor = cv.bitwise_xor(thresh1, thresh)
        bitwise_xor1 = cv.bitwise_and(thresh, bitwise_xor)
        opening = cv.morphologyEx(bitwise_xor1, cv.MORPH_OPEN, kernel)
        if len(opening.shape) != 2:
            gray = cv.cvtColor(opening, cv.COLOR_BGR2GRAY)
        else:
            gray = opening

        gray = cv.bitwise_not(gray)
        bw = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, \
            cv.THRESH_BINARY, 15, -2)
        
        horizontal = np.copy(bw)
        vertical = np.copy(bw)

        cols = horizontal.shape[1]
        horizontal_size = cols // 30
        horizontalStructure = cv.getStructuringElement(cv.MORPH_RECT, (horizontal_size, 1))
        horizontal = cv.erode(horizontal, horizontalStructure)
        horizontal = cv.dilate(horizontal, horizontalStructure)

        rows = vertical.shape[0]
        verticalsize = rows // 30
        verticalStructure = cv.getStructuringElement(cv.MORPH_RECT, (1, verticalsize))
        vertical = cv.erode(vertical, verticalStructure)
        vertical = cv.dilate(vertical, verticalStructure)
        vertical = cv.bitwise_not(vertical)
        edges = cv.adaptiveThreshold(vertical, 255, cv.ADAPTIVE_THRESH_MEAN_C, \
            cv.THRESH_BINARY, 3, -2)
        kernel = np.ones((2, 2), np.uint8)
        edges = cv.dilate(edges, kernel)
        smooth = np.copy(vertical)
        smooth = cv.blur(smooth, (2, 2))
        (rows, cols) = np.where(edges != 0)
        vertical[rows, cols] = smooth[rows, cols]

        edges2 = cv.adaptiveThreshold(horizontal, 255, cv.ADAPTIVE_THRESH_MEAN_C, \
            cv.THRESH_BINARY, 3, -2)
        edges = cv.dilate(edges, kernel)
        smooth = np.copy(horizontal)
        smooth = cv.blur(smooth, (2, 2))
        (rows, cols) = np.where(edges != 0)
        horizontal[rows, cols] = smooth[rows, cols]

        notVertical = cv.bitwise_not(horizontal)
        panel = cv.bitwise_and(notVertical, vertical)
        panel2 = cv.bitwise_not(panel)
         
        convert_to_Qt_format = QtGui.QImage(panel2.data, 336, 256 , QtGui.QImage.Format_Grayscale8)
        p = convert_to_Qt_format.scaled(336, 256, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def convert_testingimage(self, image):
        kernalClose = np.ones((5,5),"uint8")
        kernalOpen = np.ones((7,7),"uint8")
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
        kernelC = cv.getStructuringElement(cv.MORPH_ELLIPSE,(15,15))

        blur = cv.GaussianBlur(image, (7,7), 3)
        b,g,r = cv.split(blur)
        ret, thresh = cv.threshold(b, 100, 255, cv.THRESH_BINARY)
        ret1, thresh1 = cv.threshold(r, 125, 255, cv.THRESH_BINARY)
        bitwise_xor = cv.bitwise_xor(thresh1, thresh)
        bitwise_xor1 = cv.bitwise_and(thresh, bitwise_xor)
        opening = cv.morphologyEx(bitwise_xor1, cv.MORPH_OPEN, kernel)
        if len(opening.shape) != 2:
            gray = cv.cvtColor(opening, cv.COLOR_BGR2GRAY)
        else:
            gray = opening

        gray = cv.bitwise_not(gray)
        bw = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, \
            cv.THRESH_BINARY, 15, -2)
        
        horizontal = np.copy(bw)
        vertical = np.copy(bw)

        cols = horizontal.shape[1]
        horizontal_size = cols // 30
        horizontalStructure = cv.getStructuringElement(cv.MORPH_RECT, (horizontal_size, 1))
        horizontal = cv.erode(horizontal, horizontalStructure)
        horizontal = cv.dilate(horizontal, horizontalStructure)

        rows = vertical.shape[0]
        verticalsize = rows // 30
        verticalStructure = cv.getStructuringElement(cv.MORPH_RECT, (1, verticalsize))
        vertical = cv.erode(vertical, verticalStructure)
        vertical = cv.dilate(vertical, verticalStructure)
        vertical = cv.bitwise_not(vertical)
        edges = cv.adaptiveThreshold(vertical, 255, cv.ADAPTIVE_THRESH_MEAN_C, \
            cv.THRESH_BINARY, 3, -2)
        kernel = np.ones((2, 2), np.uint8)
        edges = cv.dilate(edges, kernel)
        smooth = np.copy(vertical)
        smooth = cv.blur(smooth, (2, 2))
        (rows, cols) = np.where(edges != 0)
        vertical[rows, cols] = smooth[rows, cols]

        edges2 = cv.adaptiveThreshold(horizontal, 255, cv.ADAPTIVE_THRESH_MEAN_C, \
            cv.THRESH_BINARY, 3, -2)
        edges = cv.dilate(edges, kernel)
        smooth = np.copy(horizontal)
        smooth = cv.blur(smooth, (2, 2))
        (rows, cols) = np.where(edges != 0)
        horizontal[rows, cols] = smooth[rows, cols]

        notVertical = cv.bitwise_not(horizontal)
        panel = cv.bitwise_and(notVertical, vertical)
        panel2 = cv.bitwise_not(panel)
        lines = cv.HoughLinesP(gray,1,np.pi/180,100,minLineLength=100,maxLineGap=100)
        for line in lines:
            x1,y1,x2,y2 = line[0]
            cv.line(blur,(x1,y1),(x2,y2),(0,0,255),2)
        gray2 = cv.cvtColor(blur, cv.COLOR_BGR2GRAY)
        ret2, thresh2 = cv.threshold(gray2,  80, 255, cv.THRESH_BINARY)
        and2 = cv.bitwise_and(panel2, thresh2)

        convert_to_Qt_format = QtGui.QImage(and2.data, 336, 256 , QtGui.QImage.Format_Grayscale8)
        p = convert_to_Qt_format.scaled(336, 256, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def convert_final(self, image2):
        kernalClose = np.ones((5,5),"uint8")
        kernalOpen = np.ones((7,7),"uint8")
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
        kernelC = cv.getStructuringElement(cv.MORPH_ELLIPSE,(15,15))

        blur = cv.GaussianBlur(image2, (7,7), 3)
        b,g,r = cv.split(blur)
        ret, thresh = cv.threshold(b, 100, 255, cv.THRESH_BINARY)
        ret1, thresh1 = cv.threshold(r, 125, 255, cv.THRESH_BINARY)
        bitwise_xor = cv.bitwise_xor(thresh1, thresh)
        bitwise_xor1 = cv.bitwise_and(thresh, bitwise_xor)
        opening = cv.morphologyEx(bitwise_xor1, cv.MORPH_OPEN, kernel)
        if len(opening.shape) != 2:
            gray = cv.cvtColor(opening, cv.COLOR_BGR2GRAY)
        else:
            gray = opening

        gray = cv.bitwise_not(gray)
        bw = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, \
            cv.THRESH_BINARY, 15, -2)
        
        horizontal = np.copy(bw)
        vertical = np.copy(bw)

        cols = horizontal.shape[1]
        horizontal_size = cols // 30
        horizontalStructure = cv.getStructuringElement(cv.MORPH_RECT, (horizontal_size, 1))
        horizontal = cv.erode(horizontal, horizontalStructure)
        horizontal = cv.dilate(horizontal, horizontalStructure)

        rows = vertical.shape[0]
        verticalsize = rows // 30
        verticalStructure = cv.getStructuringElement(cv.MORPH_RECT, (1, verticalsize))
        vertical = cv.erode(vertical, verticalStructure)
        vertical = cv.dilate(vertical, verticalStructure)

        vertical = cv.bitwise_not(vertical)
        edges = cv.adaptiveThreshold(vertical, 255, cv.ADAPTIVE_THRESH_MEAN_C, \
            cv.THRESH_BINARY, 3, -2)
        kernel = np.ones((2, 2), np.uint8)
        edges = cv.dilate(edges, kernel)
        smooth = np.copy(vertical)
        smooth = cv.blur(smooth, (2, 2))
        (rows, cols) = np.where(edges != 0)
        vertical[rows, cols] = smooth[rows, cols]

        edges2 = cv.adaptiveThreshold(horizontal, 255, cv.ADAPTIVE_THRESH_MEAN_C, \
            cv.THRESH_BINARY, 3, -2)
        kernel = np.ones((2, 2), np.uint8)
        edges = cv.dilate(edges, kernel)
        smooth = np.copy(horizontal)
        smooth = cv.blur(smooth, (2, 2))
        (rows, cols) = np.where(edges != 0)
        horizontal[rows, cols] = smooth[rows, cols]

        notVertical = cv.bitwise_not(horizontal)
        panel = cv.bitwise_and(notVertical, vertical)
        panel2 = cv.bitwise_not(panel)
        lines = cv.HoughLinesP(gray,1,np.pi/180,100,minLineLength=100,maxLineGap=100)
        for line in lines:
            x1,y1,x2,y2 = line[0]
            cv.line(blur,(x1,y1),(x2,y2),(0,0,255),2)
        gray2 = cv.cvtColor(blur, cv.COLOR_BGR2GRAY)
        ret2, thresh2 = cv.threshold(gray2,  80, 255, cv.THRESH_BINARY)
        and2 = cv.bitwise_and(panel2, thresh2)
        open_ = cv.morphologyEx(and2, cv.MORPH_OPEN,kernalOpen) 
        dilation = cv.dilate(open_,kernalClose,iterations = 2 )
        erosion = cv.erode(dilation,kernalClose,iterations = 1)
        contours2, hierarchy = cv.findContours(erosion, 1, 2)
        result = cv.drawContours(image2, contours2 , 0, (0,0,255), 2)
        result = cv.cvtColor(result, cv.COLOR_BGR2RGB)
        convert_to_Qt_format = QtGui.QImage(result.data, 800, 450 , QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(504, 384, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

class VideoWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.title = "Skrips1"
        self.top = 0
        self.left = 0
        self.width = 1360
        self.height = 768
        self.InitUI()
        self.show()

    def InitUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.top, self.left, self.width, self.height)

        pushButton_mode1 = QPushButton('Mode1', self)
        pushButton_mode1.setGeometry(10, 20, 111, 81)
        pushButton_mode1.clicked.connect(self.mode_1)

        pushButton_mode2 = QPushButton('Mode2', self)
        pushButton_mode2.setGeometry(10, 110, 111, 81)
        pushButton_mode2.clicked.connect(self.mode_2)

        pushButton_mode3 = QPushButton('Mode3', self)
        pushButton_mode3.setGeometry(10, 200, 111, 81)
        pushButton_mode3.clicked.connect(self.mode_3)

        pushButton_openImage = QPushButton('Open', self)
        pushButton_openImage.setGeometry(10, 410, 111, 81)
        pushButton_openImage.clicked.connect(self.loadimage)

        pushButton_save = QPushButton('Save', self)
        pushButton_save.setGeometry(10, 500, 111, 81)
        pushButton_save.clicked.connect(self.loadimage)

        self.label_imageOri = QLabel('Image Ori', self)
        self.label_imageOri.setGeometry(330, 128, 336, 256)
        self.label_imageOri.setFrameShape(QFrame.Panel)
        self.label_imageOri.setObjectName('label_imageOri')

        self.label_finalImage = QLabel('Final Image', self)
        self.label_finalImage.setGeometry(694, 128, 336, 256)
        self.label_finalImage.setFrameShape(QFrame.Panel)
        self.label_finalImage.setObjectName('label_finalImage') 

        self.label = QLabel('Logo', self)
        self.label.setGeometry(860,420,500,175)
        self.label.setObjectName('label') 
        pixmap = QPixmap('logoUI.png')
        self.label.setPixmap(pixmap)

        line = QFrame(self)
        line.setGeometry(122, 0, 31, 760)
        line.setLineWidth(5)
        line.setFrameShape(QFrame.VLine)
        line.setFrameShadow(QFrame.Plain)

    def mode_1(self):
        self.cams = Display() 
        self.cams.show()
        self.close()

    def mode_2(self):
        self.cams = Window1() 
        self.cams.show()
        self.close()
    
    def mode_3(self):
        pass

    def loadimage(self):
        #self.filename = QFileDialog.getOpenFileName(filter="Video (*.*)")[0]
        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread2 = VideoThread2()
        self.thread2.change_pixmap_signal2.connect(self.update_image2)
        self.thread.start()
        self.thread2.start()

    #def closeEvent(self, event):
    #    self.thread.stop()
    #    event.accept()

    @pyqtSlot(np.ndarray)
    def update_image(self,finalImg):
        final_img = self.convert_final(finalImg)
        self.label_finalImage.setPixmap(final_img)

    def update_image2(self,cv_img):
        qt_img = self.convert_cv_qt(cv_img)
        self.label_imageOri.setPixmap(qt_img)
    
    def convert_cv_qt(self, cv_img):
        rgb_image = cv.cvtColor(cv_img, cv.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        #bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(336, 268, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)
    
    def convert_final(self, finalImg):
        h, w, ch = finalImg.shape
        #bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(finalImg.data, w, h, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(336, 268, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

class VideoThread2(QThread):
    change_pixmap_signal2 = pyqtSignal(np.ndarray)

    def run(self):
        # capture from web cam
        cap = cv.VideoCapture('/home/radityo/Desktop/HotspotDetection-main/4May4M21.mp4')
        while True:
            ret, cv_img = cap.read()
            if ret:
                self.change_pixmap_signal2.emit(cv_img)
        

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._run_flag = True

    def run(self):
        cap = cv.VideoCapture('/home/radityo/Desktop/HotspotDetection-main/4May8.mp4')
        while self._run_flag:
            ret, cv_img = cap.read()
            gray_image = cv.cvtColor(cv_img, cv.COLOR_BGR2GRAY)
            cntp, areap, Mp, cxp, cyp, pembatasp, kotakp = ([] for i in range(7))
            cnt, area, M, cx, cy, pembatas, kotak = ([] for t in range(7))

            thres_cell = 346.0
            thres_panel = 14025.3
            finalImg = cv.cvtColor(gray_image, cv.COLOR_GRAY2BGR)
            mean, std = cv.meanStdDev(gray_image)
            gray_image = cv.medianBlur(gray_image, 5)
            gray_image = cv.GaussianBlur(gray_image, (5,5), 2)
            thres1 = (mean + 1.75*std)
            thres2 = (mean + 0.59*std)

            _, th1 = cv.threshold(gray_image, thres1, 255, cv.THRESH_BINARY)
            _, th4 = cv.threshold(gray_image, thres2, 255, cv.THRESH_BINARY)
            kernal = np.ones((4,4), "uint8")
            hot = cv.morphologyEx(th1,cv.MORPH_ELLIPSE,kernal)
            hot = cv.dilate(hot,cv.getStructuringElement(cv.MORPH_ELLIPSE, (6,6)))
            hot = cv.dilate(hot,cv.getStructuringElement(cv.MORPH_RECT, (7,7)))
            hot = cv.erode(hot,cv.getStructuringElement(cv.MORPH_ERODE, (4,4)))

            contours, hierarchy = cv.findContours(hot, 1, 2)

            for n in range(len(contours)):
                cnt.append(contours[n])
                area.append(cv.contourArea(cnt[n]))
                if area[n] > 0.5*thres_cell:
                    M.append(cv.moments(cnt[n]))
                    cx.append(int(M[n]['m10']/M[n]['m00']))
                    cy.append(int(M[n]['m01']/M[n]['m00']))
                    pembatas.append(cv.minAreaRect(cnt[n]))
                    kotak = cv.boxPoints(pembatas[n])
                    kotak = np.int32(kotak)
                    cv.drawContours(finalImg, [kotak], -1, (255, 0, 0), 1)

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

            penanda = penanda +1

            penanda[daerahTakBertuan == 255] = 0

            penanda = cv.watershed(finalImg, penanda)

            cv_img[penanda == -1] = [255, 0, 0]

            tinggi, lebar, channel = cv_img.shape[:3]
            cadar = np.zeros((tinggi, lebar, 1), np.uint8)
            bisa = np.zeros((tinggi, lebar, 1), np.uint8)
            bisa2 = cv.rectangle(bisa, (0,0), (lebar, tinggi), 255, -1)
            for indeks in range(1, jumlahPanel):
                cadar[penanda == indeks + 1] = 1
                ayo = bisa2 * cadar[:,:]

            contourspanel, hierarchy = cv.findContours(ayo, 1, 2)

            for n in range(len(contourspanel)):
                cntp.append(contourspanel[n])
                areap.append(cv.contourArea(cntp[n]))
                if areap[n] > 0.5*thres_panel:
                    for m in range (len(areap)): 
                        Mp.append(cv.moments(cntp[n]))
                        pembatasp.append(cv.minAreaRect(cntp[n]))
                        x, y, w, h = cv.boundingRect(cntp[n])
                        kotakp = cv.boxPoints(pembatasp[m])
                        kotakp = np.int0(kotakp)
                        cxp.append(int(Mp[m]['m10']/Mp[m]['m00']))
                        cyp.append(int(Mp[m]['m01']/Mp[m]['m00']))
                        cv.drawContours(finalImg, [kotakp], -1, (0, 0, 255), 1)

                        for (x,y) in kotakp:
                            (tl,tr,br,bl) = kotakp
                            (tltrX, tltrY) = ((tl[0] + tr[0]) *0.5, (tl[1] + tr[1])*0.5)
                            (blbrX, blbrY) = ((bl[0] + br[0]) *0.5, (bl[1] + br[1])*0.5)
                            (tlblX, tlblY) = ((tl[0] + bl[0]) *0.5, (tl[1] + bl[1])*0.5)
                            (trbrX, trbrY) = ((tr[0] + br[0]) *0.5, (tr[1] + br[1])*0.5)

                            dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
                            dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

                            dimA = dA * 6
                            dimB = dB * 6

                            cv.putText(finalImg, "{:.1f}mm".format(dimA), (int(tltrX+30), int(tltrY-80)),
                                cv.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)
                            cv.putText(finalImg, "{:.1f}mm".format(dimB), (int(tltrX-40), int(tltrY)),
                                cv.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)
            if ret:
                self.change_pixmap_signal.emit(finalImg)
        cap.release()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Display()
    ex.showFullScreen()
    sys.exit(app.exec_())