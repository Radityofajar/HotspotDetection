import cv2 as cv
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import numpy as np
import sys

class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.title = "Skripsi"
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

        pushButton_openImage = QPushButton('Open', self)
        pushButton_openImage.setGeometry(10, 410, 111, 81)
        pushButton_openImage.clicked.connect(self.loadimage)

        pushButton_save = QPushButton('Save', self)
        pushButton_save.setGeometry(10, 500, 111, 81)
        pushButton_save.clicked.connect(self.saveImage)

        self.label_imageOri = QLabel('Image Ori', self)
        self.label_imageOri.setGeometry(150, 20, 336, 256)
        self.label_imageOri.setFrameShape(QFrame.Panel)
        self.label_imageOri.setObjectName('label_imageOri')

        self.label_maskingHotspot = QLabel('Masking Hotspot', self)
        self.label_maskingHotspot.setGeometry(500, 20, 336, 256)
        self.label_maskingHotspot.setFrameShape(QFrame.Panel)
        self.label_maskingHotspot.setObjectName('label_maskingHotspot')

        self.label_maskingPanel = QLabel('Masking Panel', self)
        self.label_maskingPanel.setGeometry(150, 300, 336, 256)
        self.label_maskingPanel.setFrameShape(QFrame.Panel)
        self.label_maskingPanel.setObjectName('label_maskingPanel')

        self.label_finalImage = QLabel('Final Image', self)
        self.label_finalImage.setGeometry(500, 300, 336, 256)
        self.label_finalImage.setFrameShape(QFrame.Panel)
        self.label_finalImage.setObjectName('label_finalImage') 

        line = QFrame(self)
        line.setGeometry(122, 0, 31, 760)
        line.setLineWidth(5)
        line.setFrameShape(QFrame.VLine)
        line.setFrameShadow(QFrame.Plain)

    def mode_1(self):
        pass

    def mode_2(self):
        self.cams = Window1() 
        self.cams.show()
        self.close()

    def loadimage(self):
        self.filename = QFileDialog.getOpenFileName(filter="Image (*.*)")[0]
        image = cv.imread(self.filename)
        ori_img = self.convert_cv_qt(image)
        mask_hot = self.convert_maskingHotspot(image)
        mask_panel = self.convert_maskingPanel(image)
        final_img = self.convert_final(image)
        max_area = self.maxArea(image)
        self.label_imageOri.setPixmap(ori_img)
        self.label_maskingHotspot.setPixmap(mask_hot)
        self.label_maskingPanel.setPixmap(mask_panel)
        self.label_finalImage.setPixmap(final_img)

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

    def saveImage(self):
        pass
        
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

        self.label_maskingBird = QLabel('Masking Bird Dropping', self)
        self.label_maskingBird.setGeometry(500, 20, 336, 256)
        self.label_maskingBird.setFrameShape(QFrame.Panel)
        self.label_maskingBird.setObjectName('label_maskingBird')

        self.label_maskingPanel = QLabel('Masking Panel', self)
        self.label_maskingPanel.setGeometry(150, 300, 336, 256)
        self.label_maskingPanel.setFrameShape(QFrame.Panel)
        self.label_maskingPanel.setObjectName('label_maskingPanel')

        self.label_finalImage = QLabel('Final Image', self)
        self.label_finalImage.setGeometry(500, 300, 336, 256)
        self.label_finalImage.setFrameShape(QFrame.Panel)
        self.label_finalImage.setObjectName('label_finalImage') 

        line = QFrame(self)
        line.setGeometry(122, 0, 31, 760)
        line.setLineWidth(5)
        line.setFrameShape(QFrame.VLine)
        line.setFrameShadow(QFrame.Plain)


    def mode_1(self):
        self.cams = Window() 
        self.cams.show()
        self.close()

    def mode_2(self):
        pass

    def loadimage(self):
        self.filename = QFileDialog.getOpenFileName(filter="Image (*.*)")[0]
        image = cv.imread(self.filename)
        ori_img = self.convert_cv_qt(image)
        mask_bird = self.convert_maskingBird(image)
        mask_panel = self.convert_maskingPanel(image)
        final_img = self.convert_final(image)
        self.label_imageOri.setPixmap(ori_img)
        self.label_maskingBird.setPixmap(mask_bird)
        self.label_maskingPanel.setPixmap(mask_panel)
        self.label_finalImage.setPixmap(final_img)

    def convert_cv_qt(self, image):
        rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        h, w , ch= rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h , QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(336, 256, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def convert_maskingPanel(self, image):
        blur = cv.GaussianBlur(image, (5,5), 2)
        hsv = cv.cvtColor(blur, cv.COLOR_BGR2HSV)
        l_thres = np.array([60, 0, 0])
        u_thres = np.array([164, 190, 255])
        mask = cv.inRange(hsv, l_thres, u_thres)
        kernal = np.ones((2,2),"uint8")

        panel = cv.morphologyEx(mask,cv.MORPH_ELLIPSE,kernal)
        panel = cv.erode(panel,cv.getStructuringElement(cv.MORPH_ELLIPSE, (8,8)))
        panel = cv.dilate(panel,cv.getStructuringElement(cv.MORPH_RECT, (8,8)))
        panel = cv.dilate(panel,cv.getStructuringElement(cv.MORPH_RECT, (9,9)))
        panel = cv.erode(panel,cv.getStructuringElement(cv.MORPH_ERODE, (7,7)))

        result = cv.bitwise_and(image, image, mask=panel)

        contours, hierarchy = cv.findContours(panel, 1, 2)

        cnt = contours[0]

        result = cv.drawContours(result, contours, 0, (255,0,0), 3)

        result= cv.cvtColor(result, cv.COLOR_BGR2RGB)

        h, w , ch= result.shape

        convert_to_Qt_format = QtGui.QImage(result.data, w, h , QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(336, 256, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)


    def convert_maskingBird(self,image):
        blur = cv.GaussianBlur(image, (5,5), 2)
        rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        hsv = cv.cvtColor(blur, cv.COLOR_BGR2HSV)
        l_thres = np.array([60, 0, 0])
        u_thres = np.array([164, 190, 255])
        mask = cv.inRange(hsv, l_thres, u_thres)
        kernal = np.ones((2,2),"uint8")

        panel = cv.morphologyEx(mask,cv.MORPH_ELLIPSE,kernal)
        panel = cv.erode(panel,cv.getStructuringElement(cv.MORPH_ELLIPSE, (8,8)))
        panel = cv.dilate(panel,cv.getStructuringElement(cv.MORPH_RECT, (8,8)))
        panel = cv.dilate(panel,cv.getStructuringElement(cv.MORPH_RECT, (9,9)))
        panel = cv.erode(panel,cv.getStructuringElement(cv.MORPH_ERODE, (7,7)))

        result = cv.bitwise_and(image, image, mask=panel)

        contours, hierarchy = cv.findContours(panel, 1, 2)

        cnt = contours[0]

        modul = cv.drawContours(result, contours, 0, (255,0,0), 3)

        citra2 = cv.cvtColor(result, cv.COLOR_BGR2HSV)

        l_thres2 = np.array([14, 0, 96])
        u_thres2 = np.array([60, 119, 238])

        mask2 = cv.inRange(citra2, l_thres2, u_thres2)

        soil = cv.morphologyEx(mask2,cv.MORPH_ELLIPSE,kernal)
        soil = cv.erode(soil,cv.getStructuringElement(cv.MORPH_ELLIPSE, (8,8)))
        soil = cv.dilate(soil,cv.getStructuringElement(cv.MORPH_RECT, (14,14)))
        soil = cv.dilate(soil,cv.getStructuringElement(cv.MORPH_RECT, (14,14)))
        soil = cv.erode(soil,cv.getStructuringElement(cv.MORPH_ERODE, (3,3)))

        contours2, hierarchy = cv.findContours(soil, 1, 2)

        cnt1 = contours2[0]

        result2 = cv.drawContours(rgb_image, contours2, 0, (255,0,0), 2)

        #result2 = cv.cvtColor(result2, cv.COLOR_BGR2RGB)

        h, w , ch= result2.shape

        convert_to_Qt_format = QtGui.QImage(result2.data, w, h , QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(336, 256, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def convert_final(self, image):
        blur = cv.GaussianBlur(image, (5,5), 2)
        hsv = cv.cvtColor(blur, cv.COLOR_BGR2HSV)
        l_thres = np.array([60, 0, 0])
        u_thres = np.array([164, 190, 255])
        mask = cv.inRange(hsv, l_thres, u_thres)
        kernal = np.ones((2,2),"uint8")

        panel = cv.morphologyEx(mask,cv.MORPH_ELLIPSE,kernal)
        panel = cv.erode(panel,cv.getStructuringElement(cv.MORPH_ELLIPSE, (8,8)))
        panel = cv.dilate(panel,cv.getStructuringElement(cv.MORPH_RECT, (8,8)))
        panel = cv.dilate(panel,cv.getStructuringElement(cv.MORPH_RECT, (9,9)))
        panel = cv.erode(panel,cv.getStructuringElement(cv.MORPH_ERODE, (7,7)))

        result = cv.bitwise_and(image, image, mask=panel)

        contours, hierarchy = cv.findContours(panel, 1, 2)

        cnt = contours[0]

        modul = cv.drawContours(image, contours, 0, (255,0,0), 3)

        citra2 = cv.cvtColor(result, cv.COLOR_BGR2HSV)

        l_thres2 = np.array([14, 0, 96])
        u_thres2 = np.array([60, 119, 238])

        mask2 = cv.inRange(citra2, l_thres2, u_thres2)

        soil = cv.morphologyEx(mask2,cv.MORPH_ELLIPSE,kernal)
        soil = cv.erode(soil,cv.getStructuringElement(cv.MORPH_ELLIPSE, (8,8)))
        soil = cv.dilate(soil,cv.getStructuringElement(cv.MORPH_RECT, (14,14)))
        soil = cv.dilate(soil,cv.getStructuringElement(cv.MORPH_RECT, (14,14)))
        soil = cv.erode(soil,cv.getStructuringElement(cv.MORPH_ERODE, (3,3)))

        contours2, hierarchy = cv.findContours(soil, 1, 2)

        cnt1 = contours2[0]

        result2 = cv.drawContours(modul, contours2, 0, (0,0,255), 2)

        result2 = cv.cvtColor(result2, cv.COLOR_BGR2RGB)

        h, w , ch= result2.shape

        convert_to_Qt_format = QtGui.QImage(result2.data, w, h , QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(336, 256, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def saveImage(self):
        pass

if __name__ == '__main__':
    app=QApplication(sys.argv)
    ex=Window()
    sys.exit(app.exec_())