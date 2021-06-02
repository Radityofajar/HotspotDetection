import numpy as np
import cv2 as cv
from scipy.spatial import distance as dist
cap = cv.VideoCapture('4May.mp4')
while cap.isOpened():
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("error")
        break
    gray_image = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cntp, areap, Mp, cxp, cyp, pembatasp, kotakp = ([] for i in range(7))
    CXpanel, CYpanel, areaPanel, CX, CY, areaCell = ([] for b in range(6)) 
    cnt, area, M, cx, cy, pembatas, kotak = ([] for t in range(7))

    thres_cell = 346.0
    thres_panel = 14025.3
    luas_panel = luas_hotspot = 0
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
            cv.drawContours(finalImg, [kotak], -1, (0, 0, 255), 1)
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

    frame[penanda == -1] = [255, 0, 0]

    tinggi, lebar, channel = frame.shape[:3]
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
                cv.drawContours(finalImg, [kotakp], -1, (255, 0, 0), 1)

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

    cv.imshow('frame', finalImg)
    if cv.waitKey(1) == ord('q'):
        break
cap.release()
cv.destroyAllWindows()