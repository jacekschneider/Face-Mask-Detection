
import numpy as np
import cv2
fgbg = cv2.createBackgroundSubtractorKNN(detectShadows=False)
lowb = np.array([30, 83, 0], dtype = "uint8")
highb = np.array([112, 255, 53], dtype = "uint8")
low = np.array([30, 20, 120], dtype = "uint8")
high = np.array([131, 200, 255], dtype = "uint8")

def do(path,pathface):
    faceCascade = cv2.CascadeClassifier(pathface)
    led = False
    img = cv2.imread(path)
    img_temp = img
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(imgGray, 1.1, 4)
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    if len(faces):
        faces = np.uint16(np.around(faces))
        for (x, y, w, h) in faces:
            cv2.rectangle(mask, (x, y), (x + w, y + h), (255, 0, 0), 2)
        th, im_th = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        cv2.floodFill(im_th, None, (0, 0), 255)
        im_floodfill_inv = cv2.bitwise_not(im_th)
        img_roi = cv2.bitwise_and(img, img, mask=im_floodfill_inv)
        img = img_roi
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        skinRegionHSV = cv2.inRange(hsv, low, high)
        mean = cv2.mean(img, skinRegionHSV)
        if mean[0]>=180 and mean[1] >=120 and mean[2] >=80:
            led = True
            for (x, y, w, h) in faces:
                cv2.rectangle(img_temp, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            for (x, y, w, h) in faces:
                cv2.rectangle(img_temp, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.imwrite(path, img_temp)
    else:
        fgmask = fgbg.apply(img)
        img_mask = cv2.bitwise_and(img, img, mask=fgmask)
        img_gray = cv2.cvtColor(img_mask, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.medianBlur(img_gray, 5)
        img_canny = cv2.Canny(img_gray, 50, 200)
        circles = cv2.HoughCircles(img_canny, cv2.HOUGH_GRADIENT, 1, 500, param1=50, param2=30, minRadius=50,
                                   maxRadius=100)
        h, w = img.shape[:2]
        mask = np.zeros((h, w), np.uint8)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                cv2.circle(mask, (i[0], i[1]), i[2], (255, 255, 255), 0)
                cv2.circle(mask, (i[0], i[1] + i[2]), int(i[2] * 0.9), (255, 255, 255), 0)
            th, im_th = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            mask = np.zeros((h + 2, w + 2), np.uint8)
            cv2.floodFill(im_th, mask, (0, 0), 255)
            im_floodfill_inv = cv2.bitwise_not(im_th)
            img_roi = cv2.bitwise_and(img, img, mask=im_floodfill_inv)
            img = img_roi
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            skinRegionHSV = cv2.inRange(hsv, low, high)
            mean = cv2.mean(img, skinRegionHSV)

            skinRegionHSVb = cv2.inRange(hsv, lowb, highb)
            meanb = cv2.mean(img, skinRegionHSVb)

            if mean[0] >= 180 and mean[1] >= 120 and mean[2] >= 80:
                led = True
                for i in circles[0, :]:
                    cv2.rectangle(img_temp, (i[0]-i[2], i[1]-i[2]), (i[0]+i[2], i[1]+2*i[2]), (0, 255, 0), 2)
            elif meanb[0] < 25 and meanb[1] < 20 and meanb[2] < 10:
                led = True
                for i in circles[0, :]:
                    cv2.rectangle(img_temp, (i[0] - i[2], i[1] - i[2]), (i[0] + i[2], i[1] + 2 * i[2]), (0, 255, 0), 2)
            else:
                led = False
            cv2.imwrite(path, img_temp)
    return led