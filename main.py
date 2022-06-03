import cv2 as cv
import numpy as np
from PIL import Image, ImageFilter
import math

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

def increase_brightness(img, value=30):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv.merge((h, s, v))
    img = cv.cvtColor(final_hsv, cv.COLOR_HSV2BGR)
    return img

class LineComponent:
    def __init__(self, p1, p2):
        self.centroid = Point((p1.x + p2.x) / 2, (p1.y + p2.y) / 2)
        self.length = self.distanceBetweenPoint2D(p1, p2)
        self.m = self.lineSlope(p1, p2)
        self.angle = math.atan(self.m) * 180 / math.pi
        if self.angle < 0: self.angle = 180 + self.angle
        self.p1 = p1
        self.p2 = p2

    def lineSlope(self, p1, p2):
        diffX = p2.x - p1.x
        diffY = p2.y - p1.y
        if diffX == 0:
            return 100
        else:
            return diffY / diffX

    def distanceBetweenPoint2D(self, p1, p2):
        return math.sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2))

if __name__ == "__main__":
    cap = cv.VideoCapture('3686.mp4')

    width = int(cap.get(3))
    height = int(cap.get(4))

    cropwidth = 600

    pts = np.array([[int(width/2 - cropwidth), height], [int(width/2 - cropwidth), height-200], [int(width/2 + cropwidth - 400), height-200], [int(width/2 + cropwidth - 400), height]])
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    writer = cv.VideoWriter('output.mp4', fourcc, 30, (cropwidth * 2 - 400, 200))


    if cap.isOpened() == False:
        print("didn't open")
    else:
        print("video opened")

    while cap.isOpened():
        rect = cv.boundingRect(pts)
        x, y, wi, h = rect
        ret, frame = cap.read()
        # frame = cv.bitwise_not(frame)
        if ret == True:
            # alpha = 1.4
            # beta = 0
            # gamma = 1
            # #
            # #
            # frame = frame.copy()
            # frame = cv.convertScaleAbs(frame, alpha = alpha, beta = beta)
            # lookUpTable = np.empty((1,256),np.uint8)
            # for i in range(256):
            #     lookUpTable[0,i] = np.clip(pow(i/255.0, gamma) * 255.0, 0, 255)
            # frame = cv.LUT(frame, lookUpTable)


            # convert array to pil image
            im_pil = Image.fromarray(frame)

            # Converting the image to grayscale, as edge detection
            # requires input image to be of mode = Grayscale (L)
            im_pil = im_pil.convert("L")

            # Detecting Edges on the Image using the argument ImageFilter.FIND_EDGES
            im_pil = im_pil.filter(ImageFilter.FIND_EDGES)

            im_np = np.asarray(im_pil)
            cv.imshow('pil_find_edge', im_np)

            cv.threshold(im_np, 20, 255, cv.THRESH_BINARY, im_np)
            cv.imshow("threshold", im_np)
            # modified = increase_brightness(modified, value=40)
            # modified = modified[y:y+h, x:x+wi]

            # modified = cv.cvtColor(modified, cv.COLOR_BGR2GRAY)
            # (thresh, modified) = cv.threshold(modified, 50, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

            # kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
            # smooth = cv.morphologyEx(grayimg, cv.MORPH_DILATE, kernel)
            #
            # divide = cv.divide(grayimg, smooth, scale = 255)
            # threshold = cv.adaptiveThreshold(grayimg, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 7, 4)
            # modified = frame.copy()
            # edges = cv.Canny(modified, 0, 255)
            # cv.imshow('canny', edges)
            lines = []

            linesP = cv.HoughLinesP(im_np, rho=4, theta=np.pi / 180, threshold=43, minLineLength=20, maxLineGap=50)
            if linesP is not None:
                targetRange = (Point(650, 550), Point(800, 700))
                for i in range(0, len(linesP)):
                    l = linesP[i][0]
                    line = LineComponent(Point(l[0], l[1]), Point(l[2], l[3]))

                    if line.centroid.x < targetRange[0].x or line.centroid.x > targetRange[1].x or line.centroid.y < \
                            targetRange[0].y or line.centroid.y > targetRange[1].y:
                        continue
                    if line.length < 50:
                        continue
                    if line.m < 3 and line.m > -3:
                        continue

                    cv.line(frame, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv.LINE_AA)
                    lines.append(line)
            if len(lines) > 0:
                sum = 0
                for l in lines:
                    sum += l.angle
                angle = sum / len(lines)
                if angle > 85 and angle < 95:
                    cv.arrowedLine(frame, (1000, 600), (1000, 500), (0, 255, 0), 10)
                if angle <= 85 and angle >= 0:
                    cv.arrowedLine(frame, (1000, 600), (950, 500), (0, 255, 255), 10)
                if angle >= 95 and angle < 180:
                    cv.arrowedLine(frame, (1000, 600), (1050, 500), (0, 255, 255), 10)
            else:
                cv.arrowedLine(frame, (1000, 600), (1000, 500), (0, 255, 0), 10)

            cv.imshow("output", frame)
            writer.write(im_np)
            if cv.waitKey(25) & 0xFF == ord('q'):
                break

        else:
            print("video ended")
            break

    writer.release()
    cap.release()





