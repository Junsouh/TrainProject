import cv2
import cv2 as cv
import numpy as np
import math, functools
from PIL import Image, ImageFilter
from enum import Enum


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Status(Enum):
    Green = 0
    Left = 1
    Right = 2


def increase_brightness(img, value=30):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv.merge((h, s, v))
    img = cv.cvtColor(final_hsv, cv.COLOR_HSV2BGR)
    return img


def lineSlope(p1, p2):
    diffX = p2.x - p1.x
    diffY = p2.y - p1.y
    if diffX == 0:
        return 100
    else:
        return diffY / diffX


def distanceBetweenPoint2D(p1, p2):
    return math.sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2))


class LineComponent:
    def __init__(self, p1, p2):
        self.centroid = Point((p1.x + p2.x) / 2, (p1.y + p2.y) / 2)
        self.length = distanceBetweenPoint2D(p1, p2)
        self.m = lineSlope(p1, p2)
        self.angle = math.atan(self.m) * 180 / math.pi
        if self.angle < 0: self.angle = 180 + self.angle
        self.p1 = p1
        self.p2 = p2


def verify_angle(lineComponents, status):
    lineSeparation = 760
    if status == Status.Left:
        lineSeparation = 750
    if status == Status.Right:
        lineSeparation = 770
    leftLines = []
    rightLines = []
    for l in lineComponents:
        if l.centroid.x < lineSeparation:
            leftLines.append(l)
        if l.centroid.x >= lineSeparation:
            rightLines.append(l)

    if len(leftLines) > 0 and len(rightLines) > 0:
        sum = 0
        for l in leftLines:
            sum += l.angle
        angleL = sum / len(leftLines)
        sum = 0
        for l in rightLines:
            sum += l.angle
        angleR = sum / len(rightLines)
        # leftLines = sorted(leftLines, key=lambda l : l.length)
        # angleL = leftLines[0].angle

        # rightLines = sorted(rightLines, key=lambda l : l.length)
        # angleR = rightLines[0].angle
        angle = (angleL + angleR) / 2
        if status == Status.Left and angle <= 80 and angle >= 0:  # yellow (left turn)
            return True
        elif status == Status.Right and angle >= 100 and angle < 180:  # yellow (right turn)
            return True
        else:
            return False

    else:
        lines = []
        for l in lineComponents:
            if line.angle > 87 and line.angle < 93:
                continue
            lines.append(l)
        sum = 0
        for l in lines:
            sum += l.angle
        if len(lines) == 0: return False
        angle = sum / len(lines)

        # lines = sorted(lineComponents, key=lambda l : l.length)
        # angle = lines[0].angle
        if (angle <= 80 and angle >= 0) or (angle >= 100 and angle < 180):  # yellow (right turn)
            return True
        else:
            return False


if __name__ == "__main__":
    cap = cv.VideoCapture('3686.mp4')

    width = int(cap.get(3))
    height = int(cap.get(4))

    cropwidth = 600

    pts = np.array([[int(width / 2 - cropwidth), height], [int(width / 2 - cropwidth), height - 200],
                    [int(width / 2 + cropwidth - 400), height - 200], [int(width / 2 + cropwidth - 400), height]])
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    writer = cv.VideoWriter('output.mp4', fourcc, 30, (width, height))

    if cap.isOpened() == False:
        print("didn't open")
    else:
        print("video opened")

    count = 0
    status = Status.Green
    toggle = False
    totalcount = 0
    while cap.isOpened():
        rect = cv.boundingRect(pts)
        x, y, wi, h = rect
        ret, frame = cap.read()
        # modified = frame.copy()
        # modified = cv.bitwise_not(frame)

        if ret == True:
            # alpha = 1
            # beta = 60
            # # gamma = 1
            # frame = cv.convertScaleAbs(frame, alpha = alpha, beta = beta)
            # modified = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            #
            # kernel = np.array([[0, -1, 0],
            #                    [-1, 5, -1],
            #                    [0, -1, 0]])
            # frame = cv.filter2D(frame, ddepth=-1, kernel=kernel)
            # cv.imshow("sharpen", modified)
            #
            # modified = scipy.ndimage.gaussian_filter(modified, 3)
            # cv.imshow("gaussian", modified)
            #
            # modified = cv2.Laplacian(modified, cv.CV_8U)
            # cv.imshow("laplacian", modified)
            #

            # convert array to pil image

            im_pil = Image.fromarray(frame)

            # Converting the image to grayscale, as edge detection
            # requires input image to be of mode = Grayscale (L)
            im_pil = im_pil.convert("L")

            # Detecting Edges on the Image using the argument ImageFilter.FIND_EDGES
            im_pil = im_pil.filter(ImageFilter.FIND_EDGES)

            im_np = np.asarray(im_pil)
            # cv.imshow('pil_find_edge', im_np)
            # cv.imwrite('pil_find_edge.png', im_np)

            cv.threshold(im_np, 20, 100, cv.THRESH_BINARY, im_np)
            # cv.imshow("threshold", im_np)
            # cv.imwrite('threshold.png', im_np)
            # (thresh, modified) = cv.threshold(modified, 50, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
            # cv.imshow("threshold", modified)
            # lookUpTable = np.empty((1,256),np.uint8)
            # for i in range(256):
            #     lookUpTable[0,i] = np.clip(pow(i/255.0, gamma) * 255.0, 0, 255)
            #
            # modified = cv.LUT(modified, lookUpTable)
            # modified = increase_brightness(modified, value=40)
            # modified = modified[y:y+h, x:x+wi]

            # grayimg = cv.cvtColor(modified, cv.COLOR_BGR2GRAY)
            # kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
            # smooth = cv.morphologyEx(grayimg, cv.MORPH_DILATE, kernel)
            #
            # divide = cv.divide(grayimg, smooth, scale = 255)
            # threshold = cv.adaptiveThreshold(grayimg, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 7, 4)
            # modified = frame.copy()
            lines = []
            # edges = cv.Canny(im_np, 0, 255)
            # cv.imshow('canny', edges)
            linesP = cv.HoughLinesP(im_np, rho=4, theta=np.pi / 180, threshold=20, minLineLength=20, maxLineGap=100)
            if linesP is not None:
                targetRange = (Point(600, 550), Point(820, 1000))
                if status is Status.Left:
                    targetRange = (Point(600, 550), Point(800, 750))
                if status is Status.Right:
                    targetRange = (Point(700, 550), Point(820, 750))

                for i in range(0, len(linesP)):
                    l = linesP[i][0]
                    line = LineComponent(Point(l[0], l[1]), Point(l[2], l[3]))

                    if line.centroid.x < targetRange[0].x or line.centroid.x > targetRange[1].x or line.centroid.y < \
                            targetRange[0].y or line.centroid.y > targetRange[1].y:
                        continue
                    # if line.length < 50:
                    #     continue
                    # if line.angle > 85 and line.angle < 95:
                    #     continue
                    if line.m < 3 and line.m > -3:
                        continue

                    cv.line(frame, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv.LINE_AA)
                    lines.append(line)
            if len(lines) > 0:
                sum = 0
                for l in lines:
                    sum += l.angle
                angle = sum / len(lines)

                if angle > 80 and angle < 100:  # green (forward)
                    if totalcount > 70:
                        status = Status.Green
                        cv.arrowedLine(frame, (1000, 600), (1000, 500), (0, 255, 0), 10)
                        cv.circle(frame, (1800, 100), 20, (0, 255, 0), -1)
                        totalcount = 0
                    elif status is Status.Green:
                        cv.arrowedLine(frame, (1000, 600), (1000, 500), (0, 255, 0), 10)
                        cv.circle(frame, (1800, 100), 20, (0, 255, 0), -1)
                        totalcount = 0
                    elif toggle is True and status is Status.Right:
                        if count > 3:
                            cv.arrowedLine(frame, (1000, 600), (1000, 500), (0, 255, 0), 10)
                            cv.circle(frame, (1800, 100), 20, (0, 255, 0), -1)
                            status = Status.Green
                            toggle = False
                            count = 0
                            totalcount += 1
                    elif toggle is True:
                        cv.arrowedLine(frame, (1000, 600), (1000, 500), (0, 255, 0), 10)
                        cv.circle(frame, (1800, 100), 20, (0, 255, 0), -1)
                        status = Status.Green
                        toggle = False
                        count = 0
                        totalcount += 1
                    else:
                        count += 1
                if angle <= 80 and angle >= 0:  # yellow (left turn)
                    if status is Status.Left:
                        if verify_angle(lines, status):
                            cv.arrowedLine(frame, (1000, 600), (950, 500), (0, 255, 255), 10)
                            cv.circle(frame, (1800, 100), 20, (0, 255, 255), -1)
                    elif toggle is True:
                        cv.arrowedLine(frame, (1000, 600), (950, 500), (0, 255, 255), 10)
                        cv.circle(frame, (1800, 100), 20, (0, 255, 255), -1)
                        status = Status.Left
                        toggle = False
                        count = 0
                    elif verify_angle(lines, status):
                        count += 1
                if angle >= 100 and angle < 180:  # yellow (right turn)
                    if status is Status.Right:
                        if verify_angle(lines, status):
                            cv.arrowedLine(frame, (1000, 600), (1050, 500), (0, 255, 255), 10)
                            cv.circle(frame, (1800, 100), 20, (0, 255, 255), -1)
                            totalcount += 1
                    elif toggle is True:
                        cv.arrowedLine(frame, (1000, 600), (1050, 500), (0, 255, 255), 10)
                        cv.circle(frame, (1800, 100), 20, (0, 255, 255), -1)
                        status = Status.Right
                        toggle = False
                        count = 0
                        totalcount += 1
                    elif verify_angle(lines, status):
                        count += 1

            if status is Status.Green:
                cv.arrowedLine(frame, (1000, 600), (1000, 500), (0, 255, 0), 10)
                cv.circle(frame, (1800, 100), 20, (0, 255, 0), -1)
            if status is Status.Left:
                cv.arrowedLine(frame, (1000, 600), (950, 500), (0, 255, 255), 10)
                cv.circle(frame, (1800, 100), 20, (0, 255, 255), -1)
            if status is Status.Right:
                cv.arrowedLine(frame, (1000, 600), (1050, 500), (0, 255, 255), 10)
                cv.circle(frame, (1800, 100), 20, (0, 255, 255), -1)
                totalcount += 1
            if count > 2: toggle = True

            cv.imshow("output", frame)
            writer.write(frame)
            if cv.waitKey(25) & 0xFF == ord('q'):
                break

        else:
            print("video ended")
            break

    writer.release()
    cap.release()





