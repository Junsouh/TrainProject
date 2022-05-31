import cv2 as cv
import numpy as np

def increase_brightness(img, value=30):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv.merge((h, s, v))
    img = cv.cvtColor(final_hsv, cv.COLOR_HSV2BGR)
    return img

if __name__ == "__main__":
    cap = cv.VideoCapture('3686.mp4')

    width = int(cap.get(3))
    height = int(cap.get(4))

    cropwidth = 600

    pts = np.array([[int(width/2 - cropwidth), height], [int(width/2 - cropwidth), height-200], [int(width/2 + cropwidth - 200), height-200], [int(width/2 + cropwidth - 200), height]])
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    writer = cv.VideoWriter('output.mp4', fourcc, 30, (600, 200))


    if cap.isOpened() == False:
        print("didn't open")
    else:
        print("video opened")

    while cap.isOpened():
        rect = cv.boundingRect(pts)
        x, y, wi, h = rect
        ret, frame = cap.read()
        if ret == True:

            modified = frame.copy()
            modified = increase_brightness(modified, value=20)
            modified = modified[y:y+h, x:x+wi]

            edges = cv.Canny(modified, 0, 255)
            linesP = cv.HoughLinesP(edges, rho=4, theta=np.pi / 180, threshold=43, minLineLength=20, maxLineGap=50)
            if linesP is not None:
                for i in range(0, len(linesP)):
                    l = linesP[i][0]
                    cv.line(modified, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv.LINE_AA)

            cv.imshow("output", modified)
            writer.write(modified)
            if cv.waitKey(25) & 0xFF == ord('q'):
                break

        else:
            print("video ended")
            break

    writer.release()
    cap.release()





