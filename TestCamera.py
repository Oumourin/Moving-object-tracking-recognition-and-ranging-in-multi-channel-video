import cv2 as cv


if __name__ == '__main__':
    cap1 = cv.VideoCapture(1)
    cap2 = cv.VideoCapture(2)
    while cap1.isOpened() and cap2.isOpened():
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        # ret3, frame3 = cap3.read()
        # ret4, frame4 = cap4.read()
        cv.imshow("Cap1", frame1)
        cv.imshow("Cap2", frame2)
        # cv.imshow("Cap3", frame3)
        # cv.imshow("Cap4", frame4)
        c = cv.waitKey(1)
        if c == 27:
            break
    cap1.release()
    cap2.release()
    cv.destroyAllWindows()
