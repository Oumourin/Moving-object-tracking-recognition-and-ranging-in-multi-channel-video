import ctypes
import cv2 as cv
import numpy as np
import time


def load_library():
    return ctypes.cdll.LoadLibrary("C:/Users/maoli/PycharmProjects/Graduation Project/SURF/SURF_DLL.dll")


# void c_surfImageMatch(unsigned char* _leftImage, int _leftRows,
# int _leftCols, unsigned char* _rightImage, int _rightRows, int _rightCols, unsigned char* _returnImage)
def surf_image_match(left_image, left_rows, left_cols,
                     right_image, right_rows, right_cols, return_image):
    surf = load_library()
    surf.c_surfImageMatch(left_image, left_rows, left_cols,
                          right_image, right_rows, right_cols, return_image)
    return return_image


# void c_isSameObject(unsigned char* _leftImage, int _leftRows, int _leftCols,
# 	unsigned char* _rightImage, int _rightRows, int _rightCols,
# 	int _leftX[], int _leftY[], int _leftWidth[], int _leftHeight[], int _leftArrayLength,
# 	int _rightX[], int _rightY[], int _rightWidth[], int _rightHeight[], int _rightArrayLength,
# 	int _flag[], int _flagArrayLenth)
# def is_same_object(left_image, right_image, left_X[], left_Y[], left_width[], left_height[], right_X[], right_Y[], right_width[], right_height[], flag[]):
#     (left_rows, left_cols) = (left_image.shape[0], left_image.shape[1])
#     (right_rows, right_cols) = (right_image.shape[0], left_image.shape[1])


def image_match(left_image, right_image):
    left_image = cv.resize(left_image, (640, 360))
    right_image = cv.resize(right_image, (640, 360))
    (left_rows, left_cols) = (left_image.shape[0], left_image.shape[1])
    (right_rows, right_cols) = (right_image.shape[0], right_image.shape[1])
    return_image = np.zeros(dtype=np.uint8, shape=(left_rows, left_cols + right_cols, 3))
    surf_image_match(left_image.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)), left_rows, left_cols,
                     right_image.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)), right_rows, right_cols,
                     return_image=return_image.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)))
    # cv.imshow("Matching", return_image)
    return return_image


if __name__ == '__main__':
    left_cap = cv.VideoCapture(1)
    right_cap = cv.VideoCapture(2)
    # left_cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    # left_cap.set(cv.CAP_PROP_FRAME_HEIGHT, 360)
    # right_cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    # right_cap.set(cv.CAP_PROP_FRAME_HEIGHT, 360)
    while left_cap.isOpened() and right_cap.isOpened():
        ret, left_image = left_cap.read()
        ret, right_image = right_cap.read()
        start_time = time.time()
        match_image = image_match(left_image, right_image)
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        # left_image = cv.resize(left_image, (640, 360))
        # right_image = cv.resize(right_image, (640, 360))
        # cv.imshow("left", left_image)
        # cv.imshow("right", right_image)
        # (left_rows, left_cols) = (left_image.shape[0], left_image.shape[1])
        # (right_rows, right_cols) = (right_image.shape[0], right_image.shape[1])
        # return_image = np.zeros(dtype=np.uint8, shape=(left_rows, left_cols + right_cols, 3))
        # surf_image_match(left_image.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)), left_rows, left_cols,
        #                  right_image.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)), right_rows, right_cols,
        #                  return_image=return_image.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)))
        cv.putText(match_image, str(round(fps, 2)), (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (187, 197, 57))
        cv.imshow("Matching", match_image)
        c = cv.waitKey(1)
        if c == 27:
            break
    left_cap.release()
    right_cap.release()
    cv.destroyAllWindows()
