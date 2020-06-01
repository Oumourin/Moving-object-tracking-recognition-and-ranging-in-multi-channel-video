import cv2 as cv
import numpy as np
import time


def surf_image_match(left_image, right_image):
    gray_left_image = cv.cvtColor(left_image, cv.COLOR_BGR2GRAY)
    gray_right_image = cv.cvtColor(right_image, cv.COLOR_BGR2GRAY)
    gpu_gray_left_image = cv.cuda_GpuMat(gray_left_image)
    gpu_gray_right_image = cv.cuda_GpuMat(gray_right_image)
    surf = cv.cuda.SURF_CUDA_create(4000)
    # gpu_left_key_points, gpu_left_descriptors = surf(gpu_gray_left_image, None)
    # gpu_right_key_points, gpu_right_descriptors = surf(gpu_gray_right_image, None)
    gpu_left_key_points, gpu_left_descriptors = surf.detectWithDescriptors(gpu_gray_left_image, None)
    gpu_right_key_points, gpu_right_descriptors = surf.detectWithDescriptors(gpu_gray_right_image, None)
    # left_key_points, left_descriptors = cv.cuda_SURF_CUDA.detectWithDescriptors(surf, gpu_gray_left_image, None)
    # right_key_points, right_descriptors = cv.cuda_SURF_CUDA.detectWithDescriptors(surf, gpu_gray_right_image, None)

    matcher = cv.cuda.DescriptorMatcher_createBFMatcher(cv.NORM_L2)
    matches = matcher.knnMatch(gpu_left_descriptors, gpu_right_descriptors, k=2)
    # DMatch List
    good_matches = [[m] for m, n in matches if m.distance < 0.7 * n.distance]
    # KeyPoints
    left_key_points = cv.cuda_SURF_CUDA.downloadKeypoints(surf, gpu_left_key_points)
    right_key_points = cv.cuda_SURF_CUDA.downloadKeypoints(surf, gpu_right_key_points)
    # return_image = cv.drawMatchesKnn(left_image, left_key_points, right_image, right_key_points, good_matches, None)
    return cv.drawMatchesKnn(left_image, left_key_points, right_image, right_key_points, good_matches, None, flags=2)


def is_same_object(left_frame, right_frame, left_boxes, left_indices, right_boxes, right_indices):
    gray_left_image = cv.cvtColor(left_frame, cv.COLOR_BGR2GRAY)
    gray_right_image = cv.cvtColor(right_frame, cv.COLOR_BGR2GRAY)
    gpu_gray_left_image = cv.cuda_GpuMat(gray_left_image)
    gpu_gray_right_image = cv.cuda_GpuMat(gray_right_image)
    surf = cv.cuda.SURF_CUDA_create(4000)
    gpu_left_key_points, gpu_left_descriptors = surf.detectWithDescriptors(gpu_gray_left_image, None)
    gpu_right_key_points, gpu_right_descriptors = surf.detectWithDescriptors(gpu_gray_right_image, None)

    matcher = cv.cuda.DescriptorMatcher_createBFMatcher(cv.NORM_L2)
    matches = matcher.knnMatch(gpu_left_descriptors, gpu_right_descriptors, k=2)
    good_match = [[m] for m, n in matches if m.distance < 0.7 * n.distance]
    left_key_points = cv.cuda_SURF_CUDA.downloadKeypoints(surf, gpu_left_key_points)
    right_key_points = cv.cuda_SURF_CUDA.downloadKeypoints(surf, gpu_right_key_points)

    flag = []
    for left_index in left_indices:
        is_find = False
        left_idx = left_index[0]
        left_box = left_boxes[left_idx]
        for match in good_match:
            if (left_box[0] <= left_key_points[match[0].queryIdx].pt[0] <= (left_box[0] + left_box[2])) and (left_box[1] <= left_key_points[match[0].queryIdx].pt[1] <= (left_box[1] + left_box[3])):
                right_match_point = right_key_points[match[0].trainIdx].pt
                for right_index in right_indices:
                    right_idx = right_index[0]
                    right_box = right_boxes[right_idx]
                    if (right_box[0] <= right_match_point[0] <= right_box[0] + right_box[2]) and (right_box[1] <= right_match_point[1] <= right_box[3]):
                        if right_idx not in flag:
                            flag.append(right_idx)
                            is_find = True
                            break
            if is_find:
                break
        if not is_find:
            flag.append(-1)
    return flag


def surf_image_stitch(left_frame, right_frame):
    surf = cv.cuda.SURF_CUDA_create(200)
    gray_left_frame = cv.cvtColor(left_frame, cv.COLOR_BGR2GRAY)
    gray_right_frame = cv.cvtColor(right_frame, cv.COLOR_BGR2GRAY)
    gpu_gray_left_frame = cv.cuda_GpuMat(gray_left_frame)
    gpu_gray_right_frame = cv.cuda_GpuMat(gray_right_frame)
    gpu_left_key_points, gpu_left_descriptors = surf.detectWithDescriptors(gpu_gray_left_frame, None)
    gpu_right_key_points, gpu_right_descriptors = surf.detectWithDescriptors(gpu_gray_right_frame, None)

    matcher = cv.cuda.DescriptorMatcher_createBFMatcher(cv.NORM_L2)
    matches = matcher.knnMatch(gpu_left_descriptors, gpu_right_descriptors, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.5 * n.distance]
    left_key_points = cv.cuda_SURF_CUDA.downloadKeypoints(surf, gpu_left_key_points)
    right_key_points = cv.cuda_SURF_CUDA.downloadKeypoints(surf, gpu_right_key_points)
    left_points = np.array([left_key_points[m.queryIdx].pt for m in good_matches])
    right_points = np.array([right_key_points[m.trainIdx].pt for m in good_matches])
    h_matrix = cv.findHomography(left_points, right_points)
    left_height, left_width = left_frame.shape[:2]
    right_height, right_height = right_frame.shape[:2]
    transform_matrix = np.array([[1.0, 0, left_width], [0, 1.0, 0], [0, 0, 1.0]])
    m_matrix = np.dot(transform_matrix, h_matrix[0])
    corners = cv.warpPerspective(left_frame, m_matrix, (left_width * 2, left_height))
    corners[0:left_height, left_width:left_width*2] = right_frame
    return corners


def cpu_surf_stitch(left_frame, right_frame):
    surf = cv.xfeatures2d.SURF_create(400)
    left_points, left_descriptors = surf.detectAndCompute(left_frame, None)
    right_points, right_descriptors = surf.detectAndCompute(right_frame, None)

    index_parameter = dict(algorithm=0, trees=5)
    search_parameter = dict(check=50)
    matcher = cv.FlannBasedMatcher(index_parameter, search_parameter)
    matches = matcher.knnMatch(left_descriptors, right_descriptors, k=2)

    good_matches = [m for m, n in matches if m.distance < 0.5 * n.distance]
    left_point = np.array([left_points[m.queryIdx].pt for m in good_matches])
    right_point = np.array([right_points[m.trainIdx].pt for m in good_matches])
    h_matrix = cv.findHomography(left_point, right_point)
    left_height, left_width = left_frame.shape[:2]
    transform_matrix = np.array([[1.0, 0, left_width], [0, 1.0, 0], [0, 0, 1.0]])
    m_matrix = np.dot(transform_matrix, h_matrix[0])
    corners = cv.warpPerspective(left_frame, m_matrix, (left_width * 2, left_height))
    corners[0:left_height, left_width:left_width * 2] = right_frame
    return corners


def cpu_surf_match(left_frame, right_frame):
    surf = cv.xfeatures2d.SURF_create(400)
    left_points, left_descriptors = surf.detectAndCompute(left_frame, None)
    right_points, right_descriptors = surf.detectAndCompute(right_frame, None)

    index_parameter = dict(algorithm=0, trees=5)
    search_parameter = dict(check=50)
    matcher = cv.FlannBasedMatcher(index_parameter, search_parameter)
    matches = matcher.knnMatch(left_descriptors, right_descriptors, k=2)

    good_matches = [[m] for m, n in matches if m.distance < 0.7 * n.distance]
    return cv.drawMatchesKnn(left_image, left_points, right_image, right_points, good_matches, None, flags=2)


if __name__ == '__main__':
    # left_capture = cv.VideoCapture(1)
    # right_capture = cv.VideoCapture(2)
    # while left_capture.isOpened() and right_capture.isOpened():
    #     left_ret, left_image = left_capture.read()
    #     right_ret, right_image = right_capture.read()
    #     if left_ret is False or right_ret is False:
    #         break
    #     return_image = surf_image_match(left_image, right_image)
    #     cv.imshow("Matching", return_image)
    #     c = cv.waitKey(1)
    #     if c == 27:
    #         break
    # left_capture.release()
    # right_capture.release()
    # # cv.destroyAllWindows()
    #
    # left_image = cv.imread("")
    # right_image = cv.imread("")
    # left_image = cv.resize(left_image, (640, 360))
    # right_image = cv.resize(right_image, (640, 360))
    # stitch_image = surf_image_stitch(left_image, right_image)
    # cv.imshow("Stitch", stitch_image)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    left_cap = cv.VideoCapture("")
    right_cap = cv.VideoCapture("")
    start_time = time.time()
    while left_cap.isOpened() and right_cap.isOpened():
        left_ret, left_frame = left_cap.read()
        right_ret, right_frame = right_cap.read()
        if left_ret is False or right_ret is False:
            break
        left_frame = cv.resize(left_frame, (640, 360))
        right_frame = cv.resize(right_frame, (640, 360))
        stitch_frame = surf_image_stitch(left_frame, right_frame)
        cv.imshow("Stitch", stitch_frame)
        c = cv.waitKey(1)
        if c == 27:
            break
    end_time = time.time()
    print((end_time - start_time))
    left_cap.release()
    right_cap.release()
    cv.destroyAllWindows()
