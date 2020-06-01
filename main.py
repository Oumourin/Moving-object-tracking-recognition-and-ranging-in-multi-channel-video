import cv2 as cv
import object_detection
import object_distance
import random
from SURF import surf
import numpy as np


if __name__ == '__main__':
    # cap = cv.VideoCapture(1)
    #
    # # Load Net
    # net, classes = object_detection.load_net()
    # while True:
    #     ret, image = cap.read()
    #     if ret is False:
    #         break
    #     height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
    #     width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
    #     indices, boxes, class_ids = object_detection.object_detection(image, net)
    #     for i in indices:
    #         i = i[0]
    #         box = boxes[i]
    #         left = box[0]
    #         top = box[1]
    #         width = box[2]
    #         height = box[3]
    #         cv.rectangle(image, (left, top), (left+width, top+height), (57, 197, 187), 2, 8, 0)
    #         distance = object_distance.get_distance(bottom_y=top+height, top_y=top, class_name=classes[class_ids[i]], camera_name="LOGITECH")
    #         if distance == 0:
    #             cv.putText(image, classes[class_ids[i]], (left, top),
    #                        cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 192, 203), 2)
    #         else:
    #             string = classes[class_ids[i]] + "  Distance" + str(round(distance, 2)) + "cm"
    #             cv.putText(image, string, (left, top),
    #                        cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 192, 203), 2)
    #         c = cv.waitKey(1)
    #
    #         if c == 27:
    #             break
    #         cv.imshow("Result", image)
    # cap.release()
    # cv.destroyAllWindows()
    # logitech Cap 1
    # ausdom Cap 2
    right_cap = cv.VideoCapture(2)
    left_cap = cv.VideoCapture(1)
    net, classes = object_detection.load_net()
    while left_cap.isOpened() and right_cap.isOpened():
        left_ret, left_image = left_cap.read()
        right_ret, right_image = right_cap.read()
        if left_ret is False or right_ret is False:
            break
        left_image = cv.resize(left_image, (640, 360))
        right_image = cv.resize(right_image, (640, 360))
        multi_image = np.hstack([left_image, right_image])
        left_indices, left_boxes, left_class_ids = object_detection.object_detection(left_image, net)
        right_indices, right_boxes, right_class_ids = object_detection.object_detection(right_image, net)
        flag = surf.is_same_object(left_image, right_image, left_boxes, left_indices=left_indices,
                                   right_boxes=right_boxes, right_indices=right_indices)
        # print(flag)
        for left_index, right_index in zip(left_indices, right_indices):
            left, right = left_index[0], right_index[0]
            left_box, right_box = left_boxes[left], right_boxes[right]
            left_X, right_X = left_box[0], right_box[0]
            left_Y, right_Y = left_box[1], right_box[1]
            left_width, right_width = left_box[2], right_box[2]
            left_height, right_height = left_box[3], right_box[3]
            cv.rectangle(left_image, (left_X, left_Y),
                         (left_X + left_width, left_Y + left_height), (187, 197, 57), 2, 8, 0)
            cv.rectangle(right_image, (right_X, right_Y),
                         (right_X + right_width, right_Y + right_height), (187, 197, 57), 2, 8, 0)
            left_distance = object_distance.get_distance(bottom_y=left_Y+left_height, top_y=left_Y,
                                                         class_name=classes[left_class_ids[left]], camera_name="LOGITECH")
            right_distance = object_distance.get_distance(bottom_y=right_Y+right_height, top_y=right_Y,
                                                          class_name=classes[right_class_ids[right]], camera_name="AUSDOM")
            if -1 == left_distance:
                cv.putText(left_image, classes[left_class_ids[left]], (left_X, left_Y),
                         cv.FONT_HERSHEY_SIMPLEX, 1.0, (187, 197, 57), 2)
            else:
                string = classes[left_class_ids[left]] + str(round(left_distance, 2)) + "cm"
                cv.putText(left_image, string, (left_X, left_Y),
                           cv.FONT_HERSHEY_SIMPLEX, 1.0, (187, 197, 57), 2)
                print("Left %s" % string)
            if right_distance == -1:
                cv.putText(right_image, classes[right_class_ids[right]], (right_X, right_Y),
                           cv.FONT_HERSHEY_SIMPLEX, 1.0, (187, 197, 57), 2)
            else:
                string = classes[right_class_ids[right]] + str(round(right_distance, 2)) + "cm"
                cv.putText(right_image, string, (right_X, right_Y),
                           cv.FONT_HERSHEY_SIMPLEX, 1.0, (187, 197, 57), 2)
                print("Right %s" % string)
        index = 1
        for idx, val in enumerate(flag):
            if val != -1:
                left_box = left_boxes[idx]
                right_box = right_boxes[val]
                cv.putText(left_image, str(index), (left_box[0], left_box[1]-16), cv.FONT_HERSHEY_SIMPLEX, 2.0,
                           (0, 0, 255), 2)
                cv.putText(right_image, str(index), (right_box[0], right_box[1]-16), cv.FONT_HERSHEY_SIMPLEX, 2.0,
                           (0, 0, 255), 2)
                B = random.randint(0, 255)
                G = random.randint(0, 255)
                R = random.randint(0, 255)
                cv.rectangle(left_image, (left_box[0], left_box[1]),
                             (left_box[0]+left_box[2], left_box[1]+left_box[3]), (B, G, R), 2, 8, 0)
                cv.rectangle(right_image, (right_box[0], right_box[1]),
                             (right_box[0]+right_box[2], right_box[1]+right_box[3]), (B, G, R), 2, 8, 0)
                index += 1
        # left_image = cv.resize(left_image, (1280, 720))
        # right_image = cv.resize(right_image, (1280, 720))
        cv.imshow("Left", left_image)
        cv.imshow("Right", right_image)
        cv.imshow("Multi_Video", multi_image)
        c = cv.waitKey(1)
        if c == 27:
            break
    left_cap.release()
    right_cap.release()
    cv.destroyAllWindows()






