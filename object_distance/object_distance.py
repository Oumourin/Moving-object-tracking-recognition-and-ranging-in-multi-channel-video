import cv2 as cv
import numpy as np
import json


def load_json_data():
    with open("config.json", 'r') as f:
        load_json = json.load(f)
    camera_parameter = load_json["camera2"]
    known_distance = camera_parameter["known_distance"]
    known_width = camera_parameter["known_width"]
    known_height = camera_parameter["known_height"]
    return known_distance, known_width, known_height


def find_marker(image: np.array):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (5, 5), 0)
    edged = cv.Canny(gray, 35, 125)

    cnts, _ = cv.findContours(edged.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv.contourArea)

    return cv.minAreaRect(c)


def distance_to_camera(known_width, focal_length, per_width):
    return (known_width * focal_length) / per_width


def get_focal_length():
    image = cv.imread("")   # 放置已知距离图片
    marker = find_marker(image)
    known_distance, know_width, _ = load_json_data()
    return (marker[1][0] * known_distance) / know_width
    # return 700.0


def get_distance(bottom_y, top_y, class_name, camera_name):
    with open("config.json", 'r') as f:
        load_json = json.load(f)
    get_class_parameter = load_json["classes"]
    get_camera_focal_length = load_json["focal_length"]
    try:
        get_static_parameter = get_class_parameter[str(class_name)]
        get_camera_parameter = get_camera_focal_length[str(camera_name)]
    except KeyError:
        return -1
    pix_height = bottom_y - top_y
    if pix_height == 0:
        return -1
    inches = distance_to_camera(get_static_parameter, get_camera_parameter, pix_height)
    # print(get_focal_length())
    # print(inches)
    transform_cm = (inches * 30.48) / 12
    return transform_cm


if __name__ == "__main__":
    cap = cv.VideoCapture(1)
    while True:
        # ret, image = cap.read()
        # while ret is False:
        #     break
        image = cv.imread("")
        # ret, image = cap.read()
        marker = find_marker(image)

        box = cv.boxPoints(marker)
        box = np.int0(box)
        cv.drawContours(image, [box], -1, (0, 0, 255), 2)

        cv.imshow("Box", image)

        # for (xA, yA, xB, yB) in marker:
        #     cv.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
        # distance = get_distance(yA, yB, "known_height")
        # if distance == 0:
        #     continue
        # print(distance)
        # cv.putText(image, distance, (xA, yA), cv.FONT_HERSHEY_SIMPLEX, 1.0,
        #            (255, 0, 0), 2)
        # cv.imshow("capture", image)
        c = cv.waitKey(1)
        if c == 27:
            break
    cap.release()
    cv.destroyAllWindows()
