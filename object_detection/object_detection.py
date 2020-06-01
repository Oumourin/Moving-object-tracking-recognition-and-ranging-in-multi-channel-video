import json
import cv2 as cv
import numpy as np


# Load net
def load_net():
    with open("config.json", 'r') as f:
        load_json = json.load(f)
    for get_model in load_json["object_detection"]:
        model = get_model["model"]
        cfg = get_model["cfg"]
        classes = get_model["classes"]
    with open(classes, 'rt') as f:
        load_classes = f.read().rstrip('\n').split('\n')
    net = cv.dnn.readNetFromDarknet(cfg, model)
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
    return net, load_classes


# Object Detection
def object_detection(image: np.array, net):
    h, w = image.shape[:2]
    blob_image = cv.dnn.blobFromImage(image, 1.0/255, (416, 416), None, True, False)
    out_name = net.getUnconnectedOutLayersNames()
    net.setInput(blob_image)
    outs = net.forward(out_name)

    t, _ = net.getPerfProfile()
    fps = 1000 / (t * 1000.0 / cv.getTickFrequency())
    label = 'FPS:%.2f' % fps
    cv.putText(image, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (187, 197, 57))

    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * w)
                center_y = int(detection[1] * h)
                box_width = int(detection[2] * w)
                box_height = int(detection[3] * h)
                left = int(center_x - box_width / 2)
                top = int(center_y - box_height / 2)
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([left, top, box_width, box_height])

    indices = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    return indices, boxes, class_ids


if __name__ == '__main__':
    net, classes = load_net()
    cap = cv.VideoCapture("81826627-1-6.mp4")
    while True:
        ret, image = cap.read()
        if ret is False:
            break

        # image = cv.resize(image, (int(image.shape[1]/2), int(image.shape[0]/2)), interpolation=cv.INTER_AREA)
        height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
        width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
        indices, boxes, class_ids = object_detection(image, net)
        for i in indices:
            i = i[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            cv.rectangle(image, (left, top), (left+width, top+height), (0, 0, 255), 2, 8, 0)
            cv.putText(image, classes[class_ids[i]], (left, top),
                       cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0, 2))
            c = cv.waitKey(1)

        if c == 27:
            break
        cv.imshow("Detection", image)
    cv.waitKey(0)
    cv.destroyAllWindows()
