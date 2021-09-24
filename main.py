import os

import cv2
import numpy as np

VIDEOS_PATH = './videos'
OUTPUT_PATH = './output'
FRAMES_ROOM = 30

intervals = []


def initialize_yolo():
    net = cv2.dnn.readNet("./yolo-coco/yolov3.weights", "./yolo-coco/yolov3.cfg")
    classes = []
    with open("./yolo-coco/obj.names", "r") as f:
        for line in f.readlines():
            classes.append(line.strip())
            intervals.append([])

    layers_names = net.getLayerNames()
    output_layers = [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    return net, classes, colors, output_layers


def detect_objects(img, net, outputLayers):
    blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(outputLayers)
    return blob, outputs


def get_box_dimensions(curr_frame_idx, outputs, height, width):
    boxes = []
    confs = []
    class_ids = []
    for output in outputs:
        for detect in output:
            scores = detect[5:]
            class_id = np.argmax(scores)
            conf = scores[class_id]
            if conf > 0.3:
                if not intervals[class_id] or curr_frame_idx - intervals[class_id][-1][1] > FRAMES_ROOM:
                    intervals[class_id].append([curr_frame_idx, curr_frame_idx])
                    center_x = int(detect[0] * width)
                    center_y = int(detect[1] * height)
                    w = int(detect[2] * width)
                    h = int(detect[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confs.append(float(conf))
                    class_ids.append(class_id)
                else:
                    intervals[class_id][-1][1] = curr_frame_idx
    return boxes, confs, class_ids


def draw_labels(boxes, confs, colors, class_ids, classes, img):
    indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y - 5), font, 1, color, 1)
    img = cv2.resize(img, (800, 600))
    cv2.imshow("Image", img)


def process_video(video):
    model, classes, colors, output_layers = initialize_yolo()
    cap = cv2.VideoCapture(video)
    curr_frame_idx = 0

    while True:
        grabbed, frame = cap.read()

        # if the frame was not grabbed, then we have reached the end
        # of the stream
        if not grabbed:
            break

        height, width, channels = frame.shape
        blob, outputs = detect_objects(frame, model, output_layers)
        boxes, confs, class_ids = get_box_dimensions(curr_frame_idx, outputs, height, width)
        if boxes:
            draw_labels(boxes, confs, colors, class_ids, classes, frame)

        key = cv2.waitKey(1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        curr_frame_idx += 1

    cap.release()


initialize_yolo()
videos = os.listdir(VIDEOS_PATH)
# for video in videos:
process_video(VIDEOS_PATH + '/fire.mp4')
print(intervals)
