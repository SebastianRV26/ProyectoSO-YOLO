import os

import cv2
import numpy as np

import matplotlib.pyplot as plt
import os

VIDEOS_PATH = './videos'
OUTPUT_PATH = './output'
FRAMES_ROOM = 30

intervals = []


# frames_to_seconds(30, 60) -> 0.5
def frames_to_seconds(frames, movie_frames):
    return frames / movie_frames


def frames_in_movie(video):
    return video.get(cv2.CAP_PROP_FPS)


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


def draw_labels(video_name, curr_frame_idx, fps, boxes, confs, colors, class_ids, classes, img):
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
    # create folder
    folder_path = f"{OUTPUT_PATH}/{video_name.replace('.mp4', '')}"
    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)
    # save image
    cv2.imwrite(f"{folder_path}/{str(int(frames_to_seconds(curr_frame_idx, fps)))}.jpg", img)


def process_video(video):
    model, classes, colors, output_layers = initialize_yolo()
    cap = cv2.VideoCapture(VIDEOS_PATH + '/' + video)
    curr_frame_idx = 0
    fps = frames_in_movie(cap)

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
            draw_labels(video, curr_frame_idx, fps, boxes, confs, colors, class_ids, classes, frame)

        key = cv2.waitKey(1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        curr_frame_idx += 1

    cap.release()
    generate_graph(video, fps)


def generate_graph(video_name, fps):
    # Declaring a figure "gnt"
    fig, gnt = plt.subplots()

    # Setting Y-axis limits
    gnt.set_ylim(0, 50)

    # Setting X-axis limits
    # gnt.set_xlim(0, 160)

    # Setting labels for x-axis and y-axis
    gnt.set_xlabel('Segundos desde el inicio')
    gnt.set_ylabel('Objetos')

    # Setting ticks on y-axis
    gnt.set_yticks([15, 25, 35])
    # Labelling tickes of y-axis
    gnt.set_yticklabels(['Pistola', 'Fuego', 'Rifle'])

    # Setting graph attribute
    # gnt.grid(True)

    gun_tuples = [(frames_to_seconds(range[0], fps), frames_to_seconds(range[1] - range[0], fps))
                  for range in intervals[0]]
    gnt.broken_barh(gun_tuples, (10, 9), facecolors='tab:blue')

    fire_tuples = [(frames_to_seconds(range[0], fps), frames_to_seconds(range[1] - range[0], fps))
                   for range in intervals[1]]
    gnt.broken_barh(fire_tuples, (20, 9), facecolors=('tab:red'))

    rifle_tuples = [(frames_to_seconds(range[0], fps), frames_to_seconds(range[1] - range[0], fps))
                    for range in intervals[2]]
    gnt.broken_barh(rifle_tuples, (30, 9), facecolors=('tab:orange'))

    folder_path = f"{OUTPUT_PATH}/{video_name.replace('.mp4', '')}"
    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)
    # save image
    plt.savefig(f"{folder_path}/graph.png")


initialize_yolo()
videos = os.listdir(VIDEOS_PATH)
for video in videos:
    process_video('fire.mp4')
    for _, index in intervals:
        intervals[index] = []
