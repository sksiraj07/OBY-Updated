from flask import Flask, request, render_template, redirect, send_file, jsonify
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
import multiprocessing as mp

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists(RESULT_FOLDER):
    os.makedirs(RESULT_FOLDER)

# YOLO Configuration
def load_yolo(yolo_path):
    labelsPath = os.path.sep.join([yolo_path, "coco.names"])
    LABELS = open(labelsPath).read().strip().split("\n")

    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

    weightsPath = os.path.sep.join([yolo_path, "yolov3.weights"])
    configPath = os.path.sep.join([yolo_path, "yolov3.cfg"])

    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

    return net, ln, LABELS, COLORS

def detect_objects(image, net, ln, LABELS, COLORS, confidence_threshold=0.5, threshold=0.3):
    (H, W) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)

    boxes = []
    confidences = []
    classIDs = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > confidence_threshold:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, threshold)

    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return image

def process_image(filepath, yolo_path):
    image = cv2.imread(filepath)
    net, ln, LABELS, COLORS = load_yolo(yolo_path)
    result_image = detect_objects(image, net, ln, LABELS, COLORS)

    result_image_path = os.path.join(app.config['RESULT_FOLDER'], 'result_' + os.path.basename(filepath))
    cv2.imwrite(result_image_path, result_image)

    return result_image_path

def process_video(filepath, yolo_path):
    vs = cv2.VideoCapture(filepath)
    (W, H) = (None, None)
    frames = []
    while True:
        (grabbed, frame) = vs.read()
        if not grabbed:
            break
        frames.append(frame)
        if W is None or H is None:
            (H, W) = frame.shape[:2]

    vs.release()

    # Process frames using multiprocessing
    pool = mp.Pool(mp.cpu_count())
    result_frames = pool.starmap(process_frame, [(frame, yolo_path) for frame in frames])
    pool.close()
    pool.join()

    result_video_path = os.path.join(app.config['RESULT_FOLDER'], 'result_' + os.path.basename(filepath))
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(result_video_path, fourcc, 20, (W, H), True)
    for frame in result_frames:
        writer.write(frame)
    writer.release()

    return result_video_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return redirect(request.url)

    file = request.files['image']
    if file.filename == '':
        return redirect(request.url)

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        result_image_path = process_image(filepath, yolo_path)

        return send_file(result_image_path, mimetype='image/jpeg')

@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return redirect(request.url)

    file = request.files['video']
    if file.filename == '':
        return redirect(request.url)

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        result_video_path = process_video(filepath, yolo_path)

        return send_file(result_video_path, mimetype='video/avi')

if __name__ == '__main__':
    yolo_path = "c:/Users/sksir/OneDrive/Desktop/Minor/yolo"
    app.run(debug=True)
