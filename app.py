from flask import Flask, render_template, request, url_for, flash, redirect, Response
import cv2
import datetime
import imutils
import numpy as np
from bs4 import BeautifulSoup

protopath = "MobileNetSSD_deploy.prototxt"
modelpath = "MobileNetSSD_deploy.caffemodel"
detector = cv2.dnn.readNetFromCaffe(prototxt=protopath, caffeModel=modelpath)

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# ...
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your secret key'


messages = [{'name_device': 'My Webcam',
             'ip_address': None,
             'port': None,
             'url_device': 0},           
            ]
# ...
# ...

@app.route('/')
def view_devices():
    return render_template('view_devices.html', messages=messages)

@app.route('/create/', methods=('GET', 'POST'))
def add_device():
    if request.method == 'POST':
        device_name = request.form['device_name']
        IP_addr = request.form['IP_device']
        port_device = request.form['Port']

        if not device_name:
            flash('Input device name')
        elif not IP_addr:
            flash('Please enter a valid IP address')  
        elif not port_device:
            flash('Please enter a valid port')        
        else:
            URL_device ='http://'+ IP_addr + ':' + port_device + '/video'
            messages.append({'name_device': device_name, 'ip_address': IP_addr , 'port_device': port_device, 'url_device': URL_device})
            return redirect(url_for('view_devices'))
    return render_template('add_device.html')

def generate_frames(source):
    if source == "webcam":
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(source)
    fps_start_time = datetime.datetime.now()
    fps = 0
    total_frames = 0
    while True:
                    
        ## read the camera frame
        success,frame=cap.read()
        frame = imutils.resize(frame, 600)
        total_frames = total_frames + 1
        
        (H, W) = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
        
        detector.setInput(blob)
        person_detections = detector.forward()  
        
        num = 0
        for i in np.arange(0, person_detections.shape[2]):
            confidence = person_detections[0, 0, i, 2]
            if confidence > 0.6:
                idx = int(person_detections[0, 0, i, 1])

                if CLASSES[idx] != "person":
                    continue

                person_box = person_detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = person_box.astype("int")
                num = num + 1

                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 255), 2)
        
        fps_end_time = datetime.datetime.now()
        time_diff = fps_end_time - fps_start_time
        if time_diff.seconds == 0:
            fps = 0.0
        else:
            fps = (total_frames / time_diff.seconds)

        fps_text = "FPS: {:.2f}/n".format(fps)+str(num)

        cv2.putText(frame, fps_text, (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255))
        
        if not success:
            break
        else:
            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()

        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    image = cv2.imread('im2.jpg')
    image = imutils.resize(image, width=600)

    (H, W) = image.shape[:2]

    blob = cv2.dnn.blobFromImage(image, 0.007843, (W, H), 127.5)

    detector.setInput(blob)
    person_detections = detector.forward()
    num = 0

    for i in np.arange(0, person_detections.shape[2]):
        confidence = person_detections[0, 0, i, 2]
        if confidence > 0.1:
            idx = int(person_detections[0, 0, i, 1])

            if CLASSES[idx] != "person":
                continue

            person_box = person_detections[0, 0, i, 3:7] * np.array([W, H, W, H])
            (startX, startY, endX, endY) = person_box.astype("int")

            num = num + 1
            
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(image, str(num), (startX,startY), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255))
            
    cv2.putText(image, str(num), (5,30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255))

@app.route('/webcam_stream',)
def webcam_stream():
    return Response(generate_frames(0),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/device_stream')
def device_stream():
    url = request.args.get('url')

    # Verify that the URL is not empty
    if url:
        return Response(generate_frames(url), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return "Invalid URL"


if __name__ == '__main__':
    app.run(debug=True)