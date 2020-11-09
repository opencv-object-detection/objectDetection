
import os
import cv2
import time
import numpy as np
from urllib.request import urlretrieve

class ObjectDetection():
    
    _MAPPER = {'weights_url' : 'https://pjreddie.com/media/files/yolov3-tiny.weights',
               'weights_dir': './models/yolov3-tiny.weights',
               'model_cfg' : './models/model-config.cfg',
               'names' : './models/names.txt'
               }
    
    def __init__(self):
        self._model = cv2.dnn.readNet(self._MAPPER['weights_dir'], self._MAPPER['model_cfg'])
        with open(self._MAPPER['names'], "r") as infile:
            self._names = [row.strip() for row in infile.readlines()]
        self._layers = [self._model.getLayerNames()[i[0] - 1] for i in self._model.getUnconnectedOutLayers()]
        self._colors = np.random.uniform(0, 255, size=(len(self._names), 3))

    def _download_model(self):
        if not os.path.isfile(self._MAPPER['weights_dir']):
            print('Downloading model.....')
            urlretrieve(self._MAPPER['weights_url'], self._MAPPER['weights_dir'])
            print('Model download complete......')
            
    @property
    def _start_camera(self):
        return cv2.VideoCapture(0)
    
    @property
    def _get_font(self):
        return cv2.FONT_HERSHEY_PLAIN
    
    def _detect(self):
        start = time.time()
        f_count = 0
        cam = self._start_camera
        while True:
            _, frame = cam.read()
            f_count += 1

            blob_from_image = cv2.dnn.blobFromImage(frame, 0.00392, (320, 320), (0, 0, 0), True, crop=False)

            self._model.setInput(blob_from_image)
            outputs = self._model.forward(self._layers)

            name_ids, probabilities, boundings = [], [], []
            height, width, channels = frame.shape
            for output in outputs:
                for detect in output:
                    scores = detect[5:]
                    class_id = np.argmax(scores)
                    probability = scores[class_id]
                    if probability > 0.3:
                        x = int(int(detect[0] * width) - int(detect[2] * width) / 2)
                        y = int(int(detect[1] * height) - int(detect[3] * height) / 2)
                        w, h = int(detect[2] * width), int(detect[3] * height)

                        boundings.append([x, y, w, h])
                        probabilities.append(
                            float(probability))
                        name_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boundings, probabilities, 0.4, 0.6)

            for i in range(len(boundings)):
                if i in indexes:
                    x, y, w, h = boundings[i]
                    label = str(self._names[name_ids[i]])
                    probability = probabilities[i]
                    color = self._colors[name_ids[i]]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, label + " " + str(round(probability, 2)), (x, y + 30), self._get_font, 1, (255, 255, 255),
                                2)

            end_time = time.time() - start
            fps = f_count / end_time
            cv2.putText(frame, "FPS:" + str(round(fps, 2)), (10, 50), self._get_font, 2, (0, 0, 0), 1)

            cv2.imshow("Image", frame)
            key = cv2.waitKey(1)

            if key == 27:  # esc key stops the process
                break;

        cam.release()
        cv2.destroyAllWindows()

    def run(self):
        self._download_model()
        self._detect()

obj = ObjectDetection()
obj.run()