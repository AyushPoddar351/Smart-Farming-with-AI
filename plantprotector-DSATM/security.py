import sys
import cv2
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore, uic
import pygame
import time

class LivestockMotionDemo(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        # Load the UI file
        uic.loadUi("C:\\data\\dsatm\\4th sem\\cypherquest\\Smart Farming with AI\\plantprotector-DSATM\\design1.ui", self)

        # Set up UI elements
        self.showFullScreen()

        # Initialize camera
        self.capture = cv2.VideoCapture(0)
        if not self.capture.isOpened():
            print("Error: Could not open camera.")
            sys.exit()

        # Initialize variables
        self.currentFrame = np.array([])
        self.firstFrame = None
        self.threat_timestamp = None  # Timestamp for the last detected threat
        self.threat_display_duration = 5  # Duration to display the threat message in seconds

        # Connect UI buttons to methods
        self.start_button.clicked.connect(self.start_stream)
        self.stop_button.clicked.connect(self.stop_stream)

        # Initialize pygame mixer for alarm sound
        pygame.mixer.init()
        self.alarm_sound = "C:\\data\\dsatm\\4th sem\\cypherquest\\Smart Farming with AI\\plantprotector-DSATM\\siren.mp3"  # Update the path to your alarm sound file
        pygame.mixer.music.load(self.alarm_sound)

        # Load YOLO model
        self.load_yolo_model()

        # Show the main window
        self.show()

    def load_yolo_model(self):
        model_cfg_path = "C:\\data\\dsatm\\4th sem\\cypherquest\\Smart Farming with AI\\plantprotector-DSATM\\yolov3-tiny.cfg"  # Update the path to your cfg file
        model_weights_path = "C:\\data\\dsatm\\4th sem\\cypherquest\\Smart Farming with AI\\plantprotector-DSATM\\yolov3-tiny.weights"  # Update the path to your weights file
        self.net = cv2.dnn.readNet(model_weights_path, model_cfg_path)
        with open("C:\\data\\dsatm\\4th sem\\cypherquest\\Smart Farming with AI\\plantprotector-DSATM\\coco.names", "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))

        # Filter only animal classes and humans
        self.animal_classes = ["bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"]
        self.human_classes = ["person"]

    def start_stream(self):
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self.play)
        self._timer.start(100)  # Adjusted frame rate (100 ms ~ 10 fps)

    def stop_stream(self):
        self._timer.stop()
        self.capture.release()
        cv2.destroyAllWindows()
        self.close()

    def play(self):
        try:
            ret, frame = self.capture.read()
            if not ret:
                print("Error: Could not read frame from camera.")
                return

            height, width, channels = frame.shape
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            self.net.setInput(blob)
            outs = self.net.forward(self.output_layers)

            boxes = []
            confidences = []
            class_ids = []
            threat_found = False

            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.3 and (self.classes[class_id] in self.animal_classes or self.classes[class_id] in self.human_classes):  # Lowered threshold to 0.3
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        # Rectangle coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

                        # Check if the detected object is an animal
                        if self.classes[class_id] in self.animal_classes:
                            threat_found = True

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)

            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(self.classes[class_ids[i]])
                    color = self.colors[class_ids[i]]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1 / 2, color, 2)

            # Display "threat found" message if an animal is detected
            current_time = time.time()
            if threat_found:
                self.threat_timestamp = current_time
                if not pygame.mixer.music.get_busy():
                    pygame.mixer.music.play(-1)  # Play the alarm in a loop
                self.threat_label.setText("THREAT FOUND")
            else:
                self.threat_label.setText("")
                pygame.mixer.music.stop()

            # Convert frame to RGB format for Qt
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
            self.video_label.setPixmap(QtGui.QPixmap.fromImage(convert_to_Qt_format))
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = LivestockMotionDemo()
    sys.exit(app.exec_())