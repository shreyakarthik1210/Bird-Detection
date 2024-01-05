# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Main script to run the object detection routine."""
import argparse
import sys
import time
import numpy as np

import cv2
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
import utils
from tflite_runtime.interpreter import Interpreter

import cv2
import numpy
from flask import Flask, render_template, Response, stream_with_context, request

app = Flask('__name__')

# specify paths to local file assets
path_to_labels = "birds-label.txt"
path_to_model = "birds-model.tflite"

def load_labels():
    """ load labels for the ML model from the file specified """
    with open(path_to_labels, 'r') as f:
        return {i: line.strip() for i, line in enumerate(f.readlines())}
def set_input_tensor(interpreter, image):
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image
def classify_image(interpreter, image, top_k=1):
    """ return a sorted array of classification results """
    set_input_tensor(interpreter, image)
    interpreter.invoke()
    output_details = interpreter.get_output_details()[0]
    output = np.squeeze(interpreter.get_tensor(output_details['index']))

    # if model is quantized (uint8 data), then dequantize the results
    if output_details['dtype'] == np.uint8:
        scale, zero_point = output_details['quantization']
        output = scale * (output - zero_point)

    ordered = np.argpartition(-output, top_k)
    return [(i, output[i]) for i in ordered[:top_k]]

def run(model: str, camera_id: int, width: int, height: int, num_threads: int,
        enable_edgetpu: bool) -> None:
  """Continuously run inference on images acquired from the camera.

  Args:
    model: Name of the TFLite object detection model.
    camera_id: The camera id to be passed to OpenCV.
    width: The width of the frame captured from the camera.
    height: The height of the frame captured from the camera.
    num_threads: The number of CPU threads to run the model.
    enable_edgetpu: True/False whether the model is a EdgeTPU model.
  """

  # Variables to calculate FPS
  counter, fps = 0, 0
  start_time = time.time()

  # Start capturing video input from the camera
  cap = cv2.VideoCapture(camera_id)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

  # Visualization parameters
  row_size = 20  # pixels
  left_margin = 24  # pixels
  text_color = (0, 0, 255)  # red
  font_size = 1
  font_thickness = 1
  fps_avg_frame_count = 10



  # Initialize the object detection model
  base_options = core.BaseOptions(
      file_name=model, use_coral=enable_edgetpu, num_threads=num_threads)
  detection_options = processor.DetectionOptions(
      max_results=3, score_threshold=0.3)
  options = vision.ObjectDetectorOptions(
      base_options=base_options, detection_options=detection_options)
  detector = vision.ObjectDetector.create_from_options(options)

  # Continuously capture images from the camera and run inference
  while cap.isOpened():
        success, image = cap.read()
        if not success:
          sys.exit(
              'ERROR: Unable to read from webcam. Please verify your webcam settings.'
          )
        counter += 1
        image = cv2.flip(image, 1)
        labels = load_labels()
        interpreter = Interpreter(path_to_model)
        interpreter.allocate_tensors()
        _, height, width, _ = interpreter.get_input_details()[0]['shape']

        # Convert the image from BGR to RGB as required by the TFLite model.
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Create a TensorImage object from the RGB image.
        input_tensor = vision.TensorImage.create_from_array(rgb_image)

        # Run object detection estimation using the model.
        #detection_result = detector.detect(input_tensor)
        results = classify_image(interpreter, image)

        # Draw keypoints and edges on input image
        image = utils.visualize(image, detection_result)

        # Calculate the FPS
        if counter % fps_avg_frame_count == 0:
          end_time = time.time()
          fps = fps_avg_frame_count / (end_time - start_time)
          start_time = time.time()

        # Show the FPS
        fps_text = 'FPS = {:.1f}'.format(fps)
        text_location = (left_margin, row_size)
        cv2.putText(image, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                    font_size, text_color, font_thickness)

        # Stop the program if the ESC key is pressed.
        if cv2.waitKey(1) == 27:
          break
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        ret, buffer = cv2.imencode('.jpeg', image)
        image = buffer.tobytes()
        yield (b' --frame\r\n' b'Content-type: imgae/jpeg\r\n\r\n' + image + b'\r\n')

        #cv2.imshow('object_detector', image)

      #cap.release()
      #cv2.destroyAllWindows()


def main():
      parser = argparse.ArgumentParser(
          formatter_class=argparse.ArgumentDefaultsHelpFormatter)
      parser.add_argument(
          '--model',
          help='Path of the object detection model.',
          required=False,
          default='efficientdet_lite0.tflite')
      parser.add_argument(
          '--cameraId', help='Id of camera.', required=False, type=int, default=0)
      parser.add_argument(
          '--frameWidth',
          help='Width of frame to capture from camera.',
          required=False,
          type=int,
          default=640)
      parser.add_argument(
          '--frameHeight',
          help='Height of frame to capture from camera.',
          required=False,
          type=int,
          default=480)
      parser.add_argument(
          '--numThreads',
          help='Number of CPU threads to run the model.',
          required=False,
          type=int,
          default=4)
      parser.add_argument(
          '--enableEdgeTPU',
          help='Whether to run the model on EdgeTPU.',
          action='store_true',
          required=False,
          default=False)
      args = parser.parse_args()

      run(args.model, int(args.cameraId), args.frameWidth, args.frameHeight,
          int(args.numThreads), bool(args.enableEdgeTPU))

@app.route('/')
def camera():
    return render_template('camera.html')


@app.route('/video_feed')
def video_feed():
    return Response(video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
     app.run(host='0.0.0.0' ,port='5000',debug=True)
