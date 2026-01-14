# Original source:
# https://github.com/shwet369/hand-gesture-recognition/tree/b276b2b150918c22d2e701f0d9c85954ea323bdf

# Based on:
# https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb

# MIT License
#
# Copyright (c) 2024 Shweta 
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from drawing import draw_landmarks_on_image

if __name__ == '__main__':
    # Open the default camera
    capture = cv2.VideoCapture(0)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
    model_asset_path = os.path.join(script_dir, 'hand_landmarker.task')
    base_options = python.BaseOptions(model_asset_path=model_asset_path)
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
    detector = vision.HandLandmarker.create_from_options(options)
    print('Press Ctrl+C to quit')
    try:
        while True:
            # Capture a frame from the camera
            success, raw_img = capture.read()
            if not success:
                print("Error: Failed to capture frame.")
                break
            img = mp.Image(mp.ImageFormat.SRGB, data=raw_img)
            # TODO Use detector.detect_for_video instead, see if accuracy improves
            detection_result = detector.detect(img)
            annotated_image = draw_landmarks_on_image(img.numpy_view(), detection_result)
            cv2.imshow('Hand gesture recognition', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
    except KeyboardInterrupt:
        # Exit cleanly if loop is interrupted
        pass
    finally:
        # Release the camera and close all OpenCV windows
        capture.release()
        cv2.destroyAllWindows()
