# Based on:
# https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb

"""MediaPipe demo."""

import os
import sys
import time

import cv2
import mediapipe as mp
import numpy as np

from .colab import cv2_imshow
from .detectors import BodyDetector, FaceDetector, HandDetector
from .drawing import draw_landmarks_on_image

_CAMERA_INDEX = 0
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_HAND_MODEL_ASSET_PATH = os.path.join(_SCRIPT_DIR, 'hand_landmarker.task')
_FACE_MODEL_ASSET_PATH = os.path.join(_SCRIPT_DIR, 'face_landmarker.task')
_BODY_MODEL_ASSET_PATH = os.path.join(_SCRIPT_DIR, 'pose_landmarker.task')
_REFRESH_RATE_MS = 1
_EXIT_KEY = ord('q')
_WINDOW_TITLE = 'Hand, face and body recognition'

if __name__ == '__main__':
    # Opening the default camera
    capture = cv2.VideoCapture(_CAMERA_INDEX)
    # Configuring models
    hand_detector = HandDetector(_HAND_MODEL_ASSET_PATH)
    face_detector = FaceDetector(_FACE_MODEL_ASSET_PATH)
    body_detector = BodyDetector(_BODY_MODEL_ASSET_PATH)
    start_time_s = time.time()
    print(f'+=================+\n| Press {chr(_EXIT_KEY)} to quit |\n+=================+')
    try:
        while True:
            # Capturing a frame from the camera
            success, raw_img = capture.read()
            if not success:
                print("Error: Failed to capture frame.", file=sys.stderr)
                break
            # Processing image asynchronously
            img = mp.Image(mp.ImageFormat.SRGB, data=raw_img)
            passed_time_ms = int(1000 * (time.time() - start_time_s))
            body_detector.detect_async(img, passed_time_ms)
            face_detector.detect_async(img, passed_time_ms)
            hand_detector.detect_async(img, passed_time_ms)
            # Drawing latest results
            annotated_image = np.copy(img.numpy_view())
            if body_detector.result is not None:
                draw_landmarks_on_image(annotated_image, body_detector.result)
            if face_detector.result is not None:
                draw_landmarks_on_image(annotated_image, face_detector.result)
            if hand_detector.result is not None:
                draw_landmarks_on_image(annotated_image, hand_detector.result)
            # Rendering window
            cv2_imshow(_WINDOW_TITLE, annotated_image)
            if cv2.waitKey(_REFRESH_RATE_MS) & 0xFF == _EXIT_KEY:
                # Destroy window ASAP while we still have focus,
                # otherwise later destruction might hang
                cv2.destroyWindow(_WINDOW_TITLE)
                break
    except KeyboardInterrupt:
        # Exit cleanly if program is interrupted
        pass
    finally:
        # Releasing resources
        capture.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1) # https://stackoverflow.com/a/13850341
