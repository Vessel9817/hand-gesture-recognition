# Based on:
# https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb

import os
import sys
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from typing import Any

from colab import cv2_imshow
from drawing import draw_landmarks_on_image

if __name__ == '__main__':
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    _MODEL_ASSET_PATH = os.path.join(_SCRIPT_DIR, 'hand_landmarker.task')
    _MIN_INPUT_DELAY_MS = 1
    _EXIT_KEY = ord('q')
    _WINDOW_TITLE = 'Hand gesture recognition'
    exit_loop = False
    def _display_annotated_image(
        detection_result: vision.HandLandmarkerResult,
        image: Any, # vision.core.Image
        timestamp: int
    ):
        global exit_loop
        if not exit_loop:
            annotated_image = draw_landmarks_on_image(img.numpy_view(), detection_result)
            cv2_imshow(_WINDOW_TITLE, annotated_image)
            if cv2.waitKey(_MIN_INPUT_DELAY_MS) & 0xFF == _EXIT_KEY:
                # Destroy window ASAP while we still have focus,
                # otherwise later destruction might hang
                cv2.destroyWindow(_WINDOW_TITLE)
                exit_loop = True
    # Opening the default camera
    capture = cv2.VideoCapture(0)
    # https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
    base_options = python.BaseOptions(model_asset_path=_MODEL_ASSET_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        running_mode=vision.RunningMode.LIVE_STREAM,
        result_callback=_display_annotated_image
    )
    detector = vision.HandLandmarker.create_from_options(options)
    start_time_s = time.time()
    print(f'+=================+\n| Press {chr(_EXIT_KEY)} to quit |\n+=================+')
    try:
        while not exit_loop:
            # Capturing a frame from the camera
            success, raw_img = capture.read()
            if not success:
                print("Error: Failed to capture frame.", file=sys.stderr)
                break
            # Rendering processed image
            img = mp.Image(mp.ImageFormat.SRGB, data=raw_img)
            passed_time_ms = int(1000 * (time.time() - start_time_s))
            detector.detect_async(img, passed_time_ms)
    except KeyboardInterrupt:
        # Exit cleanly if program is interrupted
        pass
    finally:
        # Releasing resources
        capture.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1) # https://stackoverflow.com/a/13850341
