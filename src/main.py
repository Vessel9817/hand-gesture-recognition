# Based on:
# https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb

import os
import sys
import time
from typing import Optional

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from colab import cv2_imshow
from drawing import draw_landmarks_on_image

if __name__ == '__main__':
    _CAMERA_INDEX = 0
    _MAX_HANDS = 2
    _MAX_FACES = 1
    _MAX_BODIES = 1
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    _HAND_MODEL_ASSET_PATH = os.path.join(_SCRIPT_DIR, 'hand_landmarker.task')
    _FACE_MODEL_ASSET_PATH = os.path.join(_SCRIPT_DIR, 'face_landmarker.task')
    _BODY_MODEL_ASSET_PATH = os.path.join(_SCRIPT_DIR, 'pose_landmarker.task')
    _REFRESH_RATE_MS = 1
    _EXIT_KEY = ord('q')
    _WINDOW_TITLE = 'Hand, face and body recognition'
    hands: Optional[vision.HandLandmarkerResult] = None
    face: Optional[vision.FaceLandmarkerResult] = None
    body: Optional[vision.PoseLandmarkerResult] = None
    def _set_hands(
        detection_result: vision.HandLandmarkerResult,
        image: mp.Image,
        timestamp: int
    ):
        global hands
        hands = detection_result
    def _set_face(
        detection_result: vision.FaceLandmarkerResult,
        image: mp.Image,
        timestamp: int
    ):
        global face
        face = detection_result
    def _set_body(
        detection_result: vision.PoseLandmarkerResult,
        image: mp.Image,
        timestamp: int
    ):
        global body
        body = detection_result
    # Opening the default camera
    capture = cv2.VideoCapture(_CAMERA_INDEX)
    # Configuring models
    base_hand_options = python.BaseOptions(model_asset_path=_HAND_MODEL_ASSET_PATH)
    base_face_options = python.BaseOptions(model_asset_path=_FACE_MODEL_ASSET_PATH)
    base_body_options = python.BaseOptions(model_asset_path=_BODY_MODEL_ASSET_PATH)
    hand_options = vision.HandLandmarkerOptions(
        base_options=base_hand_options,
        num_hands=_MAX_HANDS,
        running_mode=vision.RunningMode.LIVE_STREAM,
        result_callback=_set_hands
    )
    face_options = vision.FaceLandmarkerOptions(
        base_options=base_face_options,
        num_faces=_MAX_FACES,
        running_mode=vision.RunningMode.LIVE_STREAM,
        result_callback=_set_face
    )
    body_options = vision.PoseLandmarkerOptions(
        base_options=base_body_options,
        num_poses=_MAX_BODIES,
        running_mode=vision.RunningMode.LIVE_STREAM,
        result_callback=_set_body
    )
    hand_detector = vision.HandLandmarker.create_from_options(hand_options)
    face_detector = vision.FaceLandmarker.create_from_options(face_options)
    body_detector = vision.PoseLandmarker.create_from_options(body_options)
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
            annotated_image = img.numpy_view()
            annotated_image = draw_landmarks_on_image(annotated_image, body) \
                if body is not None else annotated_image
            annotated_image = draw_landmarks_on_image(annotated_image, face) \
                if face is not None else annotated_image
            annotated_image = draw_landmarks_on_image(annotated_image, hands) \
                if hands is not None else annotated_image
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
