# Based on:
# https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb

import os
import sys

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from colab import cv2_imshow
from drawing import draw_landmarks_on_image

if __name__ == '__main__':
    _REFRESH_RATE_MS = 1
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
            # Capturing a frame from the camera
            success, raw_img = capture.read()
            if not success:
                print("Error: Failed to capture frame.", file=sys.stderr)
                break
            img = mp.Image(mp.ImageFormat.SRGB, data=raw_img)
            # TODO Use detector.detect_for_video instead, see if accuracy improves
            detection_result = detector.detect(img)
            annotated_image = draw_landmarks_on_image(img.numpy_view(), detection_result)
            cv2_imshow('Hand gesture recognition', annotated_image)
            cv2.waitKey(_REFRESH_RATE_MS)
    except KeyboardInterrupt:
        # Exit cleanly if program is interrupted
        pass
    finally:
        # Releasing the camera and closing all OpenCV windows
        capture.release()
        cv2.destroyAllWindows()
