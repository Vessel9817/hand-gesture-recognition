# Based on:
# https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb

"""MediaPipe solution drawing."""

import cv2
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.components.containers.landmark import \
    NormalizedLandmark

from . import drawing_styles
from .drawing_utils import draw_landmarks
from .type_aliases import (FaceLandmarkerResult, HandLandmarkerResult,
                           LandmarkerResult, PoseLandmarkerResult)

MARGIN = 10 # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

def draw_landmarks_on_image(
    rgb_image: np.ndarray,
    detection_result: LandmarkerResult
) -> None:
    if isinstance(detection_result, vision.HandLandmarkerResult):
        return _draw_hand_landmarks_on_image(rgb_image, detection_result)
    elif isinstance(detection_result, vision.FaceLandmarkerResult):
        return _draw_face_landmarks_on_image(rgb_image, detection_result)
    elif isinstance(detection_result, vision.PoseLandmarkerResult):
        return _draw_body_landmarks_on_image(rgb_image, detection_result)
    raise NotImplementedError('Can only draw hand or face landmarks')

def _draw_hand_landmarks_on_image(
    rgb_image: np.ndarray,
    detection_result: HandLandmarkerResult
) -> None:
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]
        # Draw the hand landmarks.
        hand_landmarks_proto = [
            NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
            for landmark in hand_landmarks
        ]
        draw_landmarks(
            rgb_image,
            hand_landmarks_proto,
            vision.HandLandmarksConnections.HAND_CONNECTIONS,
            drawing_styles.get_default_hand_landmarks_style(),
            drawing_styles.get_default_hand_connections_style())
        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = rgb_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN
        # Draw handedness (left or right hand) on the image.
        cv2.putText(rgb_image, f"{handedness[0].category_name}",
                    (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS,
                    cv2.LINE_AA)

def _draw_face_landmarks_on_image(
    rgb_image: np.ndarray,
    detection_result: FaceLandmarkerResult
) -> None:
    face_landmarks_list = detection_result.face_landmarks
    # Loop through the detected faces to visualize.
    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]
        # Draw the face landmarks.
        face_landmarks_proto = [
            NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
            for landmark in face_landmarks
        ]
        draw_landmarks(
            rgb_image,
            face_landmarks_proto,
            vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
            drawing_styles.get_default_face_mesh_tesselation_style(),
            drawing_styles.get_default_face_mesh_tesselation_style())

def _draw_body_landmarks_on_image(
    rgb_image: np.ndarray,
    detection_result: PoseLandmarkerResult
) -> None:
    body_landmarks_list = detection_result.pose_landmarks
    # Loop through the detected faces to visualize.
    for idx in range(len(body_landmarks_list)):
        body_landmarks = body_landmarks_list[idx]
        # Draw the body landmarks.
        body_landmarks_proto = [
            NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
            for landmark in body_landmarks
        ]
        draw_landmarks(
            rgb_image,
            body_landmarks_proto,
            vision.PoseLandmarksConnections.POSE_LANDMARKS,
            drawing_styles.get_default_body_landmarks_style(),
            drawing_styles.get_default_body_landmarks_style())
