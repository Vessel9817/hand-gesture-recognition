# Original source:
# https://github.com/google-ai-edge/mediapipe/blob/d1e2e0c7eed0f0ccc237b4e8b78528cd4533ca9e/mediapipe/python/solutions/drawing_styles.py

# Copyright 2021 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MediaPipe solution drawing styles."""

from typing import Mapping, Tuple

from mediapipe.tasks.python.vision.hand_landmarker import HandLandmark

import connections
from drawing_utils import DrawingSpec

_RADIUS = 5
_RED = (48, 48, 255)
_GREEN = (48, 255, 48)
_BLUE = (192, 101, 21)
_YELLOW = (0, 204, 255)
_GRAY = (128, 128, 128)
_PURPLE = (128, 64, 128)
_PEACH = (180, 229, 255)
_WHITE = (224, 224, 224)
_TURQUOISE = (192, 255, 48)
_MAGENTA = (192, 48, 255)

# Hands
_THICKNESS_WRIST_MCP = 3
_THICKNESS_FINGER = 2
_THICKNESS_DOT = -1

# Hand landmarks
_PALM_LANDMARKS = (
    HandLandmark.WRIST,
    HandLandmark.THUMB_CMC,
    HandLandmark.INDEX_FINGER_MCP,
    HandLandmark.MIDDLE_FINGER_MCP,
    HandLandmark.RING_FINGER_MCP,
    HandLandmark.PINKY_MCP
)
_THUMP_LANDMARKS = (
    HandLandmark.THUMB_MCP,
    HandLandmark.THUMB_IP,
    HandLandmark.THUMB_TIP
)
_INDEX_FINGER_LANDMARKS = (
    HandLandmark.INDEX_FINGER_PIP,
    HandLandmark.INDEX_FINGER_DIP,
    HandLandmark.INDEX_FINGER_TIP
)
_MIDDLE_FINGER_LANDMARKS = (
    HandLandmark.MIDDLE_FINGER_PIP,
    HandLandmark.MIDDLE_FINGER_DIP,
    HandLandmark.MIDDLE_FINGER_TIP
)
_RING_FINGER_LANDMARKS = (
    HandLandmark.RING_FINGER_PIP,
    HandLandmark.RING_FINGER_DIP,
    HandLandmark.RING_FINGER_TIP
)
_PINKY_FINGER_LANDMARKS = (
    HandLandmark.PINKY_PIP,
    HandLandmark.PINKY_DIP,
    HandLandmark.PINKY_TIP
)
_HAND_LANDMARK_STYLE = {
    _PALM_LANDMARKS: DrawingSpec(
        color=_RED, thickness=_THICKNESS_DOT, circle_radius=_RADIUS),
    _THUMP_LANDMARKS: DrawingSpec(
        color=_PEACH, thickness=_THICKNESS_DOT, circle_radius=_RADIUS),
    _INDEX_FINGER_LANDMARKS: DrawingSpec(
        color=_PURPLE, thickness=_THICKNESS_DOT, circle_radius=_RADIUS),
    _MIDDLE_FINGER_LANDMARKS: DrawingSpec(
        color=_YELLOW, thickness=_THICKNESS_DOT, circle_radius=_RADIUS),
    _RING_FINGER_LANDMARKS: DrawingSpec(
        color=_GREEN, thickness=_THICKNESS_DOT, circle_radius=_RADIUS),
    _PINKY_FINGER_LANDMARKS: DrawingSpec(
        color=_BLUE, thickness=_THICKNESS_DOT, circle_radius=_RADIUS),
}

# Hands connections
_HAND_CONNECTION_STYLE = {
    connections.HAND_PALM_CONNECTIONS:
        DrawingSpec(color=_GRAY, thickness=_THICKNESS_WRIST_MCP),
    connections.HAND_THUMB_CONNECTIONS:
        DrawingSpec(color=_PEACH, thickness=_THICKNESS_FINGER),
    connections.HAND_INDEX_FINGER_CONNECTIONS:
        DrawingSpec(color=_PURPLE, thickness=_THICKNESS_FINGER),
    connections.HAND_MIDDLE_FINGER_CONNECTIONS:
        DrawingSpec(color=_YELLOW, thickness=_THICKNESS_FINGER),
    connections.HAND_RING_FINGER_CONNECTIONS:
        DrawingSpec(color=_GREEN, thickness=_THICKNESS_FINGER),
    connections.HAND_PINKY_FINGER_CONNECTIONS:
        DrawingSpec(color=_BLUE, thickness=_THICKNESS_FINGER)
}

# Face connections
_THICKNESS_TESSELATION = 1
_THICKNESS_CONTOURS = 2
_FACE_CONTOURS_CONNECTION_STYLE_0 = {
    connections.FACE_LANDMARKS_LIPS:
        DrawingSpec(color=_WHITE, thickness=_THICKNESS_CONTOURS),
    connections.FACE_LANDMARKS_LEFT_EYE:
        DrawingSpec(color=_GREEN, thickness=_THICKNESS_CONTOURS),
    connections.FACE_LANDMARKS_LEFT_EYEBROW:
        DrawingSpec(color=_GREEN, thickness=_THICKNESS_CONTOURS),
    connections.FACE_LANDMARKS_RIGHT_EYE:
        DrawingSpec(color=_RED, thickness=_THICKNESS_CONTOURS),
    connections.FACE_LANDMARKS_RIGHT_EYEBROW:
        DrawingSpec(color=_RED, thickness=_THICKNESS_CONTOURS),
    connections.FACE_LANDMARKS_FACE_OVAL:
        DrawingSpec(color=_WHITE, thickness=_THICKNESS_CONTOURS)
}
_FACE_CONTOURS_CONNECTION_STYLE_1 = {
    connections.FACE_LANDMARKS_LIPS:
        DrawingSpec(color=_BLUE, thickness=_THICKNESS_CONTOURS),
    connections.FACE_LANDMARKS_LEFT_EYE:
        DrawingSpec(color=_TURQUOISE, thickness=_THICKNESS_CONTOURS),
    connections.FACE_LANDMARKS_LEFT_EYEBROW:
        DrawingSpec(color=_GREEN, thickness=_THICKNESS_CONTOURS),
    connections.FACE_LANDMARKS_RIGHT_EYE:
        DrawingSpec(color=_MAGENTA, thickness=_THICKNESS_CONTOURS),
    connections.FACE_LANDMARKS_RIGHT_EYEBROW:
        DrawingSpec(color=_RED, thickness=_THICKNESS_CONTOURS),
    connections.FACE_LANDMARKS_FACE_OVAL:
        DrawingSpec(color=_WHITE, thickness=_THICKNESS_CONTOURS),
}

# Pose
_THICKNESS_BODY_LANDMARKS = 2

def get_default_hand_connections_style(
) -> Mapping[Tuple[int, int], DrawingSpec]:
  """Returns the default hand connections drawing style.

  Returns:
      A mapping from each hand connection to its default drawing spec.
  """
  hand_connection_style = {}
  for k, v in _HAND_CONNECTION_STYLE.items():
        for connection in k:
            hand_connection_style[connection] = v
  return hand_connection_style

def get_default_hand_landmarks_style(
) -> Mapping[int, DrawingSpec]:
  """Returns the default hand landmarks drawing style.

  Returns:
      A mapping from each hand landmark to its default drawing spec.
  """
  hand_landmark_style = {}
  for k, v in _HAND_LANDMARK_STYLE.items():
        for landmark in k:
            hand_landmark_style[landmark] = v
  return hand_landmark_style

def get_default_face_mesh_contours_style(
    i: int = 0
) -> Mapping[Tuple[int, int], DrawingSpec]:
    """Returns the default face mesh contours drawing style.

    Args:
        i: The id for default style. Currently there are two default styles.

    Returns:
        A mapping from each face mesh contours connection to its default drawing
        spec.
    """
    default_style = (
        _FACE_CONTOURS_CONNECTION_STYLE_1
        if i == 1
        else _FACE_CONTOURS_CONNECTION_STYLE_0
    )
    face_mesh_contours_connection_style = {}
    for k, v in default_style.items():
        for connection in k:
            face_mesh_contours_connection_style[connection] = v
    return face_mesh_contours_connection_style

def get_default_face_mesh_tesselation_style(
) -> DrawingSpec:
    """Returns the default face mesh tesselation drawing style.

    Returns:
        A DrawingSpec.
    """
    return DrawingSpec(color=_GRAY, thickness=_THICKNESS_TESSELATION)

def get_default_face_mesh_iris_connections_style(
) -> Mapping[Tuple[int, int], DrawingSpec]:
    """Returns the default face mesh iris connections drawing style.

    Returns:
        A mapping from each iris connection to its default drawing spec.
    """
    face_mesh_iris_connections_style = {}
    left_spec = DrawingSpec(color=_GREEN, thickness=_THICKNESS_CONTOURS)
    for connection in connections.FACE_LANDMARKS_LEFT_IRIS:
        face_mesh_iris_connections_style[connection] = left_spec
    right_spec = DrawingSpec(color=_RED, thickness=_THICKNESS_CONTOURS)
    for connection in connections.FACE_LANDMARKS_RIGHT_IRIS:
        face_mesh_iris_connections_style[connection] = right_spec
    return face_mesh_iris_connections_style

def get_default_body_landmarks_style(
) -> DrawingSpec:
    """Returns the default pose landmarks drawing style.

    Returns:
        A mapping from each pose landmark to its default drawing spec.
    """
    return DrawingSpec(
        color=_RED, thickness=_THICKNESS_BODY_LANDMARKS)
