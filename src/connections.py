# Based on:
# https://github.com/google-ai-edge/mediapipe/blob/d1e2e0c7eed0f0ccc237b4e8b78528cd4533ca9e/mediapipe/python/solutions/hands_connections.py

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
"""MediaPipe Hands connections."""

from typing import Iterable, TypeAlias

from mediapipe.tasks.python.vision.face_landmarker import \
    FaceLandmarksConnections
from mediapipe.tasks.python.vision.hand_landmarker import \
    HandLandmarksConnections
from mediapipe.tasks.python.vision.pose_landmarker import \
    PoseLandmarksConnections

Connection: TypeAlias = \
    HandLandmarksConnections.Connection | \
    FaceLandmarksConnections.Connection | \
    PoseLandmarksConnections.Connection

def _as_legacy_connection(c: Connection) -> tuple[int, int]:
    return (c.start, c.end)

def _as_legacy_connections(
    cs: Iterable[Connection]
) -> frozenset[tuple[int, int]]:
    return frozenset().union(map(_as_legacy_connection, cs))

HAND_PALM_CONNECTIONS = _as_legacy_connections(
    HandLandmarksConnections.HAND_PALM_CONNECTIONS)
HAND_THUMB_CONNECTIONS = _as_legacy_connections(
    HandLandmarksConnections.HAND_THUMB_CONNECTIONS)
HAND_INDEX_FINGER_CONNECTIONS = _as_legacy_connections(
    HandLandmarksConnections.HAND_INDEX_FINGER_CONNECTIONS)
HAND_MIDDLE_FINGER_CONNECTIONS = _as_legacy_connections(
    HandLandmarksConnections.HAND_MIDDLE_FINGER_CONNECTIONS)
HAND_RING_FINGER_CONNECTIONS = _as_legacy_connections(
    HandLandmarksConnections.HAND_RING_FINGER_CONNECTIONS)
HAND_PINKY_FINGER_CONNECTIONS = _as_legacy_connections(
    HandLandmarksConnections.HAND_PINKY_FINGER_CONNECTIONS)
HAND_CONNECTIONS = _as_legacy_connections(
    HandLandmarksConnections.HAND_CONNECTIONS)

FACE_LANDMARKS_CONTOURS = _as_legacy_connections(
    FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS)
FACE_LANDMARKS_FACE_OVAL = _as_legacy_connections(
    FaceLandmarksConnections.FACE_LANDMARKS_FACE_OVAL)
FACE_LANDMARKS_LEFT_EYE = _as_legacy_connections(
    FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYE)
FACE_LANDMARKS_LEFT_EYEBROW = _as_legacy_connections(
    FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYEBROW)
FACE_LANDMARKS_LEFT_IRIS = _as_legacy_connections(
    FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS)
FACE_LANDMARKS_LIPS = _as_legacy_connections(
    FaceLandmarksConnections.FACE_LANDMARKS_LIPS)
FACE_LANDMARKS_RIGHT_EYE = _as_legacy_connections(
    FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYE)
FACE_LANDMARKS_RIGHT_EYEBROW = _as_legacy_connections(
    FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYEBROW)
FACE_LANDMARKS_RIGHT_IRIS = _as_legacy_connections(
    FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS)
FACE_LANDMARKS_TESSELATION = _as_legacy_connections(
    FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION)
