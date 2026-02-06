# Based on:
# https://github.com/google-ai-edge/mediapipe/blob/d1e2e0c7eed0f0ccc237b4e8b78528cd4533ca9e/mediapipe/python/solutions/drawing_utils.py

"""MediaPipe solution drawing utils."""

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Set

import cv2
import numpy as np
from mediapipe.tasks.python.components.containers.landmark import \
    NormalizedLandmark

from .connections import Connection

_PRESENCE_THRESHOLD = 0.5
_VISIBILITY_THRESHOLD = 0.5
_BGR_CHANNELS = 3

WHITE_COLOR = (224, 224, 224)
BLACK_COLOR = (0, 0, 0)
RED_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 128, 0)
BLUE_COLOR = (255, 0, 0)

@dataclass
class DrawingSpec:
    # Color for drawing the annotation. Default to the white color.
    color: tuple[int, int, int] = WHITE_COLOR
    # Thickness for drawing the annotation. Default to 2 pixels.
    thickness: int = 2
    # Circle radius. Default to 2 pixels.
    circle_radius: int = 2

def _is_valid_normalized_value(value: float) -> bool:
    '''Converts normalized value pair to pixel coordinates.'''
    return (value > 0 or math.isclose(0, value, abs_tol=1e-9)) and (
            value < 1 or math.isclose(1, value))

def _normalized_to_pixel_coordinates(
    normalized_x: float,
    normalized_y: float,
    image_width: int,
    image_height: int
) -> tuple[int, int]:
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px

def draw_landmarks(
    image: np.ndarray,
    landmark_list: List[NormalizedLandmark],
    connections: Optional[Iterable[Connection]] = None,
    landmark_drawing_spec: Optional[
        DrawingSpec | Mapping[int, DrawingSpec]
    ] = DrawingSpec(color=RED_COLOR),
    connection_drawing_spec: DrawingSpec | Mapping[tuple[int, int], DrawingSpec] = DrawingSpec(),
    is_drawing_landmarks: bool = True,
) -> None:
    """
    Draws the landmarks and the connections on the image.

    Args:
        image: A three channel BGR image represented as numpy ndarray.
        landmark_list: A normalized landmark list proto message to be annotated
            on the image.
        connections: A list of landmark index tuples that specifies how
            landmarks to be connected in the drawing.
        landmark_drawing_spec: Either a DrawingSpec object or a mapping from
            hand landmarks to the DrawingSpecs that specifies the landmarks'
            drawing settings such as color, line thickness, and circle radius.
            If this argument is explicitly set to None, no landmarks will be
            drawn.
        connection_drawing_spec: Either a DrawingSpec object or a mapping from
            hand connections to the DrawingSpecs that specifies the connections'
            drawing settings such as color and line thickness. If this argument
            is explicitly set to None, no landmark connections will be drawn.
        is_drawing_landmarks: Whether to draw landmarks. If set false, skip
            drawing landmarks, only contours will be drawed.

    Raises:
        ValueError: If one of the followings:
            a) If the input image is not three channel BGR.
            b) If any connetions contain invalid landmark index.
  """
    if not landmark_list:
        return
    if image.shape[2] != _BGR_CHANNELS:
        raise ValueError(f'Input image must contain {_BGR_CHANNELS} channel bgr data.')
    image_rows, image_cols, _ = image.shape
    idx_to_coordinates: Dict[int, tuple[int, int]] = {}
    oob_landmark_idx: Set[int] = set()
    for idx, landmark in enumerate(landmark_list):
        if ((landmark.visibility is not None and
                landmark.visibility < _VISIBILITY_THRESHOLD) or
                (landmark.presence is not None and
                landmark.presence < _PRESENCE_THRESHOLD) or
                landmark.x is None or
                landmark.y is None):
            continue
        valid_x = _is_valid_normalized_value(landmark.x)
        valid_y = _is_valid_normalized_value(landmark.y)
        idx_to_coordinates[idx] = _normalized_to_pixel_coordinates(
                landmark.x, landmark.y, image_cols, image_rows)
        if not (valid_x and valid_y):
            # Don't draw landmark if it's out of image bounds
            oob_landmark_idx.add(idx)
    if connections:
        visible_connections = filter(lambda c: not (c.start in oob_landmark_idx and c.end in oob_landmark_idx), connections)
        _draw_connections(
            image,
            landmark_list,
            visible_connections,
            connection_drawing_spec,
            idx_to_coordinates)
    # Draws landmark points after finishing the connection lines, which is
    # aesthetically better.
    for idx in oob_landmark_idx:
        del idx_to_coordinates[idx]
    if is_drawing_landmarks and landmark_drawing_spec:
        _draw_landmarks(image, landmark_drawing_spec, idx_to_coordinates)

def _draw_connections(
    image: np.ndarray,
    landmark_list: List[NormalizedLandmark],
    connections: Iterable[Connection],
    connection_drawing_spec: DrawingSpec | Mapping[tuple[int, int], DrawingSpec],
    idx_to_coordinates: Mapping[int, tuple[int, int]]
) -> None:
    '''Draws the connections between landmark points.'''
    num_landmarks = len(landmark_list)
    for connection in connections:
        start_idx, end_idx = connection.start, connection.end
        if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
            raise ValueError(f'Landmark index is out of range. Invalid connection '
                            f'from landmark #{start_idx} to landmark #{end_idx}.')
        if start_idx in idx_to_coordinates and end_idx in idx_to_coordinates:
            if isinstance(connection_drawing_spec, Mapping):
                drawing_spec = connection_drawing_spec[(start_idx, end_idx)]
            else:
                drawing_spec = connection_drawing_spec
            cv2.line(image, idx_to_coordinates[start_idx],
                    idx_to_coordinates[end_idx], drawing_spec.color,
                    drawing_spec.thickness)

def _draw_landmarks(
    image: np.ndarray,
    landmark_drawing_spec: DrawingSpec | Mapping[int, DrawingSpec],
    idx_to_coordinates: Mapping[int, tuple[int, int]]
) -> None:
    '''Draws landmark points on the image.'''
    for idx, landmark_px in idx_to_coordinates.items():
        drawing_spec = landmark_drawing_spec[idx] if isinstance(
            landmark_drawing_spec, Mapping) else landmark_drawing_spec
        # White circle border
        circle_border_radius = max(drawing_spec.circle_radius + 1,
                                    int(drawing_spec.circle_radius * 1.2))
        cv2.circle(image, landmark_px, circle_border_radius, WHITE_COLOR,
                    drawing_spec.thickness)
        # Fill color into the circle
        cv2.circle(image, landmark_px, drawing_spec.circle_radius,
                    drawing_spec.color, drawing_spec.thickness)
