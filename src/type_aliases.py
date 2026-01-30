from typing import TypeAlias

from mediapipe.tasks.python import vision

PoseLandmarker: TypeAlias = vision.PoseLandmarker
HandLandmarker: TypeAlias = vision.HandLandmarker
FaceLandmarker: TypeAlias = vision.FaceLandmarker
HandLandmarkerResult: TypeAlias = vision.HandLandmarkerResult
FaceLandmarkerResult: TypeAlias = vision.FaceLandmarkerResult
PoseLandmarkerResult: TypeAlias = vision.PoseLandmarkerResult
LandmarkerResult: TypeAlias = HandLandmarkerResult \
    | FaceLandmarkerResult \
    | PoseLandmarkerResult
