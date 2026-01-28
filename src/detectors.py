import abc
from typing import Generic, Optional, TypeVar

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

T = TypeVar('T')

class Detector(Generic[T]):
    @abc.abstractmethod
    def detect_async(
        self,
        img: mp.Image,
        timestamp: int
    ) -> None:
        pass

    @property
    @abc.abstractmethod
    def result(self) -> T:
        pass

class HandDetector(Detector[Optional[vision.HandLandmarkerResult]]):
    def __init__(
        self,
        model_asset_path: str,
        max_hands: int = 2
    ) -> None:
        self._result: Optional[vision.HandLandmarkerResult] = None
        base_options = python.BaseOptions(model_asset_path=model_asset_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=max_hands,
            running_mode=vision.RunningMode.LIVE_STREAM,
            result_callback=self.__set_hands
        )
        self._detector = vision.HandLandmarker.create_from_options(options)

    def __set_hands(
        self,
        detection_result: vision.HandLandmarkerResult,
        image: mp.Image,
        timestamp: int
    ) -> None:
        '''The hand landmark detection callback.'''
        self._result = detection_result

    def detect_async(
        self,
        img: mp.Image,
        timestamp: int
    ) -> None:
        self._detector.detect_async(img, timestamp)

    @property
    def result(self) -> Optional[vision.HandLandmarkerResult]:
        return self._result

class FaceDetector(Detector[Optional[vision.FaceLandmarkerResult]]):
    def __init__(
        self,
        model_asset_path: str,
        max_faces: int = 1
    ) -> None:
        self._result: Optional[vision.FaceLandmarkerResult] = None
        base_options = python.BaseOptions(model_asset_path=model_asset_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=max_faces,
            running_mode=vision.RunningMode.LIVE_STREAM,
            result_callback=self.__set_faces
        )
        self._detector = vision.FaceLandmarker.create_from_options(options)

    def __set_faces(
        self,
        detection_result: vision.FaceLandmarkerResult,
        image: mp.Image,
        timestamp: int
    ) -> None:
        '''The face landmark detection callback.'''
        self._result = detection_result

    def detect_async(
        self,
        img: mp.Image,
        timestamp: int
    ) -> None:
        self._detector.detect_async(img, timestamp)

    @property
    def result(self) -> Optional[vision.FaceLandmarkerResult]:
        return self._result

class BodyDetector(Detector[Optional[vision.PoseLandmarkerResult]]):
    def __init__(
        self,
        model_asset_path: str,
        max_bodies: int = 1
    ) -> None:
        self._result: Optional[vision.PoseLandmarkerResult] = None
        base_options = python.BaseOptions(model_asset_path=model_asset_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            num_poses=max_bodies,
            running_mode=vision.RunningMode.LIVE_STREAM,
            result_callback=self.__set_bodies
        )
        self._detector = vision.PoseLandmarker.create_from_options(options)

    def __set_bodies(
        self,
        detection_result: vision.PoseLandmarkerResult,
        image: mp.Image,
        timestamp: int
    ) -> None:
        '''The body landmark detection callback.'''
        self._result = detection_result

    def detect_async(
        self,
        img: mp.Image,
        timestamp: int
    ) -> None:
        self._detector.detect_async(img, timestamp)

    @property
    def result(self) -> Optional[vision.PoseLandmarkerResult]:
        return self._result
