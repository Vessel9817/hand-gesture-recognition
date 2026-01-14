# Based on:
# https://github.com/google-ai-edge/mediapipe/blob/d1e2e0c7eed0f0ccc237b4e8b78528cd4533ca9e/mediapipe/framework/formats/landmark.proto

from typing import List, Optional

# A normalized version of Landmark proto. All coordinates should be within [0, 1].
class NormalizedLandmark:
    def __init__(
        self,
        x: Optional[float] = None,
        y: Optional[float] = None,
        z: Optional[float] = None,
        visibility: Optional[float] = None,
        presence: Optional[float] = None
    ) -> None:
        self.x, self.y, self.z, self.visibility, self.presence = x, y, z, visibility, presence

class NormalizedLandmarkList:
    def __init__(self, landmark: Optional[List[NormalizedLandmark]] = None) -> None:
        self.landmark = [] if landmark is None else landmark
