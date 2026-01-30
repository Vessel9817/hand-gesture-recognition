# Hand Tracking and Facial Recognition

A real-time hand, face and full-body recognition system
built with [MediaPipe][MediaPipe]. Tracks landmarks using a webcam.
Can be used for sign language recognition, gesture-based controls
or interactive applications, such as VR.

Also included by MediaPipe, but not yet implemented here,
is [hand gesture categorization][hand gesture models].

## Table of Contents

- [Installation Instructions](#installation-instructions)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Installation Instructions

1. Clone the repository:

   ```bash
   git clone https://github.com/Vessel9817/hand-gesture-recognition.git
   cd hand-gesture-recognition
   ```

1. Download the landmarking models:

   ```shell
   mkdir models
   curl -o "./models/hand_landmarker.task" https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
   curl -o "./models/face_landmarker.task" https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
   curl -o "./models/pose_landmarker.task" https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task
   ```

   Newer [hand][hand models], [face][face models]
   or [full-body models][pose models] may be available in the future.

1. Create a virtual environment:

   ```shell
   py -m venv "./venv"
   ```

1. Activate the virtual environment:

   ```shell
   source "./venv/Scripts/activate"
   ```

1. Install the required libraries:

   ```shell
   pip install -r "./requirements-freeze.txt"
   ```

## Usage

- Ensure you have a working webcam connected to your device.
- Run the script:

   1. If not already done in the current shell,
      activate the virtual environment:

      ```shell
      source "./venv/Scripts/activate"
      ```

   1. Run the main Python script:

      ```shell
      py -m src.main
      ```

- Move your hands, face and body around in view of the camera.
  Play around with it and test its limits!
- With the window focused, press `q` to exit the program.

## Custom Models

See MediaPipe's [custom hand gesture recognition][custom models] sample
for how to create a custom model. The same principles apply to models in general.

## Contributing

See: [CONTRIBUTING.md](CONTRIBUTING.md)

## License

This project is licensed under the AGPLv3 License - see
the [LICENSE](LICENSE) file for details.

Credit to the MediaPipe authors for creating the samples this project
is in part based on. Individual files contain more specific attributions.

[MediaPipe]: https://ai.google.dev/edge/mediapipe/solutions/guide
[hand models]: https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker#models
[face models]: https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker#models
[pose models]: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker#models
[hand gesture models]: https://ai.google.dev/edge/mediapipe/solutions/vision/gesture_recognizer#models
[custom models]: https://ai.google.dev/edge/mediapipe/solutions/customization/gesture_recognizer
