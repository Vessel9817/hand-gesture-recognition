# Hand Tracking and Facial Recognition

https://ai.google.dev/edge/mediapipe/solutions/vision/gesture_recognizer#get_started

A real-time hand and facial recognition system built with MediaPipe.
Tracks landmarks using a webcam. Can be used for sign language recognition,
gesture-based controls or interactive applications.

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
   curl -o ./src/hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
   curl -o ./src/face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
   ```

   Newer [hand][hand models] or [face models][face models]
   may be available in the future.
   You can also write a [custom model][custom models].

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
      py "./src/main.py"
      ```

- Move your hands and face around in view of the camera.
  Play around with it and test its limits!
- With the window focused, press `q` to exit the program.

## Contributing

See: [CONTRIBUTING.md](CONTRIBUTING.md)

## License

This project is licensed under the AGPLv3 License - see
the [LICENSE](LICENSE) file for details.

[hand models]: https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker#models
[face models]: https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker#models
[custom models]: https://ai.google.dev/edge/mediapipe/solutions/customization/gesture_recognizer
