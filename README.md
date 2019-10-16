## Object Detection App

### Description

It uses the TensorFlow Object Detection API as backend and the pre-trained model offered by the API.

### Requirements

- Python >= 3.3
- OpenCV == 3.4.3.18
- Tensorflow == 1.14.0

### Usage

Execute this command to run application:

```bash
$ cd object-detection-app/
$ python object_detection_app.py
```

You also can configure your webcam device ID:

```bash
$ cd object-detection-app/
$ python object_detection_app.py --source=1
```