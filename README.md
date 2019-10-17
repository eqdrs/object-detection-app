## Object Detection App

### Description

It uses the TensorFlow Object Detection API as backend and the pre-trained model offered by the API.

### Requirements

- Anaconda | Python >= 3.6.5 
- OpenCV == 3.4.3.18
- Tensorflow == 1.14.0

### Usage

Execute this command to run application:

```bash
$ cd object-detection-app/
$ python object_detection_app.py
```

You can also configure your webcam device ID:

```bash
$ cd object-detection-app/
$ python object_detection_app.py --source=1
```