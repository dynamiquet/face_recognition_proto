# Face Recognition Proto

A real-time face recognition system using a webcam. Detects and identifies known individuals live from a camera feed, drawing labeled bounding boxes around recognized faces.

## How It Works

1. **Face detection** — Each video frame is downscaled to 1/4 resolution and passed through a HOG + Linear SVM detector (via dlib) to locate faces.
2. **Face encoding** — Detected faces are passed through a pretrained ResNet-34 model that maps each face to a 128-dimensional embedding vector.
3. **Identification** — The embedding is compared against known encodings using Euclidean distance. The closest match wins, provided the distance is below the threshold of `0.6`. Otherwise the face is labeled `Unknown`.

Every other frame is skipped to improve performance.

## Project Structure

```
face_recognition_proto/
├── face_recognition/
│   ├── __init__.py
│   └── api.py           # Core detection and encoding logic (wraps dlib)
├── webcam/
│   ├── webcam.py        # Main script
│   ├── Nathan.jpg       # Reference image
│   ├── Jesse.jpg        # Reference image
│   └── Dynamique.jpeg   # Reference image
└── requirements.txt
```

## Setup

**Prerequisites:** Python 3.9+, a working webcam, and [dlib](http://dlib.net) installed (requires cmake and a C++ compiler).

```bash
# Install dependencies
pip install -r requirements.txt
pip install opencv-python
pip install git+https://github.com/ageitgey/face_recognition_models
```

## Adding a New Person

1. Add a clear, well-lit photo of the person to the `webcam/` directory (e.g. `Person.jpg`).
2. In `webcam/webcam.py`, load and encode the image:
   ```python
   person_image = face_recognition.load_image_file("Person.jpg")
   person_face_encoding = face_recognition.face_encodings(person_image)[0]
   ```
3. Add the encoding and name to the known lists:
   ```python
   known_face_encodings = [..., person_face_encoding]
   known_face_names = [..., "Person"]
   ```

## Running

```bash
cd webcam/
python webcam.py
```

Press `q` to quit.

## Tuning

The match threshold is set by `DISTANCE_THRESHOLD = 0.6` in `webcam.py`. Lower values are stricter (fewer false positives, more `Unknown` labels). Higher values are more permissive.
