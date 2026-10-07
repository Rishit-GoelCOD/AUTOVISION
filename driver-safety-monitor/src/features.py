"""
Feature Extraction Module for Driver Safety Monitoring.

Extracts 4 facial biomarkers using MediaPipe Face Mesh:
1. Eye Aspect Ratio (EAR)       - Eye closure / blink detection (x, y pixel coordinates).
2. Mouth Aspect Ratio (MAR)     - Yawn detection (x, y pixel coordinates).
3. Yaw                          - Horizontal head turn angle using the cheek line (landmarks 234 to 454).
4. Pitch                        - Vertical head tilt angle using the forehead-to-chin line (landmarks 10 to 152).

No neutral offset subtraction, no clamping, no rounding, and no smoothing.
"""

from pathlib import Path
from typing import Optional, Tuple
import urllib.request

import cv2
import numpy as np

# =============================================================================
# Landmark Indices for MediaPipe Face Mesh (468 / 478 points)
# =============================================================================
# Left Eye (6 landmarks): [outer_corner, top_left, top_right, inner_corner, bottom_right, bottom_left]
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]

# Right Eye (6 landmarks): [inner_corner, top_left, top_right, outer_corner, bottom_right, bottom_left]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

# Mouth (6 landmarks): [left_corner, upper_left, upper_right, right_corner, lower_right, lower_left]
MOUTH_INDICES = [61, 39, 269, 291, 405, 181]

# Head Pose Key Landmark Indices:
# 234 : Left cheek (viewer's left)
# 454 : Right cheek (viewer's right)
# 10  : Forehead
# 152 : Chin
LEFT_CHEEK_INDEX = 234
RIGHT_CHEEK_INDEX = 454
FOREHEAD_INDEX = 10
CHIN_INDEX = 152


# =============================================================================
# Helper: Face Mesh Detector Backend (Dual Legacy / Tasks Support)
# =============================================================================
class FaceMeshDetector:
    """Encapsulates MediaPipe Face Mesh detection across package versions."""

    def __init__(self, min_detection_confidence: float = 0.5):
        import mediapipe as mp

        self.backend = None
        self.detector = None

        if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
            self.backend = "solutions"
            self.detector = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=0.5,
            )
        else:
            self.backend = "tasks"
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            project_root = Path(__file__).resolve().parent.parent
            model_dir = project_root / "models"
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / "face_landmarker.task"

            if not model_path.exists():
                model_url = (
                    "https://storage.googleapis.com/mediapipe-models/"
                    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
                )
                urllib.request.urlretrieve(model_url, model_path)

            base_options = python.BaseOptions(model_asset_path=str(model_path))
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                min_face_detection_confidence=min_detection_confidence,
                num_faces=1,
            )
            self.detector = vision.FaceLandmarker.create_from_options(options)

    def detect_landmarks(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Runs face mesh on a BGR image frame.

        Converts each landmark into pixel-scale 3D coordinates:
            X = x * frame_width
            Y = y * frame_height
            Z = z * frame_width

        Returns:
            np.ndarray of shape (N, 3), or None if no face is detected.
        """
        import mediapipe as mp

        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        if self.backend == "solutions":
            results = self.detector.process(frame_rgb)
            if not results.multi_face_landmarks:
                return None
            landmarks = results.multi_face_landmarks[0].landmark
            return np.array([(lm.x * w, lm.y * h, lm.z * w) for lm in landmarks], dtype=np.float64)

        elif self.backend == "tasks":
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            results = self.detector.detect(mp_img)
            if not results.face_landmarks:
                return None
            landmarks = results.face_landmarks[0]
            return np.array([(lm.x * w, lm.y * h, lm.z * w) for lm in landmarks], dtype=np.float64)

        return None


_GLOBAL_DETECTOR: Optional[FaceMeshDetector] = None


def get_detector() -> FaceMeshDetector:
    """Returns singleton instance of FaceMeshDetector."""
    global _GLOBAL_DETECTOR
    if _GLOBAL_DETECTOR is None:
        _GLOBAL_DETECTOR = FaceMeshDetector()
    return _GLOBAL_DETECTOR


# =============================================================================
# 1. Eye Aspect Ratio (EAR)
# =============================================================================
def eye_aspect_ratio(eye_landmarks: np.ndarray) -> float:
    """
    Computes Eye Aspect Ratio (EAR) using 2D (x, y) coordinates.

    Mathematical Formula:
        EAR = ( ||p2 - p6|| + ||p3 - p5|| ) / ( 2.0 * ||p1 - p4|| )

    Formula Explanation in Simple Language:
    ---------------------------------------
    - p1 and p4 are the outer and inner corners of the eye (horizontal width).
    - (p2, p6) and (p3, p5) are two pairs of top and bottom eyelid points (vertical height).
    - Numerator: Adds the two vertical distances to measure how tall the eye opening is.
    - Denominator: Twice the horizontal eye width to make the ratio scale-invariant.
    - An open eye yields ~0.35, while a closing/closed eye drops toward ~0.10.
    """
    p1, p2, p3, p4, p5, p6 = eye_landmarks

    # Use x and y only
    dist_v1 = np.linalg.norm(p2[:2] - p6[:2])
    dist_v2 = np.linalg.norm(p3[:2] - p5[:2])
    dist_h = np.linalg.norm(p1[:2] - p4[:2])

    if dist_h < 1e-6:
        return 0.0

    ear = (dist_v1 + dist_v2) / (2.0 * dist_h)
    return float(ear)


# =============================================================================
# 2. Mouth Aspect Ratio (MAR)
# =============================================================================
def mouth_aspect_ratio(mouth_landmarks: np.ndarray) -> float:
    """
    Computes Mouth Aspect Ratio (MAR) using 2D (x, y) coordinates.

    Mathematical Formula:
        MAR = ( ||p2 - p6|| + ||p3 - p5|| ) / ( 2.0 * ||p1 - p4|| )

    Formula Explanation in Simple Language:
    ---------------------------------------
    - p1 and p4 are the left and right corners of the mouth (horizontal width).
    - (p2, p6) and (p3, p5) are two pairs of upper and lower lip points (vertical height).
    - Numerator: Adds the two vertical opening heights.
    - Denominator: Twice the horizontal mouth width to normalize for face size.
    - A closed mouth yields ~0.30 - 0.38, while a wide open yawn reaches ~0.60 - 0.70+.
    """
    p1, p2, p3, p4, p5, p6 = mouth_landmarks

    # Use x and y only
    dist_v1 = np.linalg.norm(p2[:2] - p6[:2])
    dist_v2 = np.linalg.norm(p3[:2] - p5[:2])
    dist_h = np.linalg.norm(p1[:2] - p4[:2])

    if dist_h < 1e-6:
        return 0.0

    mar = (dist_v1 + dist_v2) / (2.0 * dist_h)
    return float(mar)


# =============================================================================
# 3. Head Pose: Yaw and Pitch (Cheek Line and Forehead-to-Chin Line)
# =============================================================================
def head_pose(
    landmarks: np.ndarray,
    frame_shape=None,
) -> Tuple[float, float]:
    """
    Computes Yaw and Pitch independently using 3D landmark lines.

    Yaw Formula:
        dx = X454 - X234
        dz = Z454 - Z234
        yaw = degrees(atan2(dz, dx))

    Yaw Explanation in Simple Language:
    -----------------------------------
    - We draw a horizontal line between the left cheek (234) and right cheek (454).
    - dx is the horizontal separation across the screen, and dz is the depth difference between cheeks.
    - When facing forward, both cheeks are at the same depth, so dz is near zero and yaw is ~0 degrees.
    - Turning the head right pushes the right cheek further away into the screen (dz > 0), giving positive yaw.
    - Turning the head left pushes the right cheek closer and the left cheek further away (dz < 0), giving negative yaw.

    Pitch Formula:
        dy = Y152 - Y10
        dz = Z152 - Z10
        pitch = -degrees(atan2(dz, dy))

    Pitch Explanation in Simple Language:
    -------------------------------------
    - We draw a vertical line from the forehead (10) down to the chin (152).
    - dy is the vertical separation down the screen, and dz is the depth difference between chin and forehead.
    - Looking up tilts the chin forward toward the camera and the forehead backward (dz < 0).
      Negating this produces positive pitch for looking up.
    - Looking down tilts the forehead forward toward the camera and the chin backward (dz > 0).
      Negating this produces negative pitch for looking down.
    - Because pitch uses only the forehead-to-chin line and yaw uses only the cheek line,
      they are completely decoupled and do not affect each other.
    """
    # 1. Yaw calculation from cheek line (landmarks 234 and 454)
    x234, _, z234 = landmarks[LEFT_CHEEK_INDEX]
    x454, _, z454 = landmarks[RIGHT_CHEEK_INDEX]

    dx = x454 - x234
    dz_yaw = z454 - z234
    yaw = float(np.degrees(np.arctan2(dz_yaw, dx)))

    # 2. Pitch calculation from forehead-to-chin line (landmarks 10 and 152)
    _, y10, z10 = landmarks[FOREHEAD_INDEX]
    _, y152, z152 = landmarks[CHIN_INDEX]

    dy = y152 - y10
    dz_pitch = z152 - z10
    # Negate so looking up is positive and looking down is negative
    pitch = float(-np.degrees(np.arctan2(dz_pitch, dy)))

    return yaw, pitch


# =============================================================================
# 4. Feature Extraction Pipeline
# =============================================================================
def extract_features(
    frame: np.ndarray,
) -> Optional[Tuple[float, float, float, float]]:
    """
    Extracts all 4 facial biomarkers (EAR, MAR, yaw, pitch) from a video frame.

    Returns:
        Optional[Tuple[float, float, float, float]]:
            (EAR, MAR, yaw, pitch) if face detected, otherwise None.
    """
    if frame is None or frame.size == 0:
        return None

    detector = get_detector()
    landmarks = detector.detect_landmarks(frame)

    if landmarks is None or len(landmarks) < 468:
        return None

    # EAR: average of left and right eyes (x and y pixel coordinates)
    left_ear = eye_aspect_ratio(landmarks[LEFT_EYE_INDICES])
    right_ear = eye_aspect_ratio(landmarks[RIGHT_EYE_INDICES])
    ear = (left_ear + right_ear) / 2.0

    # MAR: mouth aspect ratio (x and y pixel coordinates)
    mar = mouth_aspect_ratio(landmarks[MOUTH_INDICES])

    # Head Pose: independent cheek line yaw and forehead-chin pitch
    yaw, pitch = head_pose(landmarks, frame.shape[:2])

    return float(ear), float(mar), float(yaw), float(pitch)
