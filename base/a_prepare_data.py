"""
a_prepare_data.py
"""
import cv2
import pandas as pd
import os
import warnings
import logging
import shutil
import sys
import numpy as np
from insightface.app import FaceAnalysis
import cv2
import random

try:
    from base.utils import constants
    from base.utils.model_manager import get_face_model

except Exception:
    from utils import constants
    from utils.model_manager import get_face_model

# Configure logging
logging.basicConfig(filename='logs/embeddings.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

warnings.filterwarnings("ignore")
face_model = get_face_model()


def apply_augmentation(img_frame: np.uint8, aug_type: str = "random",
                       p: float = 0.5) -> np.uint8:
    """Apply a random combination of brightness, blur, and/or sharpening to a
    frame.

    Args:
        img_frame: Input frame (numpy array)
        aug_type: Not used directly; kept for compatibility ('random' for combo)
        p: Probability of applying each filter (default: 0.5)

    Returns:
        Augmented frame
    """
    augmented = img_frame.copy()  # Avoid modifying the original frame

    if aug_type == 'random':
        filters = [None, 'brightness1', 'brightness2', 'blur1', 'blur2',
                   'sharpen', 'color_jitter', 'noise']

        selected_filters = [f for f in filters if random.random() < p]

        if not selected_filters:
            selected_filters = [random.choice(filters)]

        random.shuffle(selected_filters)

        for filter_type in selected_filters:

            if filter_type is None:
                augmented = augmented

            elif filter_type.startswith('brightness'):
                brightness = random.choice([-20, -10, 10, 20])
                augmented = cv2.convertScaleAbs(augmented, alpha=1,
                                                beta=brightness)
            elif filter_type.startswith('blur'):
                bl = random.choice([3, 5, 7])
                sig = bl / 5
                augmented = cv2.GaussianBlur(augmented, (bl, bl), sigmaX=sig,
                                             sigmaY=sig)
            elif filter_type == 'sharpen':
                sh = random.choice([3, 5, 7])
                kernel = np.array([[-1, -1, -1], [-1, sh, -1], [-1, -1, -1]])
                augmented = cv2.filter2D(augmented, -1, kernel)

            elif filter_type == 'noise':
                noise = np.random.normal(0, 0.6, augmented.shape).astype(
                    np.uint8)
                augmented = cv2.add(augmented, noise)
                augmented = np.clip(augmented, 0, 255)
                augmented = cv2.GaussianBlur(augmented, (3, 3), sigmaX=1,
                                             sigmaY=1)
            elif filter_type == 'color_jitter':
                augmented = cv2.cvtColor(augmented, cv2.COLOR_BGR2HSV)
                augmented[..., 0] = np.clip(
                    augmented[..., 0] + random.randint(-5, 5), 0, 179)
                augmented = cv2.cvtColor(augmented, cv2.COLOR_HSV2BGR)

            else:
                augmented = augmented

    return augmented


def face_detect(img_frame: np.uint8, det_thresh: float = 0.6) -> Optional[list]:
    """
    Extract face embedding from a frame if detection score meets' threshold.
    """
    try:
        detections = face_model.get(img_frame)
        if len(detections) == 1 and detections[0].det_score > det_thresh:
            return detections
        return None
    except Exception as exc:
        logging.error(f"Face detection error: {exc}")
        return None


embeddings_path = f'{constants.FACE_DIR_PATH}/{constants.EMBEDDINGS_FILE_NAME}'
if os.path.exists(embeddings_path):
    emb_df = pd.read_csv(embeddings_path)
else:
    emb_df = pd.DataFrame(
        {f"f{i}": [] for i in range(constants.NUMBER_OF_FEATURES)} | {
            "Name": []})
names_path = f'{constants.FACE_DIR_PATH}/name_file.csv'

if os.path.exists(names_path):
    name_df = pd.read_csv(names_path)
else:
    name_df = pd.DataFrame({"employee_id": [], "employee_name": []})

# Process videos
max_embeddings_per_employee = 30
min_embeddings_per_employee = 10
aug_types = [None, 'brightness1', 'brightness2', 'blur1', 'blur2', 'sharpen',
             'color_jitter', 'noise']
for emp in os.listdir(constants.VIDEO_DIR_PATH):
    if not emp.lower().endswith(('.mp4', '.avi', '.mov')):
        continue
    print(emp, end=": ")
    name = emp.split('.')[0]
    source_path = f"{constants.VIDEO_DIR_PATH}/{emp}"
    destination_path = f"{constants.VIDEO_MOVE_DIR_PATH}/{emp}"

    video_capture = cv2.VideoCapture(source_path)
    if not video_capture.isOpened():
        logging.error(f"Failed to open video: {source_path}")
        continue

    embeddings = []
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    s_count = 0
    discard_frame = frame_count // 45
    f_count = 0
    while len(embeddings) < max_embeddings_per_employee:

        ret, frame = video_capture.read()

        if not ret:
            break
        f_count += 1

        if f_count % discard_frame != 0:
            continue

        aug_frame = apply_augmentation(frame)
        detection = face_detect(aug_frame)
        if detection is None:
            f_count -= 1
            continue
        embedding = detection[0].normed_embedding.astype(np.float32)
        if embedding is None:
            f_count -= 1
            continue
        s_count += 1
        if s_count == 1:
            stx, sty, enx, eny = detection[0].bbox
            stx, sty, enx, eny = int(stx), int(sty), int(enx), int(eny)
            cnt = 50
            h, w = frame.shape[:2]
            top = max(sty - cnt, 0)
            bottom = min(eny + cnt, h)
            left = max(stx - cnt, 0)
            right = min(enx + cnt, w)
            t1_frame = frame[top:bottom, left:right]
            cv2.imwrite(
                f"{constants.FACE_DIR_PATH}/ip_test/{emp}/{len(embedding)}.jpg",
                t1_frame)

        embeddings.append(list(embedding) + [name])

    video_capture.release()
    print(len(embeddings))
    emb_df = pd.concat(
        [emb_df, pd.DataFrame(embeddings, columns=emb_df.columns)],
        ignore_index=True)
    shutil.move(source_path, destination_path)

try:
    emb_df.to_csv(embeddings_path, index=False)
    logging.info(f"Saved {len(emb_df)} total embeddings to {embeddings_path}")
except Exception as e:
    logging.error(f"Error saving CSV: {e}")
finally:
    cv2.destroyAllWindows()
    sys.exit(0)
