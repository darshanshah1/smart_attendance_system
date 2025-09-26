"""
    c_recognize.py
"""
import numba
import numpy as np
import logging
import joblib

try:
    from base.utils import constants
except Exception:
    from utils import constants

import faiss

logging.basicConfig(filename='logs/face_recognition.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
knn_model = None
mlp_model = None
index = None
y_train = None


def load_models():
    """
        Load machine learning models and face index for recognition.

        Author
        --------------
        Name: Darshan H Shah

        Behavior:
            - Loads KNN and MLP models from disk using joblib.
            - Reads FAISS index for face embeddings.
            - Loads training labels for face recognition.
            - Logs success or failure.

        Globals Modified:
            knn_model (KNeighborsClassifier): Loaded KNN model.
            mlp_model (MLPClassifier): Loaded MLP model.
            index (faiss.Index): Loaded FAISS index for embeddings.
            y_train (ndarray): Training labels for recognition.

        Returns:
            None
    """
    global knn_model, mlp_model, index, y_train
    try:
        knn_model = joblib.load(f'{constants.MODEL_DIR_PATH}/knn_model.pkl')
        mlp_model = joblib.load(f'{constants.MODEL_DIR_PATH}/mlp_model.pkl')
        index = faiss.read_index(f"{constants.MODEL_DIR_PATH}/face_index.faiss")
        y_train = joblib.load(f"{constants.MODEL_DIR_PATH}/face_labels.pkl")
        logging.info("KNN and MLP models loaded successfully")
    except Exception as e:
        logging.error(f"Could not load models: {e}")


load_models()


def match_face(embeddings, knn_thresh=constants.KNN_THRESHOLD,
               mlp_thresh=constants.MLP_THRESHOLD,
               combined_thresh=constants.RECOGNIZE_THRESHOLD):
    """
        Match face embeddings to employee labels using KNN, MLP, and thresholds.

        Author
        --------------
        Name: Darshan H Shah

        Behavior:
            - Accepts single or batch embeddings.
            - Predicts labels using KNN and MLP classifiers.
            - Applies confidence thresholds for each method.
            - Combines results to return most reliable matches.

        Args:
            embeddings (ndarray): Face embeddings of shape (n, 512) or (512,).
            knn_thresh (float, optional): Threshold for KNN confidence.
                                          Defaults to 0.6.
            mlp_thresh (float, optional): Threshold for MLP confidence.
                                          Defaults to 0.8.
            combined_thresh (float, optional): Threshold for combined confidence
                                               Defaults to 0.8.

        Globals Modified:
            None

        Returns:
            list[tuple[str, float]] | tuple[str, float]:
                For batch input: list of (label, score).
                For single input: single (label, score).
    """

    global knn_model, mlp_model

    knn_dist, _ = knn_model.kneighbors(embeddings)
    knn_labels = knn_model.predict(embeddings)
    knn_scores = np.round(1 - (knn_dist / 2), 4)

    mlp_probs = mlp_model.predict_proba(embeddings)
    mlp_labels = mlp_model.classes_[np.argmax(mlp_probs, axis=1)]
    mlp_scores = np.max(mlp_probs, axis=1).round(4)

    label_ls = []
    pred_ls = []

    for i in range(embeddings.shape[0]):
        knn_label = knn_labels[i] if knn_scores[i, 0] > knn_thresh \
            else constants.UNKNOWN_STR
        mlp_label = mlp_labels[i] if mlp_scores[i] > mlp_thresh \
            else constants.UNKNOWN_STR

        scores = {}
        labels = [(mlp_scores[i], mlp_label, 0.7),
                  (knn_scores[i, 0], knn_label, 0.6)]
        for pred_score, label, weight in labels:
            scores[label] = scores.get(label, 0) + pred_score * weight

        most_common_name = max(scores, key=scores.get)
        best_score = round(scores[most_common_name], 4)

        if best_score / 1.3 < combined_thresh or \
                most_common_name == constants.UNKNOWN_STR:
            most_common_name = constants.UNKNOWN_STR

        label_ls.append(most_common_name)
        pred_ls.append(round(best_score / 1.3, 4))

    return label_ls, pred_ls
