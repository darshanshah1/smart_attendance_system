"""
b_train_all.py
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import IsolationForest

try:
    from base.utils import constants
except Exception:
    from utils import constants
import warnings
import faiss

warnings.filterwarnings("ignore")

face_db = {}
X_train = []
y_train = []


def register_known_faces(folder: str = 'static/known_faces'):
    """
        Register known faces into the face database using embeddings.

        Author
        --------------
        Name: Darshan H Shah

        Behavior:
            - Loads embeddings from the given folder.
            - Applies IsolationForest to remove outliers from embeddings.
            - Groups remaining embeddings by recognized name.
            - Stores embeddings in the global face_db for recognition use.

        Args:
            folder (str, optional): Path to folder containing embeddings file.
                                    Defaults to 'static/known_faces'.

        Globals Modified:
            face_db (dict): Populated with filtered embeddings keyed by name.

        Returns:
            None
    """
    emb_df = pd.read_csv(f"{folder}/{constants.EMBEDDINGS_FILE_NAME}").dropna()
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    outliers = iso_forest.fit_predict(emb_df.iloc[:, :-1])
    emb_df = emb_df[outliers == 1]
    for recognized_name in emb_df.iloc[:, -1].unique():
        tmp_df = emb_df[emb_df['Name'] == recognized_name].iloc[:, :-1]
        face_db[str(recognized_name)] = [row.to_numpy() for _, row in
                                         tmp_df.iterrows()]


register_known_faces()

for name, embeddings in face_db.items():
    for emb in embeddings:
        X_train.append(emb)
        y_train.append(name)

X_train = np.array(X_train, dtype="float64")
y_train = np.array(y_train)

unique_classes = len(set(y_train))
print(f"{unique_classes} UNIQUE CLASSES FOUND")

n_neighbours = min(unique_classes, 3)
knn_model = KNeighborsClassifier(n_neighbors=n_neighbours, metric='cosine')
knn_model.fit(X_train, y_train)

mlp_model = MLPClassifier(hidden_layer_sizes=(128, 64), alpha=0.01,
                          learning_rate='adaptive', learning_rate_init=1e-3,
                          max_iter=500, random_state=42)
mlp_model.fit(X_train, y_train)

if len(knn_model.classes_) == len(mlp_model.classes_) == unique_classes:
    print("All classes detected by all models")
else:
    missing_classes = set(y_train) - set(mlp_model.classes_)
    print(f"MLP classes not detected: {missing_classes}")
    missing_classes = set(y_train) - set(knn_model.classes_)
    print(f"KNN classes not detected: {missing_classes}")

joblib.dump(knn_model, f"{constants.MODEL_DIR_PATH}/knn_model.pkl")
joblib.dump(mlp_model, f"{constants.MODEL_DIR_PATH}/mlp_model.pkl")
