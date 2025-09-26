"""
model_manager.py
"""
from insightface.app import FaceAnalysis

face_model = None


def get_face_model():
    """
        Initialize and return the global face analysis model.

        Author
        --------------
        Name: Darshan H Shah

        Behavior:
            - Checks if global face_model is already initialized.
            - If not, creates a FaceAnalysis instance with detection and
            recognition enabled.
            - Prepares the model with GPU context (CUDA).
            - Returns the initialized face_model.

        Globals Modified:
            face_model (FaceAnalysis): Initialized if previously None.

        Returns:
            FaceAnalysis: Loaded face detection and recognition model.
    """

    global face_model
    if face_model is None:
        face_model = FaceAnalysis(name='buffalo_l',
                                  providers=['CUDAExecutionProvider'],
                                  allowed_modules=["detection", "recognition"])
        face_model.prepare(ctx_id=0)
    return face_model
