"""
0_get_videos.py
"""
import cv2
import pandas as pd
import os
import sys

try:
    from base.utils import constants
except Exception:
    from utils import constants

n = os.getenv("EMP_ID")
name = os.getenv("EMP_NAME")
if not n or not name or n.lower() in ["exit", ""] or \
        name.lower() in ["exit", ""]:
    sys.exit(0)

cap = cv2.VideoCapture(0)
df = pd.read_csv(f"{constants.FACE_DIR_PATH}/name_file.csv")
new_row_df = pd.DataFrame([{'employee_id': n, "employee_name": name,
                            "status": "active"}])
df = pd.concat([df, new_row_df], ignore_index=True)
df.to_csv(f"{constants.FACE_DIR_PATH}/name_file.csv", index=False)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

out = cv2.VideoWriter(f'{constants.VIDEO_DIR_PATH}/{n}.mp4', fourcc,
                      cap.get(cv2.CAP_PROP_FPS),
                      (int(frame_width), int(frame_height)))

while True:
    ret, frame = cap.read()
    if ret:
        out.write(frame)
        cv2.imshow('Camera Feed', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
sys.exit(0)
