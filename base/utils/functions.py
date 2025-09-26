from datetime import datetime
try:
    from base.utils import constants
except: 
    from utils import constants
import pandas as pd
import pytz
import cv2
import numpy as np


def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    if fm < 50:  
        return frame, False

   
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    frame = cv2.filter2D(frame, -1, kernel)
    return frame, True


def frames_per_cycle(fps: int = 30) -> int:
    return int(fps * 60 * 60 * 24 * 365.25)


def get_classes(face_dir_path):
    class_ls = list(pd.read_csv(f'{face_dir_path}/known_faces.csv')
                    ['Name'].unique()) + ["Unknown"]
    return class_ls


def update_and_save(df, path):
    df.drop(columns=["TypeOfCamera"], inplace=True)
    df.to_csv(path, index=False)


def load_and_prepare(path, cam_type):
    df = pd.read_csv(path)
    df['TypeOfCamera'] = cam_type
    df["timed"] = pd.to_datetime(df["timed"], utc=True).dt.tz_convert(
        constants.TIME_ZONE).dt.floor('s')

    return df


def get_month_days(year=datetime.now(pytz.timezone(constants.TIME_ZONE)).year,
                   month=datetime.now(
                       pytz.timezone(constants.TIME_ZONE)).month):
    if month > 7:
        return 30 + (month + 1) % 2
    else:
        if month == 2:
            return 28 + int(bool(year % 4 == 0))
        else:
            return 30 + month % 2


def set_time(gap):
    t = str(gap).split("days")[-1].strip()
    h, m, s = map(int, t.split(":"))

    m += s / 60
    if h == 0:
        if m == 0:
            h = "0|(Hours)"
        else:
            h = f"{round(m / 60, 2)}|(HOURS)"
    else:

        h += m / 60
        h = f"{round(h, 2)}|(Hours)"

    return h
