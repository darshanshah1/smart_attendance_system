"""
d_recognition_loop.py
"""

import time
import threading
from queue import Empty, Queue, Full
import cv2
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import concurrent.futures
import pytz
import warnings
import sys
import os
import gc
import random

try:
    from base.utils.model_manager import get_face_model
    from base.c_recognize import match_face
    from base.utils import constants
    from base.utils.functions import frames_per_cycle, get_classes, \
        preprocess_frame
except Exception:
    from utils.model_manager import get_face_model
    from c_recognize import match_face
    from utils import constants
    from utils.functions import frames_per_cycle, get_classes, preprocess_frame

warnings.filterwarnings("ignore")
os.environ["ORT_LOG_LEVEL"] = "ERROR"

input_frame_queue = Queue(maxsize=constants.FRAME_QUEUE_SIZE)
output_frame_queue = Queue(maxsize=constants.FRAME_QUEUE_SIZE)
result_queue = Queue(maxsize=constants.RESULT_QUEUE_SIZE)

columns = constants.COLUMNS
time_now = datetime.now(pytz.timezone(constants.TIME_ZONE))
os.environ[constants.ENV_CAPTURE_OPTIONS_KEY] = constants.CAPTURE_OPTIONS

classes = get_classes(constants.FACE_DIR_PATH)
face_model = get_face_model()

check_ip_dict = {str(k): 0 for k in classes}
check_op_dict = {str(k): 0 for k in classes}

MAX_DETECTIONS = constants.MAX_DETECTION
EMBED_DIM = constants.NUMBER_OF_FEATURES

daily_ip = {}
daily_op = {}

count_input = 0
count_output = 0

ip_csv_path = f"{constants.IP_OP_BUFFER_PATH}/i_p.csv"
op_csv_path = f"{constants.IP_OP_BUFFER_PATH}/o_p.csv"


def re_init_buffers():
    """
    Initialize counters and buffers from CSV
    Author
    --------------
    Name: Darshan H Shah
    """
    global daily_op, daily_ip
    for t_csv_path, d_dict in [(ip_csv_path, daily_ip),
                               (op_csv_path, daily_op)]:
        df = pd.read_csv(t_csv_path, usecols=[columns[0], columns[-1]],
                         dtype={columns[0]: str}, parse_dates=[columns[-1]])
        df = df[df['timed'].dt.date == time_now.date()]
        last_dates = df.groupby(columns[0])[columns[-1]].agg(['max', 'count'])
        for emp, row in last_dates.iterrows():
            last_ts, cnt = row['max'], row['count']

            d_dict[emp] = {"date": last_ts.date(), "last": last_ts,
                           "count": cnt,
                           "pred": 0}


re_init_buffers()


def is_login_entry(employee_id: str, now_time: datetime) -> bool:
    """
    Check if an employee already has an input entry today.
    Author
    --------------
    Name: Darshan H Shah
    
    Behavior:
        - Looks up employee in daily_ip cache.
        - Returns True if no entry exists for today.

    Args:
        employee_id (str): Employee identifier.
        now_time (datetime): Current timestamp.

    Globals Modified:
        None

    Returns:
        bool: True if this employee can log a new input entry today.
    """
    rec = daily_ip.get(employee_id)
    return rec is None or rec["date"] != now_time.date()


def remove_row_from_csv(employee_id: str, csv_path: str):
    """
    Delete rows for employee_id within threshold directly from CSV.
    
    Author
    --------------
    Name: Darshan H Shah
    """
    row_time = datetime.now(pytz.timezone(constants.TIME_ZONE))
    temp_rows = []
    with open(csv_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split(",")
            if parts[0] != employee_id:
                temp_rows.append(line)
                continue
            ts = pd.to_datetime(parts[-1])
            if (row_time - ts) > timedelta(**constants.THRESHOLD_DICT):
                temp_rows.append(line)
    with open(csv_path, "w") as f:
        f.writelines(temp_rows)
    re_init_buffers()


def was_login_entry(employee_id: str, now_time: datetime) -> bool:
    """
    Determine if this is the first input entry today for an employee.
    
    Author
    --------------
    Name: Darshan H Shah
    
    Behavior:
        - Checks daily_ip cache to see if today's entry count is 1.

    Args:
        employee_id (str): Employee identifier.
        now_time (datetime): Current timestamp.

    Globals Modified:
        None

    Returns:
        bool: True if this is the employee's first entry today.
    """
    rec = daily_ip.get(employee_id)
    if rec is None or rec["date"] != now_time.date():
        return False
    return rec["count"] == 1


def next_valid_entry(employee_id: str, now_time: datetime) -> str:
    """
    Determine which camera type (INPUT/OUTPUT) is valid for next entry.

    Author
    --------------
    Name: Darshan H Shah
    
    Behavior:
        - Compares last input and output timestamps for today.
        - Enforces alternation logic: input -> output -> input.

    Args:
        employee_id (str): Employee identifier.
        now_time (datetime): Current timestamp.

    Globals Modified:
        None

    Returns:
        str: Either constants.INPUT_CAMERA_NAME or constants.OUTPUT_CAMERA_NAME.
    """
    rec_in = daily_ip.get(employee_id)
    rec_out = daily_op.get(employee_id)

    ti = rec_in.get("last") if rec_in and rec_in.get(
        "date") == now_time.date() else None
    to = rec_out.get("last") if rec_out and rec_out.get(
        "date") == now_time.date() else None

    if ti is None:
        return constants.INPUT_CAMERA_NAME
    elif to is None:
        return constants.OUTPUT_CAMERA_NAME
    elif ti < to:
        return constants.INPUT_CAMERA_NAME
    else:
        return constants.OUTPUT_CAMERA_NAME


def any_recent_exit(employee_id: str, now_time: datetime, cm_type: str) -> bool:
    """
    Check if employee has an exit record within the threshold window.

    Author
    --------------
    Name: Darshan H Shah
    
    Behavior:
        - Looks up last output timestamp from daily_op.
        - Compares with current time and THRESHOLD_DICT.

    Args:
        employee_id (str): Employee identifier.
        now_time (datetime): Current timestamp.
        cm_type (str): Type of Camera

    Globals Modified:
        None

    Returns:
        bool: True if a recent exit exists within the threshold.
    """
    rec = daily_ip if cm_type == constants.INPUT_CAMERA_NAME else daily_op
    rec_out = rec.get(employee_id)
    if rec_out is None or rec_out["date"] != now_time.date():
        return False
    return (now_time - rec_out["last"]) <= timedelta(**constants.THRESHOLD_DICT)


def relocate_count(camera_type: str, count: int):
    """
        Update global counters for input or output camera detections.

        Author
        --------------
        Name: Darshan H Shah

        Behavior:
            - Checks camera type against configured INPUT_CAMERA_NAME.
            - Updates the corresponding global counter with the given count.

        Args:
            camera_type (str): Name of the camera (input or output).
            count (int): Updated count value to assign.

        Globals Modified:
            count_input (int): Updated if camera_type matches input camera.
            count_output (int): Updated if camera_type matches output camera.

        Returns:
            None
    """

    global count_input, count_output
    if camera_type == constants.INPUT_CAMERA_NAME:
        count_input = count
    else:
        count_output = count


def save_img(detection: insightface.app.common.Face, frame: np.uint8,
             row_date, row_time, key, camera_type):
    """
    Save cropped detection image with optional side-by-side comparison.

    Author
    --------------
    Name: Darshan H Shah

    Behavior:
        - Crops detection region from frame with padding.
        - Creates output directory by date if not existing.
        - If a reference image exists for the video key, resizes and merges it
        with the cropped frame.
        - Saves final image with timestamp and camera type.

    Args:
        detection (Face object): Detection object containing bbox coordinates.
        frame (ndarray): Original image frame from video.
        row_date (str): Date string used for folder naming.
        row_time (datetime): Timestamp used in filename.
        key (str): Identifier for video or employee.
        camera_type (str): Type of camera (IN/OUT).

    Globals Modified:
        None

    Returns:
        None
    """

    stx, sty, enx, eny = detection.bbox
    stx, sty, enx, eny = int(stx), int(sty), int(enx), int(eny)
    cnt = 50
    h, w = frame.shape[:2]
    top = max(sty - cnt, 0)
    bottom = min(eny + cnt, h)
    left = max(stx - cnt, 0)
    right = min(enx + cnt, w)
    t1_frame = frame[top:bottom, left:right]
    os.makedirs(f"{constants.FACE_DIR_PATH}/tests/{row_date}", exist_ok=True)
    image_read_path = f"{constants.FACE_DIR_PATH}/ip_test/{key}.mp4/1.jpg"
    if os.path.exists(image_read_path):
        og_img = cv2.imread(image_read_path)
        h1 = t1_frame.shape[0]
        h2 = og_img.shape[0]
        if h1 != h2:
            scale_ratio = h1 / h2
            new_w = int(og_img.shape[1] * scale_ratio)
            og_img = cv2.resize(og_img, (new_w, h1))
        merged_img = np.hstack((t1_frame, og_img))
        row_time = row_time.time()
    else:
        merged_img = t1_frame
    image_store_path = f"{constants.FACE_DIR_PATH}/tests/{row_date}/" \
                       f"{key}_{row_time}_{camera_type}.jpg"
    cv2.imwrite(image_store_path, merged_img)


def add_one(key, pred, camera_type, img_frame, detection):
    """
    Process a recognition event for an employee and update logs.

    Author
    --------------
    Name: Darshan H Shah
    
    Behavior:
        - Determines whether to insert a new record based on login state,
          alternation logic, and recent exit.
        - Deletes conflicting CSV rows if necessary.
        - Appends new row to CSV.
        - Prints detection summary.

    Args:
        detection:
        img_frame:
        key (str): Employee identifier.
        pred (float): Recognition confidence.
        camera_type (str): Camera type (INPUT/OUTPUT).

    Globals Modified:
        daily_ip, daily_op
        ip_csv_path, op_csv_path

    Returns:
        None
    """
    if key == constants.UNKNOWN_STR:
        return
    global daily_op, daily_ip
    row_time = datetime.now(pytz.timezone(constants.TIME_ZONE)).replace(
        microsecond=0)
    row_data = [key, round(float(pred), 3), row_time]
    row_date = row_time.date()
    if camera_type != next_valid_entry(key, row_time):
        return

    elif camera_type == constants.INPUT_CAMERA_NAME:
        if any_recent_exit(key, row_time, constants.OUTPUT_CAMERA_NAME):
            remove_row_from_csv(key, op_csv_path)
            save_img(detection, img_frame, row_date, row_time, key, camera_type)
    elif camera_type == constants.OUTPUT_CAMERA_NAME:
        if any_recent_exit(key, row_time, constants.INPUT_CAMERA_NAME):
            if not was_login_entry(key, row_time):
                remove_row_from_csv(key, ip_csv_path)
                save_img(detection, img_frame, row_date,
                         row_time, key, camera_type)
    if camera_type == next_valid_entry(key, row_time):
        f_name = 'i_p' if camera_type == constants.INPUT_CAMERA_NAME else 'o_p'
        with open(f"{constants.IP_OP_BUFFER_PATH}/{f_name}.csv", "a") as f:
            f.write(f"{row_data[0]},{row_data[1]},{row_data[2]}\n")

        daily_dict = daily_ip if camera_type == constants.INPUT_CAMERA_NAME \
            else daily_op
        rec = daily_dict.get(key)
        if rec is None or rec["date"] != row_time.date():
            daily_dict[key] = {"date": row_time.date(), "last": row_time,
                               'pred': pred, "count": 1}
        else:
            rec['pred'] = pred
            rec["last"] = row_time
            rec["count"] += 1
        prob = round(float(pred) * 100, 2)
        save_img(detection, img_frame, row_date, row_time, key, camera_type)
        print(f"{key}\t{camera_type}\t{prob:.2f}%\t{row_time}")


def capture_loop(camera_address, frame_queue, camera_type):
    """
    Continuously capture frames from a camera and enqueue them.

    Author
    --------------
    Name: Darshan H Shah
    
    Behavior:
        - Opens video stream via OpenCV.
        - Resizes frames according to FX_RATIO/FY_RATIO.
        - Pushes frames into queue, drops if queue is full.
        - Maintains per-camera frame counter.
        - Runs continuously until termination.

    Args:
        camera_address (str): Video stream address.
        frame_queue (queue.Queue): Queue to store frames.
        camera_type (str): Camera type (INPUT/OUTPUT).

    Globals Modified:
        count_input, count_output

    Returns:
        None
    """
    global count_input, count_output
    count = count_input if camera_type == constants.INPUT_CAMERA_NAME \
        else count_output

    video_capture = cv2.VideoCapture(camera_address, cv2.CAP_FFMPEG)
    video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not video_capture.isOpened():
        return

    print(f"{camera_type} Camera Started")
    try:
        while constants.TRUE_BOOL:

            ret, frame = video_capture.read()
            if not ret:
                continue

            if count >= frames_per_cycle():
                count = 0
            count += 1
            if count % constants.OP_FPS != 0:
                continue

            frame = cv2.resize(frame, (0, 0), fx=constants.FX_RATIO,
                               fy=constants.FY_RATIO,
                               interpolation=cv2.INTER_AREA)

            if frame_queue.full():
                try:
                    frame_queue.get_nowait()
                except Empty:
                    pass
            try:
                frame_queue.put_nowait(frame)
            except Full:
                count -= 1
                relocate_count(camera_type, count)
                continue

            relocate_count(camera_type, count)

    finally:
        video_capture.release()


def recognition_loop(frame_queue, camera_type):
    """
    Process frames to detect and recognize faces.

    Author
    --------------
    Name: Darshan H Shah
    
    Behavior:
        - Retrieves frames from queue at OP_FPS rate.
        - Updates current timestamp.
        - Performs face detection and embedding extraction.
        - Matches embeddings to known identities.
        - Decrements counters for identities no longer detected.
        - Pushes recognition results to result_queue.

    Args:
        frame_queue (queue.Queue): Queue with frames.
        camera_type (str): Camera type (INPUT/OUTPUT).

    Globals Modified:
        time_now, count_input, count_output
        check_ip_dict, check_op_dict

    Returns:
        None
    """
    global time_now, count_input, count_output
    count = count_input if camera_type == constants.INPUT_CAMERA_NAME \
        else count_output

    while constants.TRUE_BOOL:
        try:
            frame = frame_queue.get(timeout=0.01)
        except Empty:
            continue

        time_now = datetime.now(pytz.timezone(constants.TIME_ZONE))
        detections = face_model.get(frame)
        if not detections:
            continue

        if len(detections) > constants.MAX_DETECTION:
            detections = sorted(detections, key=lambda d: d.confidence,
                                reverse=True)[:constants.MAX_DETECTION]
        emb = np.array([d.normed_embedding for d in detections],
                       dtype=np.float32)
        if emb.size == 0:
            continue
        if len(detections) > 10:
            ns = min(4, len(emb))
            with concurrent.futures.ThreadPoolExecutor(max_workers=ns) as pool:
                results = list(pool.map(match_face, np.array_split(emb, ns)))
            labels = []
            preds = []
            for lbl_chunk, pred_chunk in results:
                labels.extend(lbl_chunk)
                preds.extend(pred_chunk)
        else:
            labels, preds = match_face(emb)

        if labels:
            result_queue.put((labels, preds, camera_type, frame, detections))
        if camera_type == constants.INPUT_CAMERA_NAME:
            count_input = count
        else:
            count_output = count


def process_results():
    """
    Consume recognition results and update employee activity logs.

    Author
    --------------
    Name: Darshan H Shah
    
    Behavior:
        - Reads results from result_queue.
        - Calls add_one() for each recognized employee.
        - Marks processed items as done.
        - Sleeps briefly when the queue is empty.

    Globals Modified:
        check_ip_dict, check_op_dict, time_now

    Returns:
        None
    """
    global time_now
    while constants.TRUE_BOOL:
        try:
            labels, preds, camera_type, frame, detections = \
                result_queue.get_nowait()
            if labels:
                for n, (x, y) in enumerate(zip(labels, preds)):
                    add_one(x, y, camera_type, frame, detections[n])
            result_queue.task_done()
        except Empty:
            time.sleep(0.01)


def start_background_tasks():
    """
    Start background threads for video capture, recognition,
    and result processing.

    Author
    --------------
    Name: Darshan H Shah
    
    Behavior:
        - Spawns threads for capture_loop (input/output cameras).
        - Spawns threads for recognition_loop (input/output queues).
        - Spawns thread for process_results().
        - Threads run as daemons and operate continuously.

    Globals Modified:
        None

    Returns:
        None
    """
    threads = [
        threading.Thread(target=capture_loop, args=(
            constants.INPUT_CAMERA_ADDRESS1, input_frame_queue,
            constants.INPUT_CAMERA_NAME), daemon=True),

        threading.Thread(target=recognition_loop,
                         args=(input_frame_queue, constants.INPUT_CAMERA_NAME),
                         daemon=True),

        threading.Thread(target=capture_loop, args=(
            constants.OUTPUT_CAMERA_ADDRESS1, output_frame_queue,
            constants.OUTPUT_CAMERA_NAME), daemon=True),

        threading.Thread(target=recognition_loop, args=(
            output_frame_queue, constants.OUTPUT_CAMERA_NAME), daemon=True),

        threading.Thread(target=process_results, daemon=True),

    ]
    for t in threads:
        t.start()


if __name__ == "__main__":
    try:
        start_background_tasks()
        while constants.TRUE_BOOL:
            time.sleep(0.0001)
    except KeyboardInterrupt:
        print("Shutting down...")
    except Exception as e:
        raise f"Main loop error: {e}"
    finally:
        cv2.destroyAllWindows()
        print(">>>>> RESOURCES RELEASED")
        sys.exit(0)
