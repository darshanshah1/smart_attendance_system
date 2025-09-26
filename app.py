"""
    app.py
"""
from datetime import datetime, timedelta
from base.utils.security import auth_required, log_route
from base.utils.functions import load_and_prepare, update_and_save
from base.utils import constants
import threading
from base.c_recognize import load_models
import pandas as pd
import subprocess
import warnings
from flask import Flask, jsonify, url_for, request, redirect, \
    Response, render_template
import os
import queue
import pytz
from base.t_trigger_backup import run_backup
from apscheduler.schedulers.background import BackgroundScheduler

warnings.filterwarnings("ignore")

app = Flask(__name__)

event_queue = queue.Queue()
# Track running scripts
running_tasks = {}
AVAILABLE_SCRIPTS = [
    "0_get_videos.py",
    "a_prepare_data.py",
    "b_train_all.py",
    "d_recognition_loop.py",
    "e_camera_checking.py"

]
SCRIPTS_NAMES = [
    "Record Video",
    "Get Face Data From Videos",
    "Train Face Data",
    "Run Recognition",
    "Check Cameras"
]


def run_script(script_name, env=None):
    """Start a script as a subprocess, track it, and notify clients of status changes."""
    if script_name in running_tasks and running_tasks[script_name].poll() \
            is None:
        return {"status": "already running",
                "pid": running_tasks[script_name].pid}

    process = subprocess.Popen(["python", "base/" + script_name], env=env or os.environ)
    running_tasks[script_name] = process

    def watcher():
        """Monitor a subprocess until it finishes, then remove it from running tasks and notify clients."""
        process.wait()
        running_tasks.pop(script_name, None)
        notify_refresh()  # refresh when finished

    threading.Thread(target=watcher, daemon=True).start()
    notify_refresh()  # refresh immediately when started
    return {"status": "started", "pid": process.pid}


def stop_script(script_name):
    """Stop a running script if active and notify clients."""
    process = running_tasks.get(script_name)
    if not process:
        return {"status": "not running"}

    if process.poll() is None:  # still running
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
    if running_tasks.get(script_name):
        del running_tasks[script_name]
    notify_refresh()  # refresh after stop
    return {"status": "stopped"}


def script_status(script_name):
    """Return the status and PID of a given script."""
    process = running_tasks.get(script_name)
    if not process:
        return {"status": "stopped", "pid": None}
    if process.poll() is None:
        return {"status": "running", "pid": process.pid}
    return {"status": "stopped", "pid": None}


listeners = []


def call_backup_api():
    """Check backup schedule and trigger backup if required."""
    try:
        today = datetime.today()
        BACKUP_DIR = constants.DATA_BACKUP_PATH

        backups = [d for d in os.listdir(BACKUP_DIR) if d.isdigit()]
        if backups:
            latest = max(backups, key=lambda x: datetime.strptime(x, "%y%m%d"))
            last_date = datetime.strptime(latest, "%y%m%d")
        else:
            last_date = None

        if not last_date or \
                today - last_date >= timedelta(days=constants.BACKUP_DAYS):
            run_backup()
        else:
            print("NO BACKUP REQUIRED AT THIS TIME")
    except Exception as e:
        print("Error during backup scheduling:", e)


@app.route("/events")
def events():
    """Stream server-sent events to connected clients."""

    def stream():
        """Yield server-sent events from the queue to connected clients until disconnected."""
        q = queue.Queue()
        listeners.append(q)
        try:
            while True:
                msg = q.get()
                yield f"data: {msg}\n\n"
        except GeneratorExit:
            listeners.remove(q)

    return Response(stream(), mimetype="text/event-stream")


def notify_refresh():
    """Send refresh event to all connected clients."""
    for q in list(listeners):
        try:
            q.put_nowait("refresh")
        except Exception:
            pass


@app.route("/")
def index():
    """Render admin index page with script statuses and actions."""
    rows = []
    for script, sc_name in zip(AVAILABLE_SCRIPTS, SCRIPTS_NAMES):
        # script =  script
        status_info = script_status(script)
        status = status_info["status"]
        pid = status_info["pid"]

        if status == "running":
            action_url = url_for("stop_task", script_name=script)
            action_text = "Stop"
        else:
            action_url = url_for("start_task", script_name=script)
            action_text = "Start"

        rows.append({
            "name": sc_name,
            "status": status,
            "pid": pid,
            "action_url": action_url,
            "action_text": action_text,
        })

    return render_template("admin/index.html", rows=rows)


@app.route("/view-employees")
def view_employees():
    """Render employee list by matching IDs with names."""
    rows = []
    df_n = pd.read_csv(f"{constants.FACE_DIR_PATH}/name_file.csv")
    df_s = pd.read_csv(f"{constants.FACE_DIR_PATH}/known_faces.csv")
    df_s['Name'] = df_s['Name'].astype("str")
    df_n['employee_id'] = df_n['employee_id'].astype("str")
    for i in df_s['Name'].unique():
        if i.isdigit():
            i = str(int(i))
        name = df_n[df_n['employee_id'] == i]['employee_name'].values

        if name:
            name = name[0]
            rows.append({
                "employee_id": i, "employee_name": name
            })

    return render_template("admin/users.html", rows=rows)


@app.route("/delete-employee")
def remove_employees():
    """Remove an employee entry from known faces and refresh employee list."""
    employee_id = request.args.get('employee_id')
    df_s = pd.read_csv(f"{constants.FACE_DIR_PATH}/known_faces.csv")
    df_s = df_s[df_s['Name'] != employee_id]
    df_s.to_csv(f"{constants.FACE_DIR_PATH}/known_faces.csv", index=False)
    return redirect(url_for("view_employees"))


@app.route("/script/<script_name>/start")
def start_task(script_name):
    """Start a script task with optional environment variables and redirect."""
    # extra params if passed (for record video)
    n = request.args.get("n")
    name = request.args.get("name")

    env = None
    if n and name:
        env = {**dict(os.environ), "EMP_ID": n, "EMP_NAME": name}

    run_script(script_name, env=env)
    notify_refresh()
    return redirect(url_for("index"))


@app.route("/script/<script_name>/stop")
def stop_task(script_name):
    """Stop a running script task and redirect."""
    stop_script(script_name)
    notify_refresh()
    return redirect(url_for("index"))


@app.route("/script_status/<script_name>")
def check_status(script_name):
    """Return the running status of a specific script as JSON."""
    result = script_status(script_name)
    notify_refresh()
    return jsonify(result)


# --- Shortcuts for your specific flows ---

@app.route("/update_dataset", methods=["POST", "GET"])
def update_dataset():
    """Shortcut route to start the dataset update script."""
    return start_task("0_get_videos.py")


@app.route("/camera_checking", methods=["POST", "GET"])
def camera_checking():
    """Shortcut route to start the camera checking script."""
    return start_task("e_camera_checking.py")


@app.route("/prepare_data", methods=["POST", "GET"])
def prepare_data():
    """Shortcut route to start the data preparation script."""
    return start_task("a_prepare_data.py")


@app.route("/train", methods=["POST", "GET"])
def train():
    """Run training script in a background thread, reload models, and notify clients."""
    def task():
        """Execute training script, reload models, and notify clients when finished."""
        try:
            subprocess.run(["python", "b_train_all.py"], check=True)
            load_models()  # reload latest models
        finally:
            notify_refresh()  # refresh when finished

    threading.Thread(target=task, daemon=True).start()
    notify_refresh()
    return jsonify({"status": "started", "step": "train"})


@app.route("/recognition", methods=["POST", "GET"])
def recognition():
    """Shortcut route to start the recognition loop script."""
    return start_task("d_recognition_loop.py")


@app.route("/get-backup", methods=["POST", "GET"])
def get_backup():
    """Run backup process immediately and return status."""
    run_backup()
    return jsonify({"status": "backup executed"})


@app.route("/get-activities")
@auth_required
@log_route
def get_activities():
    """Fetch and return employee activity logs filtered by date."""
    date_str = request.args.get("date")  # expected format: DD-MM-YYYY
    tz = pytz.timezone(constants.TIME_ZONE)
    now = datetime.now(tz)

    if date_str:
        try:
            filter_date = datetime.strptime(date_str, "%d-%m-%Y").date()
        except ValueError:
            return jsonify({
                "success_flag": False,
                "status_code": 400,
                "message": "Invalid date format, use DD-MM-YYYY"
            })
    else:
        filter_date = now.date()
    df_o = load_and_prepare(f"{constants.IP_OP_BUFFER_PATH}/o_p.csv", "OUT")
    df_i = load_and_prepare(f"{constants.IP_OP_BUFFER_PATH}/i_p.csv", "IN")
    now = datetime.now(pytz.timezone(constants.TIME_ZONE))
    df = pd.concat([df_o, df_i], ignore_index=True)

    df = df[df['timed'].dt.date == filter_date]
    df = df[now - df['timed'] > timedelta(**constants.THRESHOLD_DICT)]
    df.sort_values(["timed"], inplace=True)
    df['LogDate'] = df['timed'].dt.strftime("%d-%m-%Y %H:%M:%S")
    df = df.rename(columns={"name": "EmployeeCode"})

    data = df[['LogDate', 'EmployeeCode', 'TypeOfCamera']].to_dict(
        orient="records")

    update_and_save(df_o, f"{constants.IP_OP_BUFFER_PATH}/o_p.csv")
    update_and_save(df_i, f"{constants.IP_OP_BUFFER_PATH}/i_p.csv")

    return jsonify({"success_flag": True, "status_code": 200,
                    "message": "Success", "data": data, "count": len(data)})


@app.route("/get-current-status")
@auth_required
@log_route
def get_current_status():
    """Return current status of employees: IN, OUT, or ABSENT."""
    date_str = request.args.get("date")
    tz = pytz.timezone(constants.TIME_ZONE)
    now = datetime.now(tz)

    if date_str:
        try:
            filter_date = datetime.strptime(date_str, "%d-%m-%Y").date()
        except ValueError:
            return jsonify({
                "success_flag": False,
                "status_code": 400,
                "message": "Invalid date format, use DD-MM-YYYY"
            })
    else:
        filter_date = now.date()
    op_dict = {"IN": {"employee_id": {}, "count": 0},
               "OUT": {"employee_id": {}, "count": 0},
               "ABSENT": {"employee_id": {}, "count": 0}}
    df_o = load_and_prepare(f"{constants.IP_OP_BUFFER_PATH}/o_p.csv", "OUT")
    df_i = load_and_prepare(f"{constants.IP_OP_BUFFER_PATH}/i_p.csv", "IN")
    now = datetime.now(pytz.timezone(constants.TIME_ZONE))
    df = pd.concat([df_o, df_i], ignore_index=True)
    df['name'] = df['name'].astype('str')
    df['name'] = df['name'].apply(lambda x: x.zfill(3) if x.isdigit() else x)
    df = df[df['timed'].dt.date == filter_date]
    df = df[now - df['timed'] > timedelta(**constants.THRESHOLD_DICT)]
    df.sort_values(["timed"], inplace=True)
    df['LogDate'] = df['timed'].dt.strftime("%d-%m-%Y %H:%M:%S")

    df = df.rename(columns={"name": "EmployeeCode"})
    df.sort_values(['timed'], inplace=True, ascending=False)
    df.drop_duplicates(inplace=True, subset=['EmployeeCode'])

    data = df[['LogDate', 'EmployeeCode', 'TypeOfCamera']].to_dict(
        orient="records")
    att_set = set([])

    for i in data:

        if i["EmployeeCode"].isdigit():
            i["EmployeeCode"] = f'{int(i["EmployeeCode"]):03d}'

        t = str(i["EmployeeCode"])

        op_dict[i["TypeOfCamera"]]["employee_id"][t] = i["LogDate"]
        op_dict[i["TypeOfCamera"]]["count"] += 1

        att_set.add(t)

    df_ts = pd.read_csv(f"{constants.FACE_DIR_PATH}/name_file.csv")
    df_ts['employee_id'] = df_ts['employee_id'].apply(lambda x: x.zfill(3) if x.isdigit() else x)
    df_ts = df_ts[df_ts['status'] == 'active']['employee_id']
    df_ts = set(df_ts)

    for i in df_ts:
        t = i
        if t.isdigit():
            t = f'{int(t):03d}'

        df_ts.discard(i)
        df_ts.add(t)
    for i in df_ts - att_set:
        t = i
        if t.isdigit():
            t = f'{int(t):03d}'

        op_dict["ABSENT"]["employee_id"][t] = "00-00-00 00:00:00"

        op_dict["ABSENT"]["count"] += 1

    return jsonify({"success_flag": True, "status_code": 200,
                    "message": "Success", "data": op_dict,
                    "count": op_dict['IN']['count'] + op_dict['OUT']['count'] + op_dict['ABSENT']['count']})


scheduler = BackgroundScheduler()
scheduler.add_job(call_backup_api, 'interval', days=1,
                  next_run_time=datetime.now())
scheduler.start()

if __name__ == "__main__":
    try:

        app.run(host="0.0.0.0", debug=True, threaded=True, port=8522)
    except KeyboardInterrupt:
        print("Shutting Down...")
    except Exception as exc:
        print(">>>>>", exc.args)
    finally:
        del app
        print(">>>>> Resources Released")
