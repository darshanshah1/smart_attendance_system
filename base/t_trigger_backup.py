"""
t_trigger_backup.py
"""
import os
import pandas as pd
from datetime import datetime, timedelta
import pytz

try:
    from base.utils import constants
except Exception:
    from utils import constants


def run_backup():
    """
    Run periodic backup of input/output logs based on retention policy.

    Author
    --------------
    Name: Darshan H Shah

    Behavior:
        - Loads IN and OUT log CSVs with timestamps.
        - Filters records older than BACKUP_DAYS threshold.
        - Moves old records into a dated backup directory.
        - Retains recent records in the original CSVs.
        - Logs success or error during backup execution.


    Globals Modified:
        None

    Returns:
        None
    """

    print("BACKUP STARTED IN BACKGROUND")
    try:
        tz = pytz.timezone(constants.TIME_ZONE)
        time_now = datetime.now(tz)

        df_i = pd.read_csv(f"{constants.IP_OP_BUFFER_PATH}/i_p.csv",
                           parse_dates=["timed"])
        df_o = pd.read_csv(f"{constants.IP_OP_BUFFER_PATH}/o_p.csv",
                           parse_dates=["timed"])
        cutoff = time_now - timedelta(days=constants.BACKUP_DAYS)

        df_bi = df_i[df_i["timed"] < cutoff]
        df_bo = df_o[df_o["timed"] < cutoff]
        df_i = df_i[df_i["timed"] >= cutoff]
        df_o = df_o[df_o["timed"] >= cutoff]
        if df_bi.shape[0] or df_bo.shape[0]:

            dir_name = time_now.strftime("%y%m%d")
            backup_dir = os.path.join(constants.DATA_BACKUP_PATH, dir_name)
            os.makedirs(backup_dir, exist_ok=True)

            if df_bi.shape[0]:
                df_bi.to_csv(os.path.join(backup_dir, "i_p.csv"), index=False)
            if df_bo.shape[0]:
                df_bo.to_csv(os.path.join(backup_dir, "o_p.csv"), index=False)

            df_i.to_csv(f"{constants.IP_OP_BUFFER_PATH}/i_p.csv", index=False)
            df_o.to_csv(f"{constants.IP_OP_BUFFER_PATH}/o_p.csv", index=False)
        else:
            raise Exception("NO NEW DATA FOUND")
    except Exception as exc:
        print(f"ERROR DURING BACKUP: {exc}")
    else:
        print(f"BACKUP COMPLETED SUCCESSFULLY {time_now.date()}")


if __name__ == "__main__":
    run_backup()
