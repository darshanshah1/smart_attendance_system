# Smart Attendance and Monitoring System

A Flask-based application for employee attendance and monitoring using face recognition.  
It provides real-time detection, recognition, activity tracking, and automated backups.

---

## Features

- Real-time **face detection and recognition** powered by [InsightFace](https://github.com/deepinsight/insightface).
- **Script management**:
  - Record video
  - Prepare dataset
  - Train models
  - Run recognition
  - Camera health check
- **Employee management**: view and remove employees.
- **Activity monitoring**: daily logs and current status dashboards.
- **Model training**: KNN, MLP, and FAISS index with auto-reload.
- **Automated backups** with retention policy.
- REST **APIs** for activities, current status, and backup triggers.

---

## Tech Stack

- **Backend**: Flask  
- **Face Recognition**: InsightFace, FAISS  
- **ML Models**: scikit-learn (KNN, MLP)  
- **Database**: CSV-based logs  
- **Scheduling**: APScheduler  
- **Utilities**: Pandas, Joblib, OpenCV  

---

## Installation & Run

You can run the system using the provided script:

```bash
chmod +x run.sh
./run.sh

```
## Manual Installation
```bash
# Clone repository
git clone https://github.com/your-username/your-repo.git
cd your-repo

# Create virtual environment
python3 -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install onnxruntime-gpu==1.16.3 --extra-index-url https://download.pytorch.org/whl/cu117

# Run server
flask run --host=0.0.0.0 --port=8521
```
## API Endpoints

| Endpoint                | Method | Description                          |
| ----------------------- | ------ | ------------------------------------ |
| `/`                     | GET    | Admin dashboard with script controls |
| `/events`               | GET    | Stream refresh events (SSE)          |
| `/view-employees`       | GET    | List all employees                   |
| `/delete-employee`      | GET    | Remove employee by ID                |
| `/script/<name>/start`  | GET    | Start a script                       |
| `/script/<name>/stop`   | GET    | Stop a script                        |
| `/script_status/<name>` | GET    | Check status of script               |
| `/update_dataset`       | POST   | Shortcut to dataset update script    |
| `/prepare_data`         | POST   | Shortcut to data preparation script  |
| `/train`                | POST   | Run training and reload models       |
| `/recognition`          | POST   | Start recognition loop               |
| `/camera_checking`      | POST   | Check camera health                  |
| `/get-backup`           | POST   | Run backup immediately               |
| `/get-activities`       | GET    | Fetch activity logs                  |
| `/get-current-status`   | GET    | Get IN/OUT/ABSENT status summary     |

## API Output Example
Using check_api, results are displayed as:
```bash
> ABSENT COUNT = 25 | 129, 143, 259, 305, 398, 429, 436, 445, 523, 527, 560,
 585, 627, 680, 687, 693, 696, 706, 713, 716, 727, 731, 737, 744, IN_45
> IN COUNT = 12 | 522, 750, 675, 602, 440, 739, 009, 262, 361, IN_0, IN_38, IN_55
> OUT COUNT = 6 | 007, 669, 188, 606, 739, 361
ACTIVITIES COUNT = 17
```
This reflects real-time employee status:

> IN → currently inside.
OUT → detected exit.
ABSENT → no entry or exit recorded for the day.
ACTIVITIES COUNT → total activity records processed.

## Data Logging
All recognition events are appended into CSV logs. Example entries:
```bash
522    input    91.32%    2025-09-26 15:25:17+05:30
750    input    87.14%    2025-09-26 15:25:59+05:30
675    input    86.95%    2025-09-26 15:32:33+05:30
007    output   87.73%    2025-09-26 15:33:45+05:30
669    output   85.49%    2025-09-26 15:36:53+05:30
669    input    86.29%    2025-09-26 15:37:32+05:30
188    output   86.64%    2025-09-26 15:39:53+05:30
606    output   88.42%    2025-09-26 15:40:56+05:30
739    output   87.63%    2025-09-26 15:42:44+05:30
663    output   87.95%    2025-09-26 15:43:41+05:30
602    input    90.36%    2025-09-26 15:45:06+05:30
440    input    86.37%    2025-09-26 15:49:09+05:30
739    input    88.27%    2025-09-26 15:53:01+05:30
361    output   88.82%    2025-09-26 16:12:59+05:30
009    input    85.17%    2025-09-26 16:19:19+05:30
262    input    85.78%    2025-09-26 16:20:53+05:30
361    input    85.68%    2025-09-26 16:28:50+05:30
```
Each row contains:
<Employee ID> | <Direction> | <Confidence> | <Timestamp>

##Backup Policy

Keeps logs for the last N days (configured in constants.BACKUP_DAYS).
Older logs are moved into a dated folder inside DATA_BACKUP_PATH.

## Folder Structure
```php
.
├── app.py                     # Main Flask app
├── run.sh                     # Setup and run script
├── base/                      # Core modules
│   ├── c_recognize.py         # Face recognition logic
│   ├── t_trigger_backup.py    # Backup trigger logic
│   └── utils/                 # Utility functions and constants
├── templates/                 # HTML templates
└── static/                    # Static files (known_faces, models, etc.)
```
