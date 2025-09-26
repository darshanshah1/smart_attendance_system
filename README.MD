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

## Installation

```bash
# Clone repository
git clone https://github.com/your-username/your-repo.git
cd your-repo

# Create virtual environment
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt

