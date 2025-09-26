BGR_COLOR_BLUE = (255, 0, 0)
BGR_COLOR_RED = (0, 0, 255)
BGR_COLOR_GREEN = (0, 255, 0)
TRUE_BOOL = True
FALSE_BOOL = False
FIRST_INDEX = 0
LAST_INDEX = -1
UNKNOWN_STR = "Unknown"
IP_FPS = 60
OP_NO_FRAMES = 8
NUMBER_OF_CAMERAS = 4
OP_FPS = 4
FRAME_QUEUE_SIZE = 100
RESULT_QUEUE_SIZE = 200

CHECK_FREQUENCY = 2
CHECK_AT_FREQUENCY = 1
FX_RATIO = 0.5
FY_RATIO = 0.5
BACKUP_DAYS = 7
MODEL_DIR_PATH = "static/models"
VIDEO_DIR_PATH = "static/videos"
DATA_BACKUP_PATH = "static/backup/dataset"
VIDEO_MOVE_DIR_PATH = "static/done_videos"
FACE_DIR_PATH = "static/known_faces"
TEST_IP_DIR_PATH = "static/test_images/ip"
TEST_OP_DIR_PATH = "static/test_images/op"
ANALYSIS_DIR_PATH = "static/analysis"
IP_OP_BUFFER_PATH = "static/ip_op_buffers"
MAX_DETECTION = 100
TIME_ZONE = 'Asia/Kolkata'
THRESHOLD_DICT = {'minutes': 1, 'seconds': 0, 'hours': 0, 'days': 0}
NUMBER_OF_FEATURES = 512
KNN_THRESHOLD = 0.6
MLP_THRESHOLD = 0.8
RECOGNIZE_THRESHOLD = 0.85
USER_NAME = "admin"
PASSWORD = "1234"
IP_ADDRESS = "192.168.1.1"
PORT_NO = "554"
CAMERA_LINK_PATH = "cam/realmonitor"
CAMERA_IP_1_NO = "19"
CAMERA_IP_2_NO = "7"
CAMERA_OP_1_NO = "21"
CAMERA_OP_2_NO = "20"
ENV_CAPTURE_OPTIONS_KEY = \
    "OPENCV_FFMPEG_CAPTURE_OPTIONS|max_delay;500000|buffer_size;1024000"
CAPTURE_OPTIONS = "rtsp_transport;tcp"
INPUT_CAMERA_ADDRESS1 = f"rtsp://{USER_NAME}:{PASSWORD}@" \
                       f"{IP_ADDRESS}:{PORT_NO}/{CAMERA_LINK_PATH}?" \
                       f"channel={CAMERA_IP_1_NO}&subtype=0"
INPUT_CAMERA_ADDRESS2 = f"rtsp://{USER_NAME}:{PASSWORD}@" \
                       f"{IP_ADDRESS}:{PORT_NO}/{CAMERA_LINK_PATH}?" \
                       f"channel={CAMERA_IP_2_NO}&subtype=0"
INPUT_CAMERA_NAME = "input"
OUTPUT_CAMERA_ADDRESS1 = f"rtsp://{USER_NAME}:{PASSWORD}@" \
                        f"{IP_ADDRESS}:{PORT_NO}/{CAMERA_LINK_PATH}?" \
                        f"channel={CAMERA_OP_1_NO}&subtype=0"
OUTPUT_CAMERA_ADDRESS2 = f"rtsp://{USER_NAME}:{PASSWORD}@" \
                        f"{IP_ADDRESS}:{PORT_NO}/{CAMERA_LINK_PATH}?" \
                        f"channel={CAMERA_OP_2_NO}&subtype=0"
OUTPUT_CAMERA_NAME = "output"
EMBEDDINGS_FILE_NAME = "known_faces.csv"
FRAME_LS = [i + 1 for i in range(0, IP_FPS, IP_FPS // OP_FPS)]
COLUMNS = ["name", "probability", "timed"]
WORKERS = 4
KEEP_RECORD_LIMIT_DICT = {'minutes': 0, 'seconds': 0, 'hours': 0, 'days': 0,
                          'months': 1, 'years': 0}
