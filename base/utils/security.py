from flask import request
from functools import wraps
import configparser
import logging

config = configparser.ConfigParser()

config.read("config.ini")

STORED_TOKEN = config["auth"]["token"]

# Setup logger
logger = logging.getLogger("flask_app_logger")
logger.setLevel(logging.INFO)

# File handler
file_handler = logging.FileHandler("logs/flask_access.log")
file_handler.setLevel(logging.INFO)

# Log format
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)


# Decorator to log route access
def log_route(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        log_data = {
            "url": "/"+request.url.split("/")[-1],
            "method": request.method,

        }
        logger.info(f"Route accessed: {f.__name__} | Data: {log_data}")
        return f(*args, **kwargs)

    return decorated_function


def auth_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get("Authorization")

        if not auth_header:
            return {
                "data": None, "success_flag": False, "status_code": 401,
                "message": "Authorization header missing"
            }

        try:
            incoming_token = auth_header.split(" ")[1]
        except IndexError:
            return {
                "data": None, "success_flag": False, "status_code": 401,
                "message": "Invalid token format"
            }

        if incoming_token != STORED_TOKEN:
            return {
                "data": None, "success_flag": False, "status_code": 403,
                "message": "Forbidden"
            }

        return f(*args, **kwargs)

    return decorated_function
