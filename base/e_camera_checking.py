"""
e_camera_checking.py
"""
import cv2
import sys
try:
    from base.utils import constants
except Exception:
    from utils import constants

count = 0
capi = cv2.VideoCapture(constants.INPUT_CAMERA_ADDRESS2)
capo = cv2.VideoCapture(constants.OUTPUT_CAMERA_ADDRESS2)
print(f"Every {constants.OP_FPS*4}th Frame checked")
try:

    while True:
    
        reti, framei = capi.read()
        reto, frameo = capo.read()

        if reti and reto:
            count += 1
            if count % constants.OP_FPS != 0:
                continue
            framei = cv2.cvtColor(framei, cv2.COLOR_BGR2GRAY)
            framei = cv2.resize(framei, (0, 0), fx=0.3, fy=0.3)

            frameo = cv2.cvtColor(frameo, cv2.COLOR_BGR2GRAY)
            frameo = cv2.resize(frameo, (0, 0), fx=0.2, fy=0.2)

            cv2.imshow('IP Camera Feed', framei)
            cv2.imshow('OP Camera Feed', frameo)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

finally:
    capi.release()
    capo.release()
    cv2.destroyAllWindows()
    sys.exit(0)
