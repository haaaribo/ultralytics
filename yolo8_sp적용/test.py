import sys
import os
# Ensure your local ultralytics2 repo is on the path
sys.path.insert(0, "/Users/yusangmin/Documents/GitHub/ultralytics2")

import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
# Import the custom C2f_PPA from the models package
from ultralytics.nn.modules.c2f_ppa import C2f_PPA

# ROI 분할 함수
def split_image(img, roi_index, Row=640, Col=640):
    img_row, img_col, _ = img.shape
    new_col = img_col - (img_col % Col)
    new_row = img_row - (img_row % Row)
    img_temp = cv2.resize(img, (new_col, new_row))
    roi_r, roi_c = new_row // Row, new_col // Col

    if roi_index < 1 or roi_index > roi_r * roi_c:
        print('roi index error')
        return None

    r_idx = (roi_index - 1) // roi_c
    c_idx = (roi_index - 1) % roi_c
    return img_temp[r_idx*Row:(r_idx+1)*Row, c_idx*Col:(c_idx+1)*Col]

# 모델 로드
model = YOLO("/Users/yusangmin/Documents/폐사체연구/best_yolo8_sp.pt")
model.to("mps")

# 비디오 캡처
video_path = "/Users/yusangmin/Documents/폐사체연구/육계 폐사체 탐지 데이터/폐사체동영상/0_8_IPC1_20220912080906.mp4"
cap = cv2.VideoCapture(video_path)

roi_index = 2

while True:
    ret, frame = cap.read()
    if not ret:
        break

    roi = split_image(frame, roi_index)
    if roi is None:
        continue

    # 추론
    results = model.predict(source=roi, show=False, save=False, verbose=False)
    result_img = results[0].plot()

    # 출력
    plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show(block=False)
    plt.pause(0.001)

cap.release()
cv2.destroyAllWindows()
