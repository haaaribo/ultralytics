import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, "/Users/yusangmin/Documents/GitHub/ultralytics2")  # 로컬 레포 경로

from ultralytics.nn.modules.c2f_ppa import C2f_PPA  # unpickler가 클래스 찾도록

# 🧠 ROI 분할 함수
def split_image(img, roi_index, Row=640, Col=640):
    img_row, img_col, k = img.shape
    img_temp = cv2.resize(img, (img_col - (img_col % Col), img_row - (img_row % Row)))
    img_row, img_col, k = img_temp.shape
    roi_r, roi_c = img_row // Row, img_col // Col

    if roi_index > roi_r * roi_c or roi_index <= 0:
        print('roi index error')
        return False

    r_idx = (roi_index - 1) // roi_c
    c_idx = (roi_index - 1) % roi_c
    return img[r_idx * Row:((r_idx + 1) * Row), c_idx * Col:((c_idx + 1) * Col)]

# 🐔 모델 로드
model = YOLO("/Users/yusangmin/Documents/폐사체연구/best_yolo8_sp.pt")  # 최적 모델 가중치 불러오기
model.to("mps")  # GPU 사용 설정
# 🎥 비디오 캡처 객체 생성
video_path = "/Users/yusangmin/Documents/폐사체연구/육계 폐사체 탐지 데이터/폐사체동영상/0_8_IPC1_20220912080906.mp4"  # ← 여기에 영상 경로 입력
cap = cv2.VideoCapture(video_path)

# ✅ 분석할 ROI index (예: 1~4 사이의 영역)
roi_index = 2

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 🔍 ROI 블록 자르기
    roi = split_image(frame, roi_index)
    if roi is False:
        continue

    # 🧪 YOLO 추론
    results = model.predict(source=roi, show=False, save=False, verbose=False)

    # 📌 결과 시각화 (matplotlib 또는 cv2로 표시)
    result_img = results[0].plot()

    # 화면 출력 (코랩에서는 matplotlib 사용)
    plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

cap.release()


