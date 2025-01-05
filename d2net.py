import cv2
import os
import numpy as np
from skimage import exposure
import torch
from d2_net import D2Net
from d2_net.utils import preprocess_image
from d2_net.matching import match_descriptors

# 이미지 폴더 경로
image_folder = "path_to_your_image_folder"

# 이미지 리스트 가져오기
image_files = sorted([
    os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))
])

# D2-Net 초기화
model = D2Net(model_file="d2_tf.pth", use_cuda=torch.cuda.is_available())

# 레퍼런스 이미지 설정
reference_image_path = image_files[0]
reference_image = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)

# Illumination Normalization (CLAHE)
def apply_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)

# 전처리: CLAHE 적용
reference_image = apply_clahe(reference_image)

# D2-Net 특징 추출 함수
def extract_d2net_features(image, model):
    image = preprocess_image(image, preprocessing="torch")
    with torch.no_grad():
        keypoints, descriptors = model.run(image)
    return keypoints, descriptors

# 레퍼런스 이미지에서 특징 추출
ref_keypoints, ref_descriptors = extract_d2net_features(reference_image, model)

# 매칭 결과를 시각화하는 함수
def display_matching(reference, current, matches, keypoints_ref, keypoints_cur):
    reference_image_rgb = cv2.cvtColor(reference, cv2.COLOR_GRAY2BGR)
    current_image_rgb = cv2.cvtColor(current, cv2.COLOR_GRAY2BGR)
    match_image = cv2.drawMatches(
        reference_image_rgb, keypoints_ref, current_image_rgb, keypoints_cur, matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    cv2.imshow("Feature Matching", match_image)

# 현재 이미지 인덱스
current_index = 1

# 마우스 클릭 콜백 함수
def on_mouse_click(event, x, y, flags, param):
    global current_index

    if event == cv2.EVENT_LBUTTONDOWN and current_index < len(image_files):
        # 다음 이미지 읽기
        current_image_path = image_files[current_index]
        current_image = cv2.imread(current_image_path, cv2.IMREAD_GRAYSCALE)

        # 전처리: CLAHE 적용
        current_image = apply_clahe(current_image)

        # 현재 이미지에서 특징 추출
        cur_keypoints, cur_descriptors = extract_d2net_features(current_image, model)

        # 매칭 수행
        matches = match_descriptors(ref_descriptors, cur_descriptors)

        # 매칭 결과 시각화
        display_matching(reference_image, current_image, matches, ref_keypoints, cur_keypoints)

        # 인덱스 증가
        current_index += 1
    elif current_index >= len(image_files):
        print("No more images to process.")

# OpenCV 윈도우 설정
cv2.namedWindow("Feature Matching", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Feature Matching", on_mouse_click)

print("Click on the window to process the next image.")

# 첫 번째 이미지를 보여줌
cv2.imshow("Feature Matching", reference_image)
cv2.waitKey(0)

cv2.destroyAllWindows()
