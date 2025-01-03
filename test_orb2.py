import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# 이미지 폴더 경로
image_folder = "path_to_your_image_folder"

# 이미지 리스트 가져오기
image_files = sorted([
    os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))
])

# 레퍼런스 이미지 설정
reference_image_path = image_files[0]
reference_image = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)

# ORB 생성
orb = cv2.ORB_create()

# 레퍼런스 이미지에서 특징점과 서술자 추출
ref_keypoints, ref_descriptors = orb.detectAndCompute(reference_image, None)

# BFMatcher 생성
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# 현재 이미지 인덱스
current_index = 1

# 결과를 표시하는 함수
def display_matching(reference, current, matches, keypoints_ref, keypoints_cur):
    # 매칭 결과 시각화
    result_image = cv2.drawMatches(reference, keypoints_ref, current, keypoints_cur, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # 이미지 보여주기
    cv2.imshow("Feature Matching", result_image)

# 마우스 클릭 콜백 함수
def on_mouse_click(event, x, y, flags, param):
    global current_index

    if event == cv2.EVENT_LBUTTONDOWN and current_index < len(image_files):
        # 다음 이미지 읽기
        current_image_path = image_files[current_index]
        current_image = cv2.imread(current_image_path, cv2.IMREAD_GRAYSCALE)

        # 현재 이미지의 특징점과 서술자 추출
        cur_keypoints, cur_descriptors = orb.detectAndCompute(current_image, None)

        # 매칭 수행
        matches = bf.match(ref_descriptors, cur_descriptors)
        matches = sorted(matches, key=lambda x: x.distance)  # 거리 기준으로 정렬

        # 매칭 결과 표시
        display_matching(reference_image, current_image, matches[:20], ref_keypoints, cur_keypoints)

        # 인덱스 증가
        current_index += 1
    elif current_index >= len(image_files):
        print("No more images to process.")

# OpenCV 윈도우 설정
cv2.namedWindow("Feature Matching")
cv2.setMouseCallback("Feature Matching", on_mouse_click)

print("Click on the window to process the next image.")

# 첫 번째 이미지를 보여줌
cv2.imshow("Feature Matching", reference_image)
cv2.waitKey(0)

cv2.destroyAllWindows()
