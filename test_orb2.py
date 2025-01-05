import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# 이미지 폴더 경로
image_folder = "path_to_your_image_folder"

# 이미지 리스트 가져오기
image_files = sorted([
    os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.bmp'))
])

# 레퍼런스 이미지 설정
reference_image_path = image_files[0]
reference_image = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)

# ORB 생성
orb = cv2.ORB_create()

# BFMatcher 생성
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# 현재 이미지 인덱스
current_index = 1

# 좌측 상단과 우측 하단 좌표 저장 변수
region_start = None
region_end = None

# 영역을 지정하는 마우스 콜백 함수
def select_region(event, x, y, flags, param):
    global region_start, region_end

    if event == cv2.EVENT_LBUTTONDOWN:
        if region_start is None:
            region_start = (x, y)
            print(f"Top-left corner selected: {region_start}")
        elif region_end is None:
            region_end = (x, y)
            print(f"Bottom-right corner selected: {region_end}")

# 결과를 표시하는 함수
def display_matching(reference, current, matches, keypoints_ref, keypoints_cur):
    global result_image
    # 매칭 결과 시각화
    result_image = cv2.drawMatches(reference, keypoints_ref, current, keypoints_cur, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # 이미지 보여주기
    cv2.namedWindow("Feature Matching", cv2.WINDOW_NORMAL)
    cv2.imshow("Feature Matching", result_image)
    cv2.resizeWindow("Feature Matching", result_image.shape[1], result_image.shape[0])

# 마우스 클릭 콜백 함수
def on_mouse_click(event, x, y, flags, param):
    global current_index, region_start, region_end, result_image

    if event == cv2.EVENT_LBUTTONDOWN and current_index < len(image_files):
        if region_start is None or region_end is None:
            print("Please select the region first by clicking top-left and bottom-right corners.")
            return

        # 다음 이미지 읽기
        current_image_path = image_files[current_index]
        current_image = cv2.imread(current_image_path, cv2.IMREAD_GRAYSCALE)

        # 선택된 영역 자르기
        x1, y1 = region_start
        x2, y2 = region_end
        cropped_ref = reference_image[y1:y2, x1:x2]
        cropped_cur = current_image[y1:y2, x1:x2]

        # 특징점 추출
        ref_keypoints, ref_descriptors = orb.detectAndCompute(cropped_ref, None)
        cur_keypoints, cur_descriptors = orb.detectAndCompute(cropped_cur, None)

        # 매칭 수행
        if ref_descriptors is not None and cur_descriptors is not None:
            matches = bf.match(ref_descriptors, cur_descriptors)
            matches = sorted(matches, key=lambda x: x.distance)  # 거리 기준으로 정렬

            # 매칭 결과 표시
            display_matching(cropped_ref, cropped_cur, matches[:20], ref_keypoints, cur_keypoints)

        # 인덱스 증가
        current_index += 1
    elif current_index >= len(image_files):
        print("No more images to process.")

# OpenCV 윈도우 설정
cv2.namedWindow("Select Region")
cv2.setMouseCallback("Select Region", select_region)
cv2.namedWindow("Feature Matching", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Feature Matching", on_mouse_click)

print("1. Select the region by clicking the top-left and bottom-right corners in the 'Select Region' window.")
print("2. Click on the 'Feature Matching' window to process the next image.")

# 첫 번째 이미지를 보여줌
cv2.imshow("Select Region", reference_image)
cv2.imshow("Feature Matching", np.zeros_like(reference_image))
cv2.waitKey(0)

cv2.destroyAllWindows()
