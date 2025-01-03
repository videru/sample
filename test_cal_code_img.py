import cv2
import numpy as np
import glob
import os

def extract_and_match_features(img1, img2):
    # ORB 생성
    orb = cv2.ORB_create(nfeatures=5000)

    # 특징점 추출
    keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

    # 디스크립터 확인
    if descriptors1 is None or descriptors2 is None:
        print("Descriptors are not generated.")
        return keypoints1, keypoints2, []

    # FLANN 매칭 설정
    index_params = dict(algorithm=6,  # FLANN_INDEX_LSH
                        table_number=6,  # 해시 테이블 수
                        key_size=12,  # 해시 키 크기
                        multi_probe_level=1)  # 탐색 레벨
    search_params = dict(checks=50)  # 탐색 횟수

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # KNN 매칭 수행
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # 좋은 매칭 점 필터링
    good_matches = []
    for m_n in matches:
        if len(m_n) == 2:  # 두 개의 매칭 결과가 있을 때만 처리
            m, n = m_n
            if m.distance < 0.9 * n.distance:
                good_matches.append(m)

    return keypoints1, keypoints2, good_matches


# 기본 카메라 내부 파라미터 설정 (가상값, 실제 사용 시 보정 필요)
camera_matrix = np.array([
    [800, 0, 320],
    [0, 800, 240],
    [0, 0, 1]
], dtype=np.float32)

dist_coeffs = np.zeros((4, 1))  # 왜곡 계수 (가정: 없음)

object_points = np.array([
    [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1],
    [1, 0, 1], [0, 1, 1], [1, 1, 1], [0.5, 0.5, 0], [0.5, 0.5, 1]
], dtype=np.float32)

# 폴더 내 이미지 파일 불러오기
image_folder = "images"  # 이미지 폴더 경로
image_files = sorted(glob.glob(os.path.join(image_folder, "*.jpg")))

if len(image_files) < 2:
    print("At least two images are required for this process.")
    exit()

# 기준 이미지 설정
ref_image = cv2.imread(image_files[0], cv2.IMREAD_GRAYSCALE)
if ref_image is None:
    print("Failed to load the reference image.")
    exit()

# 나머지 이미지를 순차적으로 처리
for image_file in image_files[1:]:
    target_image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
    if target_image is None:
        print(f"Failed to load image: {image_file}")
        continue

    # 특징점 추출 및 매칭
    keypoints1, keypoints2, good_matches = extract_and_match_features(ref_image, target_image)

    # 매칭된 3D-2D 점 생성
    matched_object_points = []
    matched_image_points = []

    for match in good_matches:
        if match.queryIdx < len(object_points) and match.trainIdx < len(keypoints2):
            matched_object_points.append(object_points[match.queryIdx])  # 3D 점
            matched_image_points.append(keypoints2[match.trainIdx].pt)  # 2D 점

    # numpy 배열로 변환
    matched_object_points = np.array(matched_object_points, dtype=np.float32)
    matched_image_points = np.array(matched_image_points, dtype=np.float32)

    # 배열 크기 확인
    print(f"Matched object points shape: {matched_object_points.shape}")
    print(f"Matched image points shape: {matched_image_points.shape}")

    # PnP 계산
    if len(matched_object_points) >= 6:  # 최소 6개의 대응점 필요
        success, rvec, tvec = cv2.solvePnP(
            matched_object_points, matched_image_points, camera_matrix, None, flags=cv2.SOLVEPNP_ITERATIVE
        )

        if success:
            # 회전 벡터 -> 회전 행렬 변환
            rotation_matrix, _ = cv2.Rodrigues(rvec)

            # 회전 각도 계산
            yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
            pitch = np.arcsin(-rotation_matrix[2, 0])
            roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])

            # 출력: 카메라의 변경된 위치 및 회전
            print(f"Image: {image_file}")
            print(f"Camera Position (Translation Vector): {tvec.ravel()}")
            print(f"Yaw: {np.degrees(yaw):.2f}°, Pitch: {np.degrees(pitch):.2f}°, Roll: {np.degrees(roll):.2f}°")
    else:
        print(f"Not enough matches for PnP calculation in image: {image_file}")
