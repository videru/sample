import numpy as np
import cv2
from scipy.ndimage import gaussian_filter

# FAST 코너 검출
def fast_corner_detection(img, threshold=50):
    corners = []
    img = img.astype(np.float32)
    h, w = img.shape

    # 16개 주변 픽셀 상대 좌표 (FAST)
    circle_offsets = [
        (-3, 0), (-3, 1), (-2, 2), (-1, 3), (0, 3), (1, 3), (2, 2), (3, 1),
        (3, 0), (3, -1), (2, -2), (1, -3), (0, -3), (-1, -3), (-2, -2), (-3, -1)
    ]

    for y in range(3, h - 3):
        for x in range(3, w - 3):
            p = img[y, x]
            circle_values = [img[y + dy, x + dx] for dx, dy in circle_offsets]

            brighter = sum([v > p + threshold for v in circle_values])
            darker = sum([v < p - threshold for v in circle_values])

            if brighter >= 9 or darker >= 9:
                corners.append((x, y))
    return corners

# 해리스 응답 계산
def harris_response(img, corners, k=0.04):
    Ix = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

    Ixx = gaussian_filter(Ix ** 2, sigma=1)
    Iyy = gaussian_filter(Iy ** 2, sigma=1)
    Ixy = gaussian_filter(Ix * Iy, sigma=1)

    responses = []
    for x, y in corners:
        Sxx = Ixx[y, x]
        Syy = Iyy[y, x]
        Sxy = Ixy[y, x]

        # 해리스 응답 함수
        R = (Sxx * Syy - Sxy ** 2) - k * (Sxx + Syy) ** 2
        responses.append((x, y, R))
    return responses

# 디스크립터 생성 (BRIEF)
def compute_brief_descriptor(img, keypoints, patch_size=31):
    descriptors = []
    for x, y in keypoints:
        if x < patch_size // 2 or y < patch_size // 2 or \
           x + patch_size // 2 >= img.shape[1] or y + patch_size // 2 >= img.shape[0]:
            continue

        patch = img[y - patch_size // 2:y + patch_size // 2 + 1,
                    x - patch_size // 2:x + patch_size // 2 + 1]

        # 샘플링 패턴 생성
        np.random.seed(42)
        pairs = np.random.randint(0, patch_size * patch_size, (256, 2))
        descriptor = [1 if patch.flat[p1] < patch.flat[p2] else 0 for p1, p2 in pairs]
        descriptors.append(descriptor)
    return np.array(descriptors, dtype=np.uint8)

# ORB 구현
def orb(img, num_features=500, threshold=100, response_threshold=0.5):
    # Grayscale 변환
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # FAST 코너 검출
    corners = fast_corner_detection(img, threshold=threshold)

    # 해리스 응답 기반 코너 선택
    responses = harris_response(img, corners)
    responses = [resp for resp in responses if resp[2] > response_threshold]  # 응답 값 필터링
    responses = sorted(responses, key=lambda x: x[2], reverse=True)[:num_features]
    keypoints = [(x, y) for x, y, _ in responses]

    # 중복 키포인트 제거
    unique_keypoints = []
    seen_positions = set()
    for x, y in keypoints:
        position = (int(x), int(y))
        if position not in seen_positions:
            unique_keypoints.append((x, y))
            seen_positions.add(position)

    # BRIEF 디스크립터 생성
    descriptors = compute_brief_descriptor(img, unique_keypoints)

    return unique_keypoints, descriptors

# 테스트
if __name__ == "__main__":
    img = cv2.imread("left01.jpg", cv2.IMREAD_GRAYSCALE)
    keypoints, descriptors = orb(img)
    # 흑백 이미지를 컬러로 변환 (BGR)
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # 결과 시각화
    for x, y in keypoints:
        cv2.circle(img_color, (x, y), 3, (255, 0, 0), -1)

    cv2.imshow("ORB Keypoints", img_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
