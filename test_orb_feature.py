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

# 매칭 시각화
def draw_matches(ref_img, ref_keypoints, tgt_img, tgt_keypoints, matches):
    # 두 이미지를 나란히 연결
    ref_img_color = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2BGR)
    tgt_img_color = cv2.cvtColor(tgt_img, cv2.COLOR_GRAY2BGR)
    combined_img = np.hstack((ref_img_color, tgt_img_color))
    h, w = ref_img.shape

    # 매칭 시각화 ramom으로 10개만 표시
    np.random.seed(42)
    matches = np.array(matches)
    np.random.shuffle(matches)
    matches = matches[:10]
    for m in matches:
        pt1 = tuple(map(int, ref_keypoints[m[0]]))
        pt2 = tuple(map(int, tgt_keypoints[m[1]]))
        pt2 = (pt2[0] + w, pt2[1])  # 두 번째 이미지의 좌표는 오른쪽으로 이동
        cv2.line(combined_img, pt1, pt2, (0, 255, 0), 1)

    return combined_img
# 매칭 수행
def match_descriptors(desc1, desc2):
    matches = []
    for i, d1 in enumerate(desc1):
        best_distance = float('inf')
        best_match = -1
        for j, d2 in enumerate(desc2):
            distance = np.sum(d1 != d2)  # Hamming distance
            if distance < best_distance:
                best_distance = distance
                best_match = j
        if best_distance < 50:  # 임계값
            matches.append((i, best_match))
    return matches
# 테스트
if __name__ == "__main__":
    ref_img = cv2.imread("left01.jpg", cv2.IMREAD_GRAYSCALE)
    tgt_img = cv2.imread("left02.jpg", cv2.IMREAD_GRAYSCALE)
    # ORB 특징점 추출
    ref_keypoints, ref_descriptors = orb(ref_img)
    tgt_keypoints, tgt_descriptors = orb(tgt_img)
    # 디스크립터 매칭
    matches = match_descriptors(ref_descriptors, tgt_descriptors)
    # 매칭 결과 시각화
    combined_img = draw_matches(ref_img, ref_keypoints, tgt_img, tgt_keypoints, matches)
    cv2.imshow("ORB Feature Matching", combined_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
