import cv2
import numpy as np

ISCALIBRATE = False
calibr = None

# Загрузка видео
video_path = 'stereo_test.mp4'
cap = cv2.VideoCapture(video_path)

def calibrate(left_image, right_image):
    # Преобразование изображений в градации серого
    gray_left = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

    # Используем SIFT для поиска особых точек и их дескрипторов
    sift = cv2.SIFT_create()
    keypoints_left, descriptors_left = sift.detectAndCompute(gray_left, None)
    keypoints_right, descriptors_right = sift.detectAndCompute(gray_right, None)

    # Используем BFMatcher для сопоставления дескрипторов
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors_left, descriptors_right, k=2)

    # Применяем ratio test, чтобы получить хорошие сопоставления
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Получаем координаты хороших сопоставленных точек на обоих изображениях
    points_left = np.float32([keypoints_left[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    points_right = np.float32([keypoints_right[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Вычисляем матрицу преобразования (affine или perspective)
    # Например, используем findHomography для вычисления перспективного преобразования
    H, _ = cv2.findHomography(points_right, points_left, cv2.RANSAC)

    # Применяем преобразование к правому изображению
    height, width = gray_left.shape
    right_aligned = cv2.warpPerspective(right_image, H, (width, height))

    # Визуализация результатов
    cv2.imshow('Left Image', left_image)
    cv2.imshow('Right Image', right_image)
    cv2.imshow('Right Aligned', right_aligned)
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):  # При нажатии пробела (код клавиши ' ')
            break
    
    cv2.destroyAllWindows()
    return right_aligned

# Создание объекта для вычисления карты диспаратности
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=16*16,
    blockSize=30,
    P1=8*3*5**2,
    P2=32*3*5**2,
    disp12MaxDiff=1,
    uniquenessRatio=15,
    speckleWindowSize=100,
    speckleRange=32,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Разделение кадра на левую и правую части
    height, width, _ = frame.shape
    half_width = width // 2
    
    left_frame = frame[:, :half_width, :]
    right_frame = frame[:, half_width:, :]
    
    if not ISCALIBRATE:
        calibr = calibrate(left_frame, right_frame)
        ISCALIBRATE = True
    
    
    # Преобразование в градации серого
    gray_left = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)
    
    # Вычисление карты диспаратности
    disparity = stereo.compute(gray_left, gray_right)
    
    # Преобразование карты диспаратности в карту глубины
    depth_map = np.zeros_like(disparity, dtype=np.float32)
    mask = disparity > 0
    depth_map[mask] = (1.0 / disparity[mask]) * 255.0
    
    # Нормализация для отображения в градации серого
    depth_map_visual = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Отображение левой картинки и карты глубины
    cv2.imshow('Left Frame', left_frame)
    cv2.imshow('Depth Map', depth_map_visual)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

