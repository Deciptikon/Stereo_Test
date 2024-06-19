import cv2
import numpy as np

# Загрузка видео
video_path = 'your_video.mp4'
cap = cv2.VideoCapture(video_path)

# Создание объекта для вычисления карты диспаратности
stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Разделение кадра на левую и правую части
    height, width, _ = frame.shape
    half_width = width // 2
    
    left_frame = frame[:, :half_width, :]
    right_frame = frame[:, half_width:, :]
    
    # Преобразование в градации серого
    gray_left = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)
    
    # Вычисление карты диспаратности
    disparity = stereo.compute(gray_left, gray_right)
    
    # Преобразование карты диспаратности в карту глубины
    depth_map = cv2.reprojectImageTo3D(disparity, Q=None, handleMissingValues=True)
    
    # Отображение левой картинки и карты глубины
    cv2.imshow('Left Frame', left_frame)
    depth_map_visual = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imshow('Depth Map', depth_map_visual)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()