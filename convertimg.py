import cv2
import numpy as np
import os

# Đọc ảnh
image_path = "tsjpg.jpg"  # Thay bằng đường dẫn đến ảnh của bạn
image = cv2.imread(image_path)

# Chuyển sang ảnh xám
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Tăng độ tương phản và làm sáng
alpha = 1.5  # Độ tương phản (1.0-3.0)
beta = 50    # Độ sáng (0-100)
adjusted = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)

# Áp dụng ngưỡng (threshold) để tạo ảnh đen trắng
_, binary = cv2.threshold(adjusted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Loại bỏ nhiễu bằng bộ lọc Gaussian (tùy chọn)
denoised = cv2.GaussianBlur(binary, (3, 3), 0)

# Lưu ảnh đã xử lý thành PNG
output_image_path = "vn.png"
cv2.imwrite(output_image_path, denoised)

print(f"Đã lưu ảnh xử lý: {output_image_path}")