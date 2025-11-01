# FIX 1: Thêm thư viện numpy
import numpy as np
from skimage.io import imread 
import os 
# Giả định 2 hàm này nằm trong file 'hog_extract.py'
from hog_extract import visualize_hsv_histograms, extract_color_histogram
import cv2
import matplotlib.pyplot as plt

file_path = r"D:\Project\2025-2026\computer_vision\data_use\40\00040_00011_00029.png"
def extract_color_histogram(image, bins=(8, 8, 8)):
    """Trích xuất đặc trưng Color Histogram từ ảnh HSV"""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    hist_h = cv2.calcHist([hsv], [0], None, [bins[0]], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [bins[1]], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [bins[2]], [0, 256])

    hist_h = cv2.normalize(hist_h, hist_h).flatten()
    hist_s = cv2.normalize(hist_s, hist_s).flatten()
    hist_v = cv2.normalize(hist_v, hist_v).flatten()

    color_features = np.concatenate([hist_h, hist_s, hist_v])
    return color_features

# === HÀM TRỰC QUAN HÓA MỚI ===
def visualize_hsv_histograms(image_rgb, bins=(8, 8, 8)):
    """
    Hàm này tính toán lại và vẽ 3 histogram H, S, V
    """
    # 1. Chuyển sang HSV
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    
    # 2. Tính toán 3 histogram (chưa chuẩn hóa để xem số đếm)
    hist_h = cv2.calcHist([hsv], [0], None, [bins[0]], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [bins[1]], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [bins[2]], [0, 256])

    # 3. Vẽ
    plt.figure(figsize=(15, 10))

    # ---- Hàng 1: Ảnh gốc và các kênh H, S, V ----
    plt.subplot(2, 4, 1)
    plt.imshow(image_rgb)
    plt.title("Ảnh gốc (RGB)")
    plt.axis('off')

    plt.subplot(2, 4, 2)
    plt.imshow(hsv[:, :, 0], cmap='hsv') # Kênh H (Hue)
    plt.title("Kênh H (Màu sắc)")
    plt.axis('off')

    plt.subplot(2, 4, 3)
    plt.imshow(hsv[:, :, 1], cmap='gray') # Kênh S (Độ rực rỡ)
    plt.title("Kênh S (Bão hòa)")
    plt.axis('off')

    plt.subplot(2, 4, 4)
    plt.imshow(hsv[:, :, 2], cmap='gray') # Kênh V (Độ sáng)
    plt.title("Kênh V (Độ sáng)")
    plt.axis('off')

    # ---- Hàng 2: Biểu đồ của H, S, V ----
    plt.subplot(2, 3, 4) # Biểu đồ H
    plt.bar(range(bins[0]), hist_h.flatten(), color='red')
    plt.title("Biểu đồ HUE (8 bins)")
    plt.xlabel("Bin màu")
    plt.ylabel("Số lượng Pixel")

    plt.subplot(2, 3, 5) # Biểu đồ S
    plt.bar(range(bins[1]), hist_s.flatten(), color='green')
    plt.title("Biểu đồ SATURATION (8 bins)")
    plt.xlabel("Bin độ rực rỡ")
    plt.ylabel("Số lượng Pixel")

    plt.subplot(2, 3, 6) # Biểu đồ V
    plt.bar(range(bins[2]), hist_v.flatten(), color='blue')
    plt.title("Biểu đồ VALUE (8 bins)")
    plt.xlabel("Bin độ sáng")
    plt.ylabel("Số lượng Pixel")

    plt.tight_layout()
    plt.show()

if not os.path.exists(file_path):
    print(f"LỖI: Không tìm thấy file tại đường dẫn: {file_path}")
    print("Vui lòng kiểm tra lại đường dẫn và dấu gạch chéo.")
else:
    print(f"Đang tải ảnh từ: {file_path}")
    image_to_test = imread(file_path)

    # 3. Chuyển về 8-bit (0-255) vì cv2.cvtColor cần
    if image_to_test.dtype == np.float64 or image_to_test.dtype == np.float32:
        # Nếu là float (0-1), nhân 255
        image_to_test_8bit = (image_to_test * 255).astype(np.uint8)
    else:
        # Nếu đã là uint8 (0-255), dùng trực tiếp
        image_to_test_8bit = image_to_test.astype(np.uint8)
        
    # ---
    # FIX 2: Xử lý Kênh màu (Rất quan trọng)
    # Đảm bảo ảnh luôn là 3 kênh (RGB) trước khi gửi vào hàm
    # ---
    
    if image_to_test_8bit.ndim == 2:
        # Ảnh là Grayscale (H, W) -> Chuyển sang RGB
        print("Phát hiện ảnh Grayscale (1 kênh), đang chuyển sang RGB (3 kênh)...")
        image_to_test_8bit = cv2.cvtColor(image_to_test_8bit, cv2.COLOR_GRAY2RGB)
        
    elif image_to_test_8bit.ndim == 3:
        # Ảnh có 3 chiều (H, W, C)
        if image_to_test_8bit.shape[2] == 4:
            # Kênh thứ 3 có 4 giá trị -> RGBA
            print("Phát hiện ảnh RGBA (4 kênh), đang chuyển sang RGB (3 kênh)...")
            image_to_test_8bit = cv2.cvtColor(image_to_test_8bit, cv2.COLOR_RGBA2RGB)
        # Nếu shape[2] == 3, thì nó đã là RGB, không cần làm gì.
        elif image_to_test_8bit.shape[2] == 3:
            print("Ảnh đã là RGB (3 kênh).")

    # 4. Gọi hàm trực quan hóa
    # Hàm visualize_hsv_histograms (trong file .py kia) phải
    # tự import matplotlib.pyplot và tự gọi plt.show()
    visualize_hsv_histograms(image_to_test_8bit)

    # 5. Gọi hàm trích xuất đặc trưng
    features_24 = extract_color_histogram(image_to_test_8bit)
    print(f"\nVector đặc trưng màu cuối cùng (24 số): \n{features_24}")
    print(f"Kích thước: {features_24.shape}")