import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.filters import sobel_h, sobel_v
from skimage.feature import hog
from skimage import exposure, transform, data  # <-- Thêm
import os  # <-- Thêm

# ========== BƯỚC 1: ĐỌC & TIỀN XỬ LÝ ẢNH ==========
file_path = r"D:\Project\2025-2026\computer_vision\data_use\40\00040_00011_00029.png"
image_size = (128, 128)  # <-- Kích thước chuẩn để đảm bảo code chạy

if os.path.exists(file_path):
    # Dùng imread(..., as_gray=True) sẽ gọn hơn
    image_gray = imread(file_path, as_gray=True)
    title_prefix = "Ảnh của bạn"
else:
    print(f"Cảnh báo: Không tìm thấy {file_path}. Dùng ảnh 'astronaut' làm ví dụ.")
    image_gray = rgb2gray(data.astronaut())
    title_prefix = "Ảnh ví dụ"

# FIX 2: Resize ảnh về kích thước chuẩn (chia hết cho 8)
image = transform.resize(image_gray, image_size)

# FIX 1: Chỉ tạo MỘT Figure 3x3
plt.figure(figsize=(16, 16)) 

plt.subplot(3, 3, 1)
plt.imshow(image, cmap='gray')
plt.title(f"1️⃣ {title_prefix} (Grayscale {image_size})")
plt.axis("off")

# ========== BƯỚC 2: TÍNH GRADIENT X, Y ==========
Gx = sobel_h(image)
Gy = sobel_v(image)

plt.subplot(3, 3, 2)
plt.imshow(Gx, cmap='gray')
plt.title("2️⃣ Gradient theo trục X")
plt.axis("off")

plt.subplot(3, 3, 3)
plt.imshow(Gy, cmap='gray')
plt.title("3️⃣ Gradient theo trục Y")
plt.axis("off")

# ========== BƯỚC 3: ĐỘ LỚN & HƯỚNG GRADIENT ==========
magnitude = np.sqrt(Gx**2 + Gy**2)
orientation = np.rad2deg(np.arctan2(Gy, Gx)) % 180  # hướng 0–180°

plt.subplot(3, 3, 4)
plt.imshow(magnitude, cmap='inferno')
plt.title("4️⃣ Độ lớn Gradient")
plt.axis("off")

plt.subplot(3, 3, 5)
plt.imshow(orientation, cmap='hsv')
plt.title("5️⃣ Hướng Gradient (0°–180°)")
plt.axis("off")

H, W = image.shape

# ================= BƯỚC 4: CHIA CELL (8×8) =================
cell_size = (8, 8)
num_bins = 9
bin_width = 180 // num_bins

n_cells_y = H // cell_size[0]
n_cells_x = W // cell_size[1]
hog_cells = np.zeros((n_cells_y, n_cells_x, num_bins), dtype=np.float32)

for i in range(n_cells_y):
    for j in range(n_cells_x):
        y0, y1 = i * cell_size[0], (i + 1) * cell_size[0]
        x0, x1 = j * cell_size[1], (j + 1) * cell_size[1]
        cell_magnitude = magnitude[y0:y1, x0:x1].ravel()
        cell_orientation = orientation[y0:y1, x0:x1].ravel()

        hist = np.zeros(num_bins)
        for k in range(cell_magnitude.size):
            angle = cell_orientation[k]
            mag = cell_magnitude[k]
            bin_idx = int(angle // bin_width) % num_bins
            hist[bin_idx] += mag
        hog_cells[i, j, :] = hist

# Hiển thị ảnh biểu diễn magnitude trung bình mỗi cell
cell_magnitude_map = np.sqrt(np.sum(hog_cells**2, axis=2))

# FIX 1 (tiếp): Xóa plt.figure() và đổi subplot(2,2,1)
plt.subplot(3, 3, 6) 
plt.imshow(cell_magnitude_map, cmap='inferno')
plt.title("6️⃣ Độ mạnh TB mỗi Cell (8x8)")
plt.axis("off")

# ================= BƯỚC 5: GOM BLOCK 2×2 CELL =================
cells_per_block = (2, 2)
n_blocks_y = n_cells_y - cells_per_block[0] + 1
n_blocks_x = n_cells_x - cells_per_block[1] + 1
block_raw = np.zeros((n_blocks_y, n_blocks_x, num_bins * 4), dtype=np.float32)

for y in range(n_blocks_y):
    for x in range(n_blocks_x):
        block = hog_cells[y:y + 2, x:x + 2, :].ravel()
        block_raw[y, x, :] = block

block_intensity_map = np.sqrt(np.sum(block_raw**2, axis=2))

# FIX 1 (tiếp): Đổi subplot(2,2,2)
plt.subplot(3, 3, 7)
plt.imshow(block_intensity_map, cmap='plasma')
plt.title("7️⃣ Gom Block (2x2 cell)")
plt.axis("off")

# ================= BƯỚC 6: CHUẨN HÓA BLOCK (L2-Hys) =================
eps = 1e-5
block_normed = np.zeros_like(block_raw)

for y in range(n_blocks_y):
    for x in range(n_blocks_x):
        b = block_raw[y, x, :]
        b = b / np.sqrt(np.sum(b ** 2) + eps ** 2) # L2-norm
        b = np.clip(b, 0, 0.2)                     # Clip
        b = b / np.sqrt(np.sum(b ** 2) + eps ** 2) # L2-norm (lần nữa)
        block_normed[y, x, :] = b

block_norm_intensity = np.sqrt(np.sum(block_normed**2, axis=2))

# FIX 1 (tiếp): Đổi subplot(2,2,3)
plt.subplot(3, 3, 8)
plt.imshow(block_norm_intensity, cmap='magma')
plt.title("8️⃣ Block đã chuẩn hóa L2-Hys")
plt.axis("off")

# ================= BƯỚC 7: GHÉP TOÀN BỘ BLOCK THÀNH VECTOR =================
# FIX 3: Xóa biểu đồ 7 (vì nó giống hệt biểu đồ 8)
hog_vector_manual = block_normed.ravel()


# ================= BƯỚC 9 (Mới): SO SÁNH VỚI HÀM CỦA SKIMAGE =================
fd_skimage, hog_image_skimage = hog(
    image,
    orientations=num_bins,
    pixels_per_cell=cell_size,
    cells_per_block=cells_per_block,
    block_norm='L2-Hys',
    visualize=True,
    feature_vector=True
)
hog_image_rescaled = exposure.rescale_intensity(hog_image_skimage, in_range=(0, 10))

plt.subplot(3, 3, 9)
plt.imshow(hog_image_rescaled, cmap='gray')
plt.title("9️⃣ CHECK: Kết quả từ `skimage.hog`")
plt.axis("off")

# --- Hiển thị và In kết quả ---
plt.tight_layout()
plt.show() # <-- Chỉ gọi 1 lần duy nhất

print("="*70)
print(f"Ảnh gốc: {image.shape}")
print(f"Số cell (Y, X): ({n_cells_y}, {n_cells_x})")
print(f"Số block (Y, X): ({n_blocks_y}, {n_blocks_x})")
print(f"Số đặc trưng mỗi block (2*2*9): {cells_per_block[0]*cells_per_block[1]*num_bins}")
print("="*70)
print("KÍCH THƯỚC VECTOR CUỐI CÙNG (CHECK):")
print(f"➡️ Code thủ công của bạn:  {hog_vector_manual.shape}")
print(f"➡️ Hàm `skimage.hog`:       {fd_skimage.shape}")
print("="*70)

if hog_vector_manual.shape == fd_skimage.shape:
    print("✅ XUẤT SẮC! Kích thước 2 vector khớp nhau!")
else:
    print("❌ Lỗi: Kích thước 2 vector không khớp.")