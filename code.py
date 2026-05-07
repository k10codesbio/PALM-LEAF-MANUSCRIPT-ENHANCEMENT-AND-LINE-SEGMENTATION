import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
img = cv2.imread('manuscript.jpg')
if img is None:
    raise ValueError("Image not loaded!")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# CLAHE
clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
enhanced = clahe.apply(gray)

# Bilateral Filter
denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)

# Background normalization
background = cv2.GaussianBlur(denoised, (51,51), 0)
normalized = cv2.divide(denoised, background, scale=255)

# Strong smoothing to remove texture
blur = cv2.GaussianBlur(normalized, (9,9), 0)

# Otsu threshold (global, stable)
_, thresh = cv2.threshold(
    blur,
    0,
    255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)

# Remove small noise
kernel = np.ones((3,3), np.uint8)
clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

# Connect text horizontally (VERY IMPORTANT)
kernel_line = cv2.getStructuringElement(cv2.MORPH_RECT, (40,3))
clean = cv2.dilate(clean, kernel_line, iterations=1)

horizontal_sum = np.sum(clean, axis=1)

threshold = np.max(horizontal_sum) * 0.2
binary = horizontal_sum > threshold

lines = []
start = None

for i in range(len(binary)):
    if binary[i] and start is None:
        start = i
    elif not binary[i] and start is not None:
        end = i

        # NEW: split inside region if needed
        segment = horizontal_sum[start:end]

        # find local minima (gaps between lines)
        for j in range(1, len(segment)-1):
            if segment[j] < segment[j-1] and segment[j] < segment[j+1]:
                # split here
                split_y = start + j
                lines.append((start, split_y))
                start = split_y

        lines.append((start, end))
        start = None

if start is not None:
    lines.append((start, len(binary)))

output = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

for (y1, y2) in lines:
    cv2.rectangle(output, (0, y1), (output.shape[1], y2), (0,255,0), 2)

plt.figure(figsize=(12,8))

plt.subplot(2,2,1)
plt.imshow(gray, cmap='gray')
plt.title("Original")

plt.subplot(2,2,2)
plt.imshow(normalized, cmap='gray')
plt.title("Enhanced")

plt.subplot(2,2,3)
plt.imshow(clean, cmap='gray')
plt.title("Binary for Segmentation")

plt.subplot(2,2,4)
plt.imshow(output)
plt.title("Line Segmentation Output")

plt.tight_layout()
plt.show()
