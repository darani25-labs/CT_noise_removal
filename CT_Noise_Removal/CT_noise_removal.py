import cv2
import matplotlib.pyplot as plt
import numpy as np

# ---------- Step 1: Read the CT Scan Image ----------
img = cv2.imread("images/t_scan_noisy.jpg", cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Error: CT scan image not found. Check the path!")
    exit()

# ---------- Step 2: Apply Median Filtering ----------
# Median filter removes salt-and-pepper (random) noise
filtered_img = cv2.medianBlur(img, 5)  # Kernel size = 5 (try 3 or 7 too)

# ---------- Step 3: Display Input vs Filtered Output ----------
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Original / Noisy CT Scan")
plt.imshow(img, cmap='gray')
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Filtered CT Scan (Median Filter)")
plt.imshow(filtered_img, cmap='gray')
plt.axis("off")

plt.tight_layout()
plt.show()

# ---------- Step 4: Save the Filtered Image ----------
cv2.imwrite("images/ct_scan_filtered.jpg", filtered_img)
print("✅ Filtered image saved as 'images/ct_scan_filtered.jpg'")

