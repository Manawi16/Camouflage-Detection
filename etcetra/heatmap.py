import cv2

# load image + mask
img = cv2.imread("download.jpg")
mask = cv2.imread("Camouflage-4.png", 0)

# create heatmap
heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)

# overlay
overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

cv2.imwrite("gt_heatmap.png", overlay)