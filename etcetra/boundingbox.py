import cv2

mask = cv2.imread("Camouflage-4.png", 0)
img = cv2.imread("download.jpg")

# threshold (just to be safe)
_, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

# find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

bbox_img = img.copy()

for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(bbox_img, (x, y), (x+w, y+h), (0,255,0), 2)

cv2.imwrite("gt_bbox.png", bbox_img)