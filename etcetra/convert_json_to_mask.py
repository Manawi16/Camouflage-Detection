import json
import numpy as np
import cv2
from PIL import Image, ImageDraw

# ---- CHANGE THIS ----
json_path = "download.json"   # your json file
output_path = "Camouflage-4.png"  # output mask
# ---------------------

with open(json_path) as f:
    data = json.load(f)

img = Image.open(data["imagePath"])
width, height = img.size

mask = Image.new("L", (width, height), 0)
draw = ImageDraw.Draw(mask)

for shape in data["shapes"]:
    points = [tuple(point) for point in shape["points"]]
    draw.polygon(points, outline=255, fill=255)

mask = np.array(mask)

cv2.imwrite(output_path, mask)

print("Mask saved as:", output_path)