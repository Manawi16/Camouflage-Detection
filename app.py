from flask import Flask, render_template, request, url_for, session, redirect
import torch
import cv2
import numpy as np
import json
from PIL import Image
import os
import torch.nn.functional as F
from model.SINet import SINet_ResNet50
from flask import send_from_directory
app = Flask(__name__)
app.secret_key = "secret123"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(BASE_DIR)
model = SINet_ResNet50()
model.load_state_dict(torch.load("weights/best_sinet.pth", map_location="cpu"))
model.eval()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/demo")
def demo():
    return render_template("demo.html") 

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]

    filepath = "static/input.png"
    file.save(filepath)

    image = Image.open(filepath)

    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    original = img.copy()

    img = cv2.resize(img, (352, 352))
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)

    with torch.no_grad():
        _, cam = model(img)

    cam = F.interpolate(cam, size=original.shape[:2], mode='bilinear')
    cam = cam.sigmoid().cpu().numpy().squeeze()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    mask = (cam * 255).astype(np.uint8)

    cv2.imwrite("static/mask.png", mask)

    heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)
    cv2.imwrite("static/overlay.png", overlay)

    _, thresh = cv2.threshold(mask, 120, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bbox_img = original.copy()
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(bbox_img, (x, y), (x+w, y+h), (0,255,0), 2)

    cv2.imwrite("static/bbox.png", bbox_img)

    return render_template("result.html", 
        input_img="static/input.png",
        mask="static/mask.png",
        heatmap="static/overlay.png",
        bbox="static/bbox.png"
    )
@app.route('/dataset/images/<path:filename>')
def dataset_images(filename):
    return send_from_directory(
        os.path.join(BASE_DIR, 'training', 'dataset', 'images'),
        filename
    )

@app.route('/dataset/masks/<path:filename>')
def dataset_masks(filename):
    return send_from_directory(
        os.path.join(BASE_DIR, 'training', 'dataset', 'masks'),
        filename
    )

@app.route("/dataset")
def dataset():

    image_dir = os.path.join(BASE_DIR, "training", "dataset", "images")
    mask_dir = os.path.join(BASE_DIR, "training", "dataset", "masks")

    images = sorted(os.listdir(image_dir))

    data = []

    for img in images:
        base = os.path.splitext(img)[0]
        mask_name = base + ".png"

        if os.path.exists(os.path.join(mask_dir, mask_name)):
            data.append({
                "img": img,
                "mask": mask_name
            })

    page = int(request.args.get("page", 1))
    per_page = 6

    start = (page - 1) * per_page
    end = start + per_page

    paginated_data = data[start:end]

    total_pages = (len(data) + per_page - 1) // per_page

    return render_template(
        "dataset.html",
        data=paginated_data,
        page=page,
        total_pages=total_pages
    )
@app.route("/training")
def training():
    with open("training_logs.json") as f:
        logs = json.load(f)

    return render_template("training.html", logs=logs)

@app.route("/comparison")
def comparison():
    return render_template("comparison.html")

if __name__ == "__main__":
    app.run(debug=True)