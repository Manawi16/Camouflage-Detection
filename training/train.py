import torch
import sys
import os, json
sys.path.append(os.path.abspath(".."))
from torch.utils.data import DataLoader
import torch.nn.functional as F
from dataset import CamouflageDataset
from model.SINet import SINet_ResNet50

# Load dataset
dataset = CamouflageDataset("dataset/images", "dataset/masks")
loader = DataLoader(dataset, batch_size=2, shuffle=True)

# Load model
model = SINet_ResNet50()
model.load_state_dict(torch.load("../weights/SINet_40.pth", map_location="cpu"))
model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_history = []
for epoch in range(10):  # small training
    total_loss = 0

    for images, masks in loader:
        pred_sm, pred_im = model(images)

        loss = F.binary_cross_entropy_with_logits(pred_im, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    loss_history.append(total_loss)
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# Save logs
log_data = {
    "epochs": 10,
    "loss": loss_history,
    "initial_loss": loss_history[0],
    "final_loss": loss_history[-1],
    "dataset_size": len(dataset)
}

with open("../training_logs.json", "w") as f:
    json.dump(log_data, f)

# Save new weights
torch.save(model.state_dict(), "../weights/custom_sinet.pth")