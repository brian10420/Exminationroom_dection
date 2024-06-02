import torch
import torch.nn as nn
from torchvision.transforms import ToTensor, Compose
from transformers import Swinv2Model
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
import os

# Define the directories for the original images and labeled masks using relative paths
Ori_image = r"D:\NCUE_lab\Vision\Train_ori"
Label_image = r"D:\NCUE_lab\Vision\Train_Label"

# Map RGB values to class names
class_map = {
    (192, 64, 0): 0,  # Human
    (64, 192, 0): 1,  # Background
    (0, 64, 192): 2   # Hands
}

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
        self.masks = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")
        
        # Convert RGB mask to class indices
        mask = np.array(mask)
        class_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int64)
        for rgb, index in class_map.items():
            class_mask[(mask == rgb).all(axis=-1)] = index

        mask = torch.from_numpy(class_mask)
        image = ToTensor()(image)

        return image, mask

transform = ToTensor()  # Only conversion to tensor is needed

dataset = SegmentationDataset(Ori_image, Label_image, transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class SwinV2Segmentation(nn.Module):
    def __init__(self, num_classes):
        super(SwinV2Segmentation, self).__init__()
        self.swin_v2 = Swinv2Model.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")
        self.upsample = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)
        self.segmentation_head = nn.Conv2d(768, num_classes, kernel_size=1)

    def forward(self, x):
        outputs = self.swin_v2(x).last_hidden_state
        outputs = outputs.permute(0, 2, 1).contiguous()
        outputs = outputs.view(outputs.size(0), -1, int(outputs.size(-1)**0.5), int(outputs.size(-1)**0.5))
        outputs = self.upsample(outputs)
        outputs = self.segmentation_head(outputs)
        return outputs

# Define the number of segmentation classes
num_classes = 3
model = SwinV2Segmentation(num_classes).to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Function to train the model
def train_model(model, dataloader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks.long())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader)}')

# Function to save the trained model
def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f'Model saved to {path}')

# Train the model
train_model(model, dataloader, criterion, optimizer)
model_save_path = 'swinv2_segmentation_model.pth'
save_model(model, model_save_path)
