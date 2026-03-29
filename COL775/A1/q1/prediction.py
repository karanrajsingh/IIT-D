import os
import argparse
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from training import ResNet18, get_valid_data_loader

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--norm_type", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)

    return parser.parse_args()

class CustomDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.image_names = sorted(os.listdir(data_path))
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image_path = os.path.join(self.data_path, image_name)

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, image_name

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3) 
    ])

    loader = get_valid_data_loader(args.data_path)
    model = ResNet18(norm=args.norm_type)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model = model.to(device)
    model.eval()

    prediction_list = []
    with torch.no_grad():
        for images, image_names in loader:
            images = images.to(device)
            out = model(images)
            predictions = out.argmax(dim=1)

            for name, prediction in zip(image_names, predictions):
                prediction_list.append([name.cpu().item(), prediction.cpu().item()])

    os.makedirs(args.save_dir, exist_ok=True)
    df = pd.DataFrame(prediction_list, columns=["image_name", "predicted_label"])
    save_path = os.path.join(args.save_dir, "predictions.csv")
    df.to_csv(save_path, index=False)

if __name__ == "__main__":
    main()