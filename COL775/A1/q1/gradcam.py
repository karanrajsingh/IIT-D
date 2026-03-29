import torch
from training import ResNet18, get_valid_data_loader
import random
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np
import matplotlib.pyplot as plt
import os

def unnormalize(img_tensor):
    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    img = std*img + mean
    img = np.clip(img, 0, 1)
    return img

def save_image(img, path):
    plt.figure(figsize=(3, 3))
    plt.imshow(img)
    plt.axis("off")
    plt.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close()

def generate_overlay(image_tensor, label, cam, device):
    input_tensor = image_tensor.unsqueeze(0).to(device)
    input_tensor.requires_grad_(True)
    targets = [ClassifierOutputTarget(label)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
    img = unnormalize(image_tensor)
    visualization = show_cam_on_image(img, grayscale_cam, use_rgb=True)
    return visualization

def main():

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    path = "./checkpoints/batch_128/norm_IN/model.pt" # best model path
    checkpoint = torch.load(path, map_location = device)
    classes = 100
    norm = "IN" # best norm type
    model = ResNet18(num_classes=classes, norm=norm)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    correct = {key : [] for key in range(classes)}
    incorrect = {key : [] for key in range(classes)}

    path = "./data/val"
    val_loader = get_valid_data_loader(path)
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        predictions = outputs.argmax(dim=1)
        for i in range(images.size(0)):
            prediction = predictions[i].item()
            label = labels[i].item()
            if prediction == label:
                correct[label].append(images[i].to("cpu"))
            else:
                incorrect[label].append(images[i].to("cpu"))

    for label in range(classes):
        correct_list = random.sample(correct[label], min(10, len(correct[label])))
        incorrect_list = random.sample(incorrect[label], min(10, len(incorrect[label])))
        correct[label] = correct_list
        incorrect[label] = incorrect_list

    cam = GradCAM(model=model, target_layers=[model.conv4_3])

    base_correct = "./visual/correct"
    base_incorrect = "./visual/incorrect"

    for label in range(classes):
        os.makedirs(os.path.join(base_correct, str(label)), exist_ok=True)
        os.makedirs(os.path.join(base_incorrect, str(label)), exist_ok=True)    
    
    for label in range(classes):
        for index, image in enumerate(correct[label]):
            visualization = generate_overlay(image, label, cam, device)
            save_path_1 = os.path.join(base_correct, str(label), f"correct_{index}.png")
            save_image(visualization, save_path_1)
            image = image.permute(1, 2, 0)
            save_path_2 = os.path.join(base_correct, str(label), f"image_{index}.png")
            save_image(image, save_path_2)
        for index, image in enumerate(incorrect[label]):
            visualization = generate_overlay(image, label, cam, device)
            save_path_1 = os.path.join(base_incorrect, str(label), f"incorrect_{index}.png")
            save_image(visualization, save_path_1)
            image = image.permute(1, 2, 0)
            save_path_2 = os.path.join(base_incorrect, str(label), f"image_{index}.png")
            save_image(image, save_path_2)

if __name__ == "__main__":
    main()