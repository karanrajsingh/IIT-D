import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from sklearn.metrics import accuracy_score, f1_score
import time
import matplotlib.pyplot as plt
import json

class NoNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()

    def forward(self, x):
        return x

class BatchNorm(nn.Module):
    def __init__(self, channels, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_channels = channels
        self.eps = eps
        self.momentum = momentum

        self.gamma = nn.Parameter(torch.ones(1,channels,1,1))
        self.beta = nn.Parameter(torch.zeros(1,channels,1,1))
        
        self.register_buffer("running_mean", torch.zeros(1,channels,1,1))
        self.register_buffer("running_var", torch.ones(1,channels,1,1))

    def forward(self, x):
        if self.training:
            mean = x.mean(dim=(0,2,3), keepdim=True)
            var = x.var(dim=(0,2,3), keepdim=True, unbiased=False)
            self.running_mean.mul_(1-self.momentum).add_(self.momentum*mean)
            self.running_var.mul_(1-self.momentum).add_(self.momentum*var)
        else:
            mean = self.running_mean
            var = self.running_var
        x_hat = (x-mean)/torch.sqrt(var+self.eps)
        out = self.gamma*x_hat + self.beta
        return out

    
class InstanceNorm(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1,channels,1,1))
        self.beta = nn.Parameter(torch.zeros(1,channels,1,1))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=(2,3), keepdim=True)
        var  = x.var(dim=(2,3), keepdim=True, unbiased=False)
        x_hat = (x-mean)/torch.sqrt(var+self.eps)
        out = self.gamma*x_hat + self.beta
        return out
    
class BatchInstanceNorm(nn.Module):
    def __init__(self, channels, eps=1e-5, momentum=0.1, rho=0.5):
        super().__init__()
        self.batch_norm = BatchNorm(channels, eps, momentum)
        self.instance_norm = InstanceNorm(channels, eps)
        self.rho = nn.Parameter(torch.ones(1,channels,1,1)*rho)

    def forward(self, x):
        rho = torch.clamp(self.rho, 0, 1)
        return rho*self.batch_norm(x) + (1-rho)*self.instance_norm(x)

class LayerNorm(nn.Module): 
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1,channels,1,1))
        self.beta = nn.Parameter(torch.zeros(1,channels,1,1))

    def forward(self, x):
        mean = x.mean(dim=(1,2,3), keepdim=True)
        var  = x.var(dim=(1,2,3), keepdim=True, unbiased=False)
        x_hat = (x-mean)/torch.sqrt(var+self.eps)
        out = self.gamma*x_hat + self.beta
        return out

class GroupNorm(nn.Module):
    def __init__(self, channels, num_groups=16, eps=1e-5):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1,channels,1,1))
        self.beta = nn.Parameter(torch.zeros(1,channels,1,1))

    def forward(self, x):
        n, c, h, w = x.shape
        G = self.num_groups
        x = x.view(n, self.num_groups, c//self.num_groups, h,w)
        mean = x.mean(dim=(2,3,4), keepdim=True)
        var  = x.var(dim=(2,3,4), keepdim=True, unbiased=False)
        x_hat = (x-mean)/torch.sqrt(var+self.eps)
        x_hat = x_hat.view(n,c,h,w)
        return self.gamma*x_hat + self.beta
    

def get_norm(channels, norm="Baseline"):
    if norm == "Baseline":
        print(f"Normalization scheme: {norm}")
        return nn.BatchNorm2d(channels)
    elif norm == "BN":
        print(f"Normalization scheme: {norm}")
        return BatchNorm(channels)
    elif norm == "NN":
        print(f"Normalization scheme: {norm}")
        return NoNorm(channels)
    elif norm == "IN":
        print(f"Normalization scheme: {norm}")
        return InstanceNorm(channels)
    elif norm == "BIN":
        print(f"Normalization scheme: {norm}")
        return BatchInstanceNorm(channels)
    elif norm == "LN":
        print(f"Normalization scheme: {norm}")
        return LayerNorm(channels)
    elif norm == "GN":
        print(f"Normalization scheme: {norm}")
        return GroupNorm(channels)

class ResNet18(nn.Module):
    def __init__(self, dropout=0.2, num_classes=100, norm = "Baseline"):
        super().__init__()
        self.num_classes = num_classes
        self.activation = nn.LeakyReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        
        # L0
        self.conv_0 = nn.Conv2d(in_channels=3, kernel_size=7, out_channels=64, padding=3, stride=2)
        self.norm_0 = get_norm(64, norm)
        self.maxpool_0 = nn.MaxPool2d(kernel_size=3, padding=1, stride=2)
        
        #L1:1
        self.conv1_0 = nn.Conv2d(in_channels=64, kernel_size=3, out_channels=64, stride=1, padding=1)
        self.norm_1_0 = get_norm(64, norm)
        # activation
        self.conv1_1 = nn.Conv2d(in_channels=64, kernel_size=3, out_channels=64, stride=1, padding=1)
        self.norm_1_1 = get_norm(64, norm)
        # residual
        # activation

        #L1:2
        self.conv1_2 = nn.Conv2d(in_channels=64, kernel_size=3, out_channels=64, stride=1, padding=1)
        self.norm_1_2 = get_norm(64, norm)
        # activation
        self.conv1_3 = nn.Conv2d(in_channels=64, kernel_size=3, out_channels=64, stride=1, padding=1)
        self.norm_1_3 = get_norm(64, norm)
        # residual
        # activation


        #L2:1
        self.conv2_0 = nn.Conv2d(in_channels=64, kernel_size=3, out_channels=128, stride=2, padding=1)
        self.norm2_0 = get_norm(128, norm)
        # activation
        self.conv2_1 = nn.Conv2d(in_channels=128, kernel_size=3, out_channels=128, stride=1, padding=1)
        self.norm2_1 = get_norm(128, norm)
        self.res2 = nn.Conv2d(in_channels=64, kernel_size=1, out_channels=128, stride=2, bias=False)
        # activation

        #L2:2
        self.conv2_2 = nn.Conv2d(in_channels=128, kernel_size=3, out_channels=128, stride=1, padding=1)
        self.norm2_2 = get_norm(128, norm)
        # activation
        self.conv2_3 = nn.Conv2d(in_channels=128, kernel_size=3, out_channels=128, stride=1, padding=1)
        self.norm2_3 = get_norm(128, norm)
        # activation


        #L3:1
        self.conv3_0 = nn.Conv2d(in_channels=128, kernel_size=3, out_channels=256, stride=2, padding=1)
        self.norm3_0 = get_norm(256, norm)
        # activation
        self.conv3_1 = nn.Conv2d(in_channels=256, kernel_size=3, out_channels=256, stride=1, padding=1)
        self.norm3_1 = get_norm(256, norm)
        self.res3 = nn.Conv2d(in_channels=128, kernel_size=1, out_channels=256, stride=2, bias=False)
        # activation

        #L3:2
        self.conv3_2 = nn.Conv2d(in_channels=256, kernel_size=3, out_channels=256, stride=1, padding=1)
        self.norm3_2 = get_norm(256, norm)
        # activation
        self.conv3_3 = nn.Conv2d(in_channels=256, kernel_size=3, out_channels=256, stride=1, padding=1)
        self.norm3_3 = get_norm(256, norm)
        # activation


        #L4:1
        self.conv4_0 = nn.Conv2d(in_channels=256, kernel_size=3, out_channels=512, stride=2, padding=1)
        self.norm4_0 = get_norm(512, norm)
        # activation
        self.conv4_1 = nn.Conv2d(in_channels=512, kernel_size=3, out_channels=512, stride=1, padding=1)
        self.norm4_1 = get_norm(512, norm)
        self.res4 = nn.Conv2d(in_channels=256, kernel_size=1, out_channels=512, stride=2, bias=False)
        # activation

        #L4:2
        self.conv4_2 = nn.Conv2d(in_channels=512, kernel_size=3, out_channels=512, stride=1, padding=1)
        self.norm4_2 = get_norm(512, norm)
        # activation
        self.conv4_3 = nn.Conv2d(in_channels=512, kernel_size=3, out_channels=512, stride=1, padding=1)
        self.norm4_3 = get_norm(512, norm)
        # activation


        #GAP
        self.gap = nn.AvgPool2d(kernel_size=7, stride=1)

        #FC
        self.fc = nn.Linear(in_features=512, out_features=self.num_classes, bias=True)

    def forward(self, x):
        #L0
        x = self.conv_0(x)
        x = self.norm_0(x)
        x = self.activation(x)
        x = self.maxpool_0(x)


        #L1:1
        x_res = x
        x = self.conv1_0(x)
        x = self.norm_1_0(x)
        x = self.activation(x)
        x = self.conv1_1(x)
        x = self.norm_1_1(x)
        x = x_res+x
        x = self.activation(x)
        
        #L1:2
        x_res = x
        x = self.conv1_2(x)
        x = self.norm_1_2(x)
        x = self.activation(x)
        x = self.conv1_3(x)
        x = self.norm_1_3(x)
        x = x_res+x
        x = self.activation(x)


        #L2:1
        x_res = self.res2(x)
        x = self.conv2_0(x)
        x = self.norm2_0(x)
        x = self.activation(x)
        x = self.conv2_1(x)
        x = self.norm2_1(x)
        x = x_res+x
        x = self.activation(x)

        #L2:2
        x_res = x
        x = self.conv2_2(x)
        x = self.norm2_2(x)
        x = self.activation(x)
        x = self.conv2_3(x)
        x = self.norm2_3(x)
        x = x_res+x
        x = self.activation(x)


        #L3:1
        x_res = self.res3(x)
        x = self.conv3_0(x)
        x = self.norm3_0(x)
        x = self.activation(x)
        x = self.conv3_1(x)
        x = self.norm3_1(x)
        x = x_res+x
        x = self.activation(x)

        #L3:2
        x_res = x
        x = self.conv3_2(x)
        x = self.norm3_2(x)
        x = self.activation(x)
        x = self.conv3_3(x)
        x = self.norm3_3(x)
        x = x_res+x
        x = self.activation(x)


        #L4:1
        x_res = self.res4(x)
        x = self.conv4_0(x)
        x = self.norm4_0(x)
        x = self.activation(x)
        x = self.conv4_1(x)
        x = self.norm4_1(x)
        x = x_res+x
        x = self.activation(x)

        #L4:2
        x_res = x
        x = self.conv4_2(x)
        x = self.norm4_2(x)
        x = self.activation(x)
        x = self.conv4_3(x)
        x = self.norm4_3(x)
        x = x_res+x
        x = self.activation(x)


        #GAP
        x = self.gap(x)

        #FC
        x = torch.flatten(x, 1)
        x = self.dropout(x) # to dropout or not
        x = self.fc(x)

        return x


def get_train_data_loader(train_data = "./data/train/", batch_size=128, num_workers=4):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    train_dataset = datasets.ImageFolder(root=train_data, transform=train_transform)
    classes = train_dataset.classes

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return train_loader, classes

def get_valid_data_loader(val_data = "./data/", batch_size=128, num_workers=4):
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3) 
    ])
    val_dataset = datasets.ImageFolder(root=val_data, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return val_loader

def performance(predictions, labels):
    accuracy = accuracy_score(labels, predictions)
    micro_f1 = f1_score(labels, predictions, average='micro')
    macro_f1 = f1_score(labels, predictions, average='macro')
    return accuracy, micro_f1, macro_f1

def per_epoch_training(model, train_loader, val_loader, optimizer, criterion, device):
    # model.to(device)
    #Training
    model.train()
    train_loss = 0
    train_predictions = []
    train_labels = []
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        out = model(images)
        loss = criterion(out, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100)
        optimizer.step()
        train_loss += loss.item()*images.size(0)
        predictions = out.argmax(dim=1)
        train_predictions.extend(predictions.cpu().tolist())
        train_labels.extend(labels.cpu().tolist())
    avg_train_loss = train_loss/len(train_loader.dataset)
    train_accuracy, train_micro_f1, train_macro_f1 = performance(train_predictions, train_labels)

    #Validation
    model.eval()
    val_loss = 0
    data_predictions = []
    data_labels = []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            out = model(images)
            loss = criterion(out, labels)
            val_loss += loss.item()*images.size(0)
            predictions = out.argmax(dim=1)
            data_predictions.extend(predictions.cpu().tolist())
            data_labels.extend(labels.cpu().tolist())
    avg_val_loss = val_loss/len(val_loader.dataset)
    val_accuracy, val_micro_f1, val_macro_f1 = performance(data_predictions, data_labels)

    return avg_train_loss, avg_val_loss, train_accuracy, val_accuracy, train_micro_f1, val_micro_f1, train_macro_f1, val_macro_f1

def train_model(model, train_loader, val_loader, num_epochs=100, patience=5, save_path="./checkpoints/"):
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Training on {device}.")

    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.CrossEntropyLoss()
    history = {key: [] for key in ["train_loss","val_loss","train_acc","val_acc","train_micro_f1","val_micro_f1","train_macro_f1","val_macro_f1"]}

    os.makedirs(save_path, exist_ok=True)

    best_val_loss = float('inf')
    best_train_acc = 0.0
    best_val_acc = 0.0
    best_train_micro_f1 = 0.0
    best_val_micro_f1 = 0.0
    best_train_macro_f1 = 0.0
    best_val_macro_f1 = 0.0

    patience_counter = 0
    t_start = time.time()
    for epoch in range(num_epochs):
        ts = time.time()
        avg_train_loss, avg_val_loss, train_accuracy, val_accuracy, train_micro_f1, val_micro_f1, train_macro_f1, val_macro_f1 = per_epoch_training(model, train_loader, val_loader, optimizer, criterion, device)
        scheduler.step()
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_acc"].append(train_accuracy)
        history["val_acc"].append(val_accuracy)
        history["train_micro_f1"].append(train_micro_f1)
        history["val_micro_f1"].append(val_micro_f1)
        history["train_macro_f1"].append(train_macro_f1)
        history["val_macro_f1"].append(val_macro_f1)

        epoch_time = time.time() - ts
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}, Train Micro F1: {train_micro_f1:.4f}, Val Micro F1: {val_micro_f1:.4f}, Train Macro F1: {train_macro_f1:.4f}, Val Macro F1: {val_macro_f1:.4f}, Time: {epoch_time:.2f} seconds.")


        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "val_acc": val_accuracy
            }, os.path.join(save_path, "model.pt"))

            best_train_acc = train_accuracy
            best_val_acc = val_accuracy
            best_train_micro_f1 = train_micro_f1
            best_val_micro_f1 = val_micro_f1
            best_train_macro_f1 = train_macro_f1
            best_val_macro_f1 = val_macro_f1

        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}.")
                break

    total_time = time.time() - t_start
    print(f"Total Training Time: {total_time:.2f} seconds.")
    return history, best_train_acc, best_val_acc, best_train_micro_f1, best_val_micro_f1, best_train_macro_f1, best_val_macro_f1

def plot_curves(history, save_dir="./curves/", batch_size=128, norm_type="Baseline"):
    epochs = list(range(1, len(history["train_loss"]) + 1))

    # Loss
    plt.figure()
    plt.plot(epochs, history["train_loss"], label="Train")
    plt.plot(epochs, history["val_loss"], label="Val")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"Loss vs Epochs - Batch Size: {batch_size}, Norm: {norm_type}")
    plt.xticks(range(1, len(epochs)+1, max(1, len(epochs)//10)))
    plt.legend()
    plt.savefig(os.path.join(save_dir, "loss.png"))
    plt.close()

    # Accuracy
    plt.figure()
    plt.plot(epochs, history["train_acc"], label="Train")
    plt.plot(epochs, history["val_acc"], label="Val")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy vs Epochs - Batch Size: {batch_size}, Norm: {norm_type}")
    plt.xticks(range(1, len(epochs)+1, max(1, len(epochs)//10)))
    plt.legend()
    plt.savefig(os.path.join(save_dir, "accuracy.png"))
    plt.close()

    # Micro-F1
    plt.figure()
    plt.plot(epochs, history["train_micro_f1"], label="Train")
    plt.plot(epochs, history["val_micro_f1"], label="Val")
    plt.xlabel("Epochs")
    plt.ylabel("Micro-F1")
    plt.title(f"Micro-F1 vs Epochs - Batch Size: {batch_size}, Norm: {norm_type}")
    plt.xticks(range(1, len(epochs)+1, max(1, len(epochs)//10)))
    plt.legend()
    plt.savefig(os.path.join(save_dir, "micro_f1.png"))
    plt.close()

    # Macro-F1
    plt.figure()
    plt.plot(epochs, history["train_macro_f1"], label="Train")
    plt.plot(epochs, history["val_macro_f1"], label="Val")
    plt.xlabel("Epochs")
    plt.ylabel("Macro-F1")
    plt.title(f"Macro-F1 vs Epochs - Batch Size: {batch_size}, Norm: {norm_type}")
    plt.xticks(range(1, len(epochs)+1, max(1, len(epochs)//10)))
    plt.legend()
    plt.savefig(os.path.join(save_dir, "macro_f1.png"))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Training ResNet18")

    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--val_path", type=str, required=True)
    parser.add_argument("--norm_type", type=str, default="Baseline", choices=["Baseline", "BN", "NN", "IN", "BIN", "LN", "GN"])
    parser.add_argument("--save_dir", type=str, required=True)
    args = parser.parse_args()
    
    epochs = 100
    batch_size = 128
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, str(f"batch_{batch_size}"), str(f"norm_{args.norm_type}"))
    os.makedirs(save_path, exist_ok=True)

    train_loader, classes = get_train_data_loader(train_data=args.train_path, batch_size=batch_size, num_workers=min(8, os.cpu_count()))
    val_loader = get_valid_data_loader(val_data=args.val_path, batch_size=batch_size, num_workers=min(8, os.cpu_count()))
    model = ResNet18(num_classes=len(classes), norm=args.norm_type)
    history, best_train_acc, best_val_acc, best_train_micro_f1, best_val_micro_f1, best_train_macro_f1, best_val_macro_f1 = train_model(model, train_loader, val_loader, num_epochs=epochs, patience=5, save_path=save_path)
    os.makedirs(os.path.join(save_path, "curves"), exist_ok=True)
    save_curves_path = os.path.join(save_path, "curves")
    plot_curves(history, save_dir=save_curves_path)
    
    metrics = {
        "best_train_acc": best_train_acc,
        "best_val_acc": best_val_acc,
        "best_train_micro_f1": best_train_micro_f1,
        "best_val_micro_f1": best_val_micro_f1,
        "best_train_macro_f1": best_train_macro_f1,
        "best_val_macro_f1": best_val_macro_f1
    }
    with open(os.path.join(save_curves_path, "best_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
        

if __name__ == "__main__":    
    main()