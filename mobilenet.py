import random
import math
import torch
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import cv2 
from sklearn.metrics import roc_auc_score
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import roc_curve, auc, roc_auc_score
import numpy as np
import torch
import torch.nn.functional as F

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
CPU_DEVICE = torch.device("cpu")

mobilenet_v3_large = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT, pretrained=False)

# Modify the final layer for a custom number of classes (e.g., 10)
mobilenet_v3_large.classifier[3] = nn.Linear(in_features=1280, out_features=14)
mobilenet_v3_large.to(DEVICE)

# Define data transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

TRAIN_TYPE_TRAIN = 0
TRAIN_TYPE_EVAL = 1
TRAIN_TYPE_TEST = 2

LABELS = [
    "Atelectasis","Cardiomegaly","Effusion","Infiltration","Mass","Nodule",
    "Pneumonia","Pneumothorax","Consolidation","Edema","Emphysema","Fibrosis",
    "Pleural_Thickening","Hernia"
]

def get_diagnosis(diagnosis_label):
    match diagnosis_label:
        case "Atelectasis":
            return 0
        case "Cardiomegaly":
            return 1
        case "Effusion":
            return 2
        case "Infiltration":
            return 3
        case "Mass":
            return 4
        case "Nodule":
            return 5
        case "Pneumonia":
            return 6
        case "Pneumothorax":
            return 7
        case "Consolidation":
            return 8
        case "Edema":
            return 9
        case "Emphysema":
            return 10
        case "Fibrosis":
            return 11
        case "Pleural_Thickening":
            return 12
        case "Hernia":
            return 13
        case _: # No Finding
            return 14

class XrayDataset(Dataset):
    # Constructor
    IMAGE_WIDTH = 224
    IMAGE_HEIGHT = 224

    def __init__(self, transform=None, train_type=TRAIN_TYPE_TRAIN):

        IMAGE_DIR_PATH = 'C:\\src\\python\\UofC\\DATA\\archive\\total_images\\images\\'
        label_file_path = 'C:\\src\\python\\UofC\\DATA\\archive\\Data_Entry_2017.csv'

        #IMAGE_DIR_PATH = 'C:\\src\\python\\UofC\\DATA\\archive\\images_001\\images\\'
        #label_file_path = 'C:\\src\\python\\UofC\\DATA\\archive\\Temp_Data_Entry_2017.csv'

        image_files = []
        disease_type_onehot = []
        self.all_files = []
        self.Y = []

        with open(label_file_path, mode='r') as file_csv:
            # csv 목록을 한 줄씩 불러들이다
            line_index = 0

            for line in file_csv:

                if(line_index == 0):
                    line_index += 1
                    continue

                parts = line.split(",")
    
                image_file = parts[0]
                # 첫번째가 wav 파일 경로
                disease_part = parts[1]

                row = np.zeros(14)
                if (disease_part.find("|") >= 0):
                    diagnosis_arr = disease_part.split("|")

                    for tmp_diagnosis in diagnosis_arr:
                        diagnosis = get_diagnosis(tmp_diagnosis)
                        if(diagnosis != 14):
                            row[diagnosis] = 1

                else:
                    diagnosis = get_diagnosis(disease_part)
                    if(diagnosis != 14):
                        row[diagnosis] = 1

                image_files.append(IMAGE_DIR_PATH + image_file)
                disease_type_onehot.append(torch.tensor(row))

            #human_files.sort()
            #random.shuffle(human_files, random = lambda: 0.7)

            number_of_samples = len(image_files)
            # The transform is goint to be used on image
            self.transform = transform

            train_total_length = math.floor(len(image_files) * 0.8)
            eval_total_length = math.floor(len(image_files) * 0.1)
            test_total_length = math.floor(len(image_files) * 0.1)

            if train_type == TRAIN_TYPE_TRAIN:
                for index in range(0, train_total_length):
                    self.all_files.append(image_files[index])
                    self.Y.append(disease_type_onehot[index])

                self.len = len(self.all_files)
            elif train_type == TRAIN_TYPE_EVAL:
                for index in range(train_total_length, train_total_length + eval_total_length):
                    self.all_files.append(image_files[index])
                    self.Y.append(disease_type_onehot[index])

                self.len = len(self.all_files)
            else:
                #train_type == TRAIN_TYPE_TEST:
                for index in range(train_total_length + eval_total_length, train_total_length + eval_total_length + test_total_length):
                    self.all_files.append(image_files[index])
                    self.Y.append(disease_type_onehot[index])

                self.len = len(self.all_files)

    def __len__(self):
        return self.len

    # Getter
    def __getitem__(self, idx):

        #image = np.fromfile(self.all_files[idx], dtype=np.float32)
        # h x 프레임 수 x 차원 수 배열로 변형
        #image = image.reshape(int(1), int(self.IMAGE_HEIGHT), int(self.IMAGE_WIDTH))
        #image = image.transpose(2, 1, 0)

        img = cv2.imread(self.all_files[idx], cv2.IMREAD_GRAYSCALE)
        img_color_changed = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #img_tensor = transform(img_color_changed).unsqueeze(0)
        img_tensor = transform(img_color_changed)

        return img_tensor, self.Y[idx]

# Load dataset
#train_dataset = datasets.ImageFolder(root='C:\\src\\python\\UofC\\DATA\\archive\\total_images\\images', transform=transform)

train_dataset = XrayDataset(transform=transform, train_type=TRAIN_TYPE_TRAIN)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)


#train_dataset = datasets.ImageFolder(root='C:\\src\\python\\UofC\\DATA\\archive\\total_images\\images', transform=transform)
val_dataset = XrayDataset(transform=transform, train_type=TRAIN_TYPE_EVAL)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

test_dataset = XrayDataset(transform=transform, train_type=TRAIN_TYPE_TEST)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# Training loop
EPOCH_SIZE = 5
prev_loss = 1.0

# Define the loss function and optimizer
criterion = nn.BCEWithLogitsLoss()

#optimizer = optim.Adam(mobilenet_v3_large.parameters(), lr=0.05)
optimizer = optim.AdamW(mobilenet_v3_large.parameters(), lr=3e-4, weight_decay=1e-2)
# Cosine LR decay: starts at lr, anneals smoothly to 0 over T_max epochs
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCH_SIZE, eta_min=1e-6)

history = {
    "train_loss": [],
    "val_loss":   [],
    "val_acc":    [],
}

def train():
    mobilenet_v3_large.train()
    running_loss = 0.0
    correct     = 0
    total       = 0
    for inputs, labels in train_loader:

        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        
        # Forward pass
        outputs = mobilenet_v3_large(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)          # un-average the mean loss
        probs       = torch.sigmoid(outputs)
        predicted   = (probs > 0.5).float()
        correct    += (predicted == labels).sum().item()
        total      += labels.numel()
    
    n            = len(train_loader.dataset)
    running_loss  /= n
    train_acc    = 100.0 * correct / total
    return running_loss, train_acc
    
    #if(loss_val <= 0.01):
    #    break

    #if(prev_loss < loss_val):
    #    break
    #prev_loss = loss_val

def evaluate():
    mobilenet_v3_large.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = mobilenet_v3_large(inputs)
            
            loss = criterion(outputs, labels)
            val_loss  += loss.item() * inputs.size(0)
            probs      = torch.sigmoid(outputs)
            predicted  = (probs > 0.5).float()
            correct   += (predicted == labels).sum().item()
            total     += labels.numel()

    n = len(val_loader.dataset)
    val_loss /= n
    val_acc   = 100.0 * correct / total
    print(f"  Val Loss: {val_loss:.5f}  |  Val Accuracy: {val_acc:.4f}%")
    return val_loss, val_acc

def testCheck():
    test_all_probs_array = []
    test_all_labels_array = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = mobilenet_v3_large(inputs)

            probs = torch.softmax(outputs, dim=1)

            test_all_probs_array.extend(probs.cpu().numpy())
            test_all_labels_array.extend(labels.cpu().numpy())

    all_probs  = np.array(test_all_probs_array)
    all_labels = np.array(test_all_labels_array)

    # ── Per-class AUROC ────────────────────────────────────────────────────────
    per_class_auc = roc_auc_score(all_labels, all_probs, average=None)
    mean_auc      = roc_auc_score(all_labels, all_probs, average="macro")

    print(f"\n{'─'*40}")
    print(f"  ROC AUC (macro):  {mean_auc:.4f}")
    print(f"{'─'*40}")
    for name, score in zip(LABELS, per_class_auc):
        bar = "█" * int(score * 20)
        print(f"  {name:22s} {score:.4f}  {bar}")
    print(f"{'─'*40}\n")

    return all_labels, all_probs, per_class_auc, mean_auc

def plot_training_curves(history):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training History", fontsize=15, fontweight="bold")
    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    axes[0].plot(epochs, history["train_loss"], "b-o", label="Train Loss",
                 markersize=4, linewidth=1.8)
    axes[0].plot(epochs, history["val_loss"],   "r-o", label="Val Loss",
                 markersize=4, linewidth=1.8)
    axes[0].set_title("Loss per Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BCE Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history["val_acc"], "g-o", label="Val Accuracy",
                 markersize=4, linewidth=1.8)
    axes[1].set_title("Validation Accuracy per Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    plt.show()


def plot_roc_curves(all_labels, all_probs, per_class_auc, mean_auc):
    """
    Two-panel figure:
      Left  — one ROC curve per disease label
      Right — macro-average ROC curve
    """
    fig = plt.figure(figsize=(18, 8))
    fig.suptitle("ROC / AUC Analysis — NIH ChestXray14", fontsize=15, fontweight="bold")
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

    # ── Left: per-class curves ─────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    cmap = plt.cm.get_cmap("tab20", len(LABELS))

    mean_fpr  = np.linspace(0, 1, 300)
    interp_tprs = []

    for i, label in enumerate(LABELS):
        fpr, tpr, _ = roc_curve(all_labels[:, i], all_probs[:, i])
        interp_tpr  = np.interp(mean_fpr, fpr, tpr)
        interp_tprs.append(interp_tpr)
        ax1.plot(fpr, tpr, color=cmap(i), linewidth=1.4,
                 label=f"{label} ({per_class_auc[i]:.3f})")

    ax1.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.6, label="Random")
    ax1.set_title("Per-class ROC Curves", fontsize=12)
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.legend(loc="lower right", fontsize=7.5, ncol=1)
    ax1.grid(alpha=0.25)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1.02])

    # ── Right: macro-average + confidence band ─────────────────────────────────
    ax2 = fig.add_subplot(gs[1])

    mean_tpr      = np.mean(interp_tprs, axis=0)
    std_tpr       = np.std(interp_tprs,  axis=0)
    mean_tpr[0]   = 0.0
    mean_tpr[-1]  = 1.0

    ax2.plot(mean_fpr, mean_tpr, color="royalblue", linewidth=2.2,
             label=f"Macro-avg ROC (AUC = {mean_auc:.4f})")
    ax2.fill_between(mean_fpr,
                     np.clip(mean_tpr - std_tpr, 0, 1),
                     np.clip(mean_tpr + std_tpr, 0, 1),
                     color="royalblue", alpha=0.15,
                     label="± 1 std across classes")
    ax2.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.6, label="Random")
    ax2.set_title("Macro-Average ROC Curve", fontsize=12)
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.legend(loc="lower right", fontsize=9)
    ax2.grid(alpha=0.25)
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1.02])

    plt.savefig("roc_auc_curves.png", dpi=150)
    plt.show()

def saveModel():
    now       = datetime.now()
    dt_string = now.strftime("_%Y%m%d_%H%M%S")
    torch.save(mobilenet_v3_large.state_dict(), "mobilenetv3" + dt_string + ".pt")
    print(f"  Model saved → mobilenetv3{dt_string}.pt")

def executeDeepLearning():
    torch.manual_seed(0)

    for epoch in range(1, EPOCH_SIZE + 1):
        print(f"\nEpoch [{epoch}/{EPOCH_SIZE}]")

        train_loss, train_acc = train()
        val_loss,   val_acc   = evaluate()
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"  Train Loss: {train_loss:.5f}  |  Train Acc: {train_acc:.4f}%")

        if train_loss <= 0.001:
            print("  Early stop: train loss threshold reached.")
            break

    # ── Final evaluation ───────────────────────────────────────────────────────
    print("\n" + "="*50)
    print("  FINAL TEST EVALUATION")
    print("="*50)
    all_labels, all_probs, per_class_auc, mean_auc = testCheck()

    plot_training_curves(history)
    plot_roc_curves(all_labels, all_probs, per_class_auc, mean_auc)

    saveModel()


if __name__ == "__main__":
    executeDeepLearning()