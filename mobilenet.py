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

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
CPU_DEVICE = torch.device("cpu")

mobilenet_v3_large = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT, pretrained=True)

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
        label_file_path = 'C:\\src\\python\\UofC\\DATA\\archive\\Temp_Data_Entry_2017.csv'

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
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


#train_dataset = datasets.ImageFolder(root='C:\\src\\python\\UofC\\DATA\\archive\\total_images\\images', transform=transform)
val_dataset = XrayDataset(transform=transform, train_type=TRAIN_TYPE_EVAL)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

test_dataset = XrayDataset(transform=transform, train_type=TRAIN_TYPE_TEST)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)


# Define the loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(mobilenet_v3_large.parameters(), lr=0.001)
# Training loop
num_epochs = 10
mobilenet_v3_large.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, labels in train_loader:

        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        
        # Forward pass
        outputs = mobilenet_v3_large(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

mobilenet_v3_large.eval()  # Set the model to evaluation mode


correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in val_loader:
        
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        outputs = mobilenet_v3_large(inputs)

        #_, predicted = torch.max(outputs.data, 1)
        #total += labels.size(0)
        #correct += (predicted == labels).sum().item()

        outputs = torch.sigmoid(outputs)
        predicted = (outputs > 0.5).float()

        correct += (predicted == labels).sum().item()
        total += labels.numel()


accuracy = 100 * correct / total
print(f'Validation Accuracy: {accuracy:.2f}%')

test_all_probs_array = []
test_all_labels_array = []

with torch.no_grad():
    for inputs, labels in test_loader:
        
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        outputs = mobilenet_v3_large(inputs)

        probs = torch.softmax(outputs, dim=1)

        test_all_probs_array.extend(probs.cpu().numpy())
        test_all_labels_array.extend(labels.cpu().numpy())

        #test_auc_per_class = roc_auc_score(labels, outputs, average=None)
        #test_mean_auc = roc_auc_score(labels, outputs, average="macro")

        #test_mean_auc_array.append(test_mean_auc)

        #print("\nTest Mean AUROC:", test_mean_auc)
        #for name, auc in zip(LABELS, test_auc_per_class):
        #    print(f"TEST {name:18s}: {auc:.4f}")


#test_auc_average = sum(test_mean_auc_array) / len(test_mean_auc_array)
#print("\nMobileNetV3 Total Test Mean AUROC:", test_auc_average)
#for name, auc in zip(LABELS, test_auc_per_class):
#    print(f"TEST {name:18s}: {auc:.4f}")

auc = roc_auc_score(test_all_labels_array, test_all_probs_array, multi_class='ovr')
print("ROC AUC OVR:", auc)

test_auc_per_class = roc_auc_score(test_all_labels_array, test_all_probs_array, average=None)
test_mean_auc = roc_auc_score(test_all_labels_array, test_all_probs_array, average="macro")

print("\nTest Mean AUROC:", test_mean_auc)

for name, auc in zip(LABELS, test_auc_per_class):
    print(f"TEST {name:18s}: {auc:.4f}")

torch.save(mobilenet_v3_large, '/model.pt')  # 전체 모델 저장

