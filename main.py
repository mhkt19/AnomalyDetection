import os
import shutil
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
from torchvision.models import ResNet18_Weights
from PIL import Image

# Set the train size ratio
train_size_ratio = 0.8
# Set the minimum and maximum number of epochs
min_epochs = 5
max_epochs = 15
# Set the patience for early stopping
patience = 3

# Function to create directories for the current run
def create_timestamped_folders(base_dir):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(base_dir, timestamp)
    train_dir = os.path.join(run_dir, 'train')
    test_dir = os.path.join(run_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    return run_dir, train_dir, test_dir

# Function to copy files to train and test directories
def split_data(source_dir, train_dir, test_dir, train_ratio):
    classes = os.listdir(source_dir)
    for cls in classes:
        class_dir = os.path.join(source_dir, cls)
        if os.path.isdir(class_dir):
            train_class_dir = os.path.join(train_dir, cls)
            test_class_dir = os.path.join(test_dir, cls)
            os.makedirs(train_class_dir, exist_ok=True)
            os.makedirs(test_class_dir, exist_ok=True)
            files = os.listdir(class_dir)
            random.shuffle(files)
            split_idx = int(len(files) * train_ratio)
            train_files = files[:split_idx]
            test_files = files[split_idx:]
            for file in train_files:
                shutil.copy(os.path.join(class_dir, file), os.path.join(train_class_dir, file))
            for file in test_files:
                shutil.copy(os.path.join(class_dir, file), os.path.join(test_class_dir, file))

# Function to save misclassified images
def save_misclassified_image(image_path, save_dir, class_name):
    os.makedirs(save_dir, exist_ok=True)
    shutil.copy(image_path, os.path.join(save_dir, os.path.basename(image_path)))

# Base directory and dataset directory
base_dir = '.'
dataset_dir = 'dataset'

# Create timestamped train and test directories
run_dir, train_dir, test_dir = create_timestamped_folders(base_dir)

# Split data into train and test sets
split_data(dataset_dir, train_dir, test_dir, train_size_ratio)

# Define Data Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load datasets
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
test_dataset = datasets.ImageFolder(test_dir, transform=transform)

# Split train dataset into training and validation
train_size = int(train_size_ratio * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Define the Model
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 2)  #  2 classes

    def forward(self, x):
        return self.model(x)

model = Classifier()

# Set Up Training Components
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=patience, factor=0.1)

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    print(f"GPU Device Name: {torch.cuda.get_device_name(0)}")

model.to(device)

# Training and Validation Loop
best_val_loss = float('inf')
epochs_no_improve = 0
best_model_wts = None

for epoch in range(max_epochs):
    # Training phase
    model.train()
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = running_corrects.double() / len(train_dataset)

    print(f"Epoch {epoch+1}/{max_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

    # Validation phase
    model.eval()
    val_running_loss = 0.0
    val_running_corrects = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            val_running_loss += loss.item() * inputs.size(0)
            val_running_corrects += torch.sum(preds == labels.data)

    val_epoch_loss = val_running_loss / len(val_dataset)
    val_epoch_acc = val_running_corrects.double() / len(val_dataset)

    print(f"Validation Loss: {val_epoch_loss:.4f}, Validation Accuracy: {val_epoch_acc:.4f}")

    # Step the scheduler
    scheduler.step(val_epoch_loss)

    # Check early stopping condition
    if val_epoch_loss < best_val_loss:
        best_val_loss = val_epoch_loss
        epochs_no_improve = 0
        best_model_wts = model.state_dict()  # Save the best model weights
    else:
        epochs_no_improve += 1

    if epoch >= min_epochs and epochs_no_improve >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break

# Load best model weights
if best_model_wts is not None:
    model.load_state_dict(best_model_wts)

# Save the trained model
torch.save(model.state_dict(), os.path.join(run_dir, 'model.pth'))

def evaluate_and_save_misclassified(model, loader, dataset, dataset_type):
    corrects = 0
    all_preds = []
    all_labels = []
    misclassified_fp_dir = os.path.join(run_dir, dataset_type, 'misclassified', 'FP')
    misclassified_fn_dir = os.path.join(run_dir, dataset_type, 'misclassified', 'FN')

    with torch.no_grad():
        for inputs, labels, paths in loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            corrects += torch.sum(preds == labels.data)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            for i in range(len(labels)):
                if preds[i] != labels[i]:
                    if preds[i] == 1:
                        save_misclassified_image(paths[i], misclassified_fp_dir, dataset.classes[labels[i]])
                    else:
                        save_misclassified_image(paths[i], misclassified_fn_dir, dataset.classes[labels[i]])

    accuracy = corrects.double() / len(dataset)
    return accuracy, all_labels, all_preds

# Define custom dataset to include image paths
class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        original_tuple = super().__getitem__(index)
        path = self.imgs[index][0]
        return original_tuple + (path,)

# Re-load datasets with custom dataset class
train_dataset_with_paths = ImageFolderWithPaths(train_dir, transform=transform)
test_dataset_with_paths = ImageFolderWithPaths(test_dir, transform=transform)

train_loader_with_paths = DataLoader(train_dataset_with_paths, batch_size=1, shuffle=False)
test_loader_with_paths = DataLoader(test_dataset_with_paths, batch_size=1, shuffle=False)

# Evaluate on train data
train_acc, train_labels, train_preds = evaluate_and_save_misclassified(model, train_loader_with_paths, train_dataset_with_paths, 'train')
print(f"Train Accuracy: {train_acc * 100:.2f}%")

# Evaluate on test data
test_acc, test_labels, test_preds = evaluate_and_save_misclassified(model, test_loader_with_paths, test_dataset_with_paths, 'test')
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# Calculate and print metrics for test data
conf_matrix = confusion_matrix(test_labels, test_preds)
precision = precision_score(test_labels, test_preds, average='binary')
recall = recall_score(test_labels, test_preds, average='binary')
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")

# Store metrics in a text file
metrics_file = os.path.join(run_dir, 'metrics.txt')
with open(metrics_file, 'w') as f:
    f.write(f"Train Accuracy: {train_acc * 100:.2f}%\n")
    f.write(f"Test Accuracy: {test_acc * 100:.2f}%\n")
    f.write(f"Confusion Matrix:\n{conf_matrix}\n")
    f.write(f"Precision: {precision * 100:.2f}%\n")
    f.write(f"Recall: {recall * 100:.2f}%\n")
