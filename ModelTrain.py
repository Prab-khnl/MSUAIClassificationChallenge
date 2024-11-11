import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, log_loss
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import numpy as np
import os

# Step 1: Data Preprocessing and Augmentation
# --------------------------------------------
def get_data_transforms():
    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(512),        
            transforms.RandomHorizontalFlip(),         
            transforms.RandomVerticalFlip(),           
            transforms.RandomRotation(20),             
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  
            transforms.RandomAffine(degrees=15),       # Randomly rotate and translate
            transforms.GaussianBlur(kernel_size=3),    # Apply Gaussian blur
            transforms.ToTensor(),                     
            transforms.Normalize([0.485, 0.456, 0.406], 
                                 [0.229, 0.224, 0.225]) 
        ]),
        'val': transforms.Compose([
            transforms.Resize(512),                    
            transforms.CenterCrop(512),                
            transforms.ToTensor(),                     
            transforms.Normalize([0.485, 0.456, 0.406], 
                                 [0.229, 0.224, 0.225]) 
        ]),
    }
    return data_transforms

# Step 2: Load Data
# -----------------
def load_data(data_dir, batch_size=32):
    data_transforms = get_data_transforms()
    
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}
    
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size,
                                 shuffle=True, num_workers=4)
                   for x in ['train', 'val']}
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    
    return dataloaders, dataset_sizes, class_names

# Step 3: Define the Model
# ------------------------
def create_model(num_classes):
    # Load a pre-trained ResNet18 model
    model = models.resnet18(pretrained=True)
    
    # Get the input dimensions of the last layer (fully connected layer)
    num_ftrs = model.fc.in_features
    
    # Replace the last layer with a new one that has `num_classes` outputs
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, num_classes),
        nn.Dropout(0.5)  # Add dropout for regularization
    )
    
    return model

# Step 4: Train the Model
# -----------------------
def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=10):
    best_model_wts = model.state_dict()  # Store the best model weights
    best_acc = 0.0  # Track the best validation accuracy
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set the model to training mode
            else:
                model.eval()   # Set the model to evaluation mode
            
            running_loss = 0.0  # Track the loss
            running_corrects = 0  # Track the number of correct predictions
            
            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)  # Move data to the GPU/CPU
                
                optimizer.zero_grad()  # Clear the gradients
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)  # Forward pass
                    _, preds = torch.max(outputs, 1)  # Get the class with the highest score
                    loss = criterion(outputs, labels)  # Calculate the loss
                    
                    if phase == 'train':
                        loss.backward()  # Backpropagate the loss
                        optimizer.step()  # Update the weights
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
        
        # Step the scheduler after each epoch
        scheduler.step()
    
    print(f'Best val Acc: {best_acc:.4f}')
    
    model.load_state_dict(best_model_wts)
    return model

# Step 5: Evaluate the Model
# --------------------------
def evaluate_model(model, dataloader, criterion, class_names):
    model.eval()  # Set model to evaluation mode
    all_labels = []
    all_preds = []
    all_probs = []
    running_loss = 0.0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Calculate the loss
            
            running_loss += loss.item() * inputs.size(0)
            
            _, preds = torch.max(outputs, 1)  # Get the class with the highest score
            probs = torch.nn.functional.softmax(outputs, dim=1)  # Get probabilities
            
            all_labels.extend(labels.cpu().numpy())  # Store true labels
            all_preds.extend(preds.cpu().numpy())    # Store predicted labels
            all_probs.extend(probs.cpu().numpy())    # Store probabilities
    
    avg_loss = running_loss / len(dataloader.dataset)  # Calculate average loss
    
    # Calculate overall metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    logloss = log_loss(all_labels, all_probs)
    
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'Log Loss: {logloss:.4f}')
    
    return accuracy, precision, recall, f1, logloss

# Example Usage
# -------------
if __name__ == "__main__":
    data_dir = "/home/secureslicing/Downloads/MergeDataset"
    dataloaders, dataset_sizes, class_names = load_data(data_dir)
    print(dataloaders, dataset_sizes, class_names)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = create_model(num_classes=len(class_names))
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    # Modify batch size and learning rate
    batch_size = 32  # Adjust this value as needed
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    # Increase the number of epochs for training
    num_epochs = 20  # Increase from 10 to 20 or higher
    
    # Add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # Train the model
    model = train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=num_epochs)
    
    # Evaluate the model on the validation set
    print("Validation Set Performance:")
    evaluate_model(model, dataloaders['val'], criterion, class_names)
    
    # Save the best model
    torch.save(model.state_dict(), "best_model.pth")



"""import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, log_loss
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import numpy as np
import os

from PIL import Image



# Step 1: Data Preprocessing and Augmentation
# --------------------------------------------
def get_data_transforms():
    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(512),        # Randomly crop the image to 512x512
            transforms.RandomHorizontalFlip(),        # Randomly flip the image horizontally
            transforms.ToTensor(),                    # Convert the image to a tensor
            transforms.Normalize([0.485, 0.456, 0.406], # Normalize with standard values
                                 [0.229, 0.224, 0.225]) # Mean and Std for RGB channels
        ]),
        'val': transforms.Compose([
            transforms.Resize(512),                    # Resize the image to 512x512
            transforms.CenterCrop(512),                # Crop from the center
            transforms.ToTensor(),                     # Convert the image to a tensor
            transforms.Normalize([0.485, 0.456, 0.406], # Normalize with standard values
                                 [0.229, 0.224, 0.225]) # Mean and Std for RGB channels
        ]),
    }
    return data_transforms

# Step 2: Load Data
# -----------------
def load_data(data_dir, batch_size=32):
    data_transforms = get_data_transforms()
    
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}
    
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size,
                                 shuffle=True, num_workers=4)
                   for x in ['train', 'val']}
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    
    return dataloaders, dataset_sizes, class_names

# Step 3: Define the Model
# ------------------------
def create_model(num_classes):
    # Load a pre-trained ResNet18 model
    model = models.resnet18(pretrained=True)
    
    # Get the input dimensions of the last layer (fully connected layer)
    num_ftrs = model.fc.in_features
    
    # Replace the last layer with a new one that has `num_classes` outputs
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model

# Step 4: Train the Model
# -----------------------
def train_model(model, dataloaders, criterion, optimizer, num_epochs=10):
    best_model_wts = model.state_dict()  # Store the best model weights
    best_acc = 0.0  # Track the best validation accuracy
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set the model to training mode
            else:
                model.eval()   # Set the model to evaluation mode
            
            running_loss = 0.0  # Track the loss
            running_corrects = 0  # Track the number of correct predictions
            
            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)  # Move data to the GPU/CPU
                
                optimizer.zero_grad()  # Clear the gradients
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)  # Forward pass
                    _, preds = torch.max(outputs, 1)  # Get the class with the highest score
                    loss = criterion(outputs, labels)  # Calculate the loss
                    
                    if phase == 'train':
                        loss.backward()  # Backpropagate the loss
                        optimizer.step()  # Update the weights
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
    
    print(f'Best val Acc: {best_acc:.4f}')
    
    model.load_state_dict(best_model_wts)
    return model

# Step 5: Evaluate the Model
# --------------------------
def evaluate_model(model, dataloader, criterion, class_names):
    model.eval()  # Set model to evaluation mode
    all_labels = []
    all_preds = []
    all_probs = []
    running_loss = 0.0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Calculate the loss
            
            running_loss += loss.item() * inputs.size(0)
            
            _, preds = torch.max(outputs, 1)  # Get the class with the highest score
            probs = torch.nn.functional.softmax(outputs, dim=1)  # Get probabilities
            
            all_labels.extend(labels.cpu().numpy())  # Store true labels
            all_preds.extend(preds.cpu().numpy())    # Store predicted labels
            all_probs.extend(probs.cpu().numpy())    # Store probabilities
    
    avg_loss = running_loss / len(dataloader.dataset)  # Calculate average loss
    
    # Calculate overall metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    logloss = log_loss(all_labels, all_probs)
    
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'Log Loss: {logloss:.4f}')
    
    return accuracy, precision, recall, f1, logloss

# Example Usage
# -------------
if __name__ == "__main__":
    data_dir = "/home/secureslicing/Downloads/MergeDataset"
    dataloaders, dataset_sizes, class_names = load_data(data_dir)
    print(dataloaders,dataset_sizes,class_names)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = create_model(num_classes=len(class_names))
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    # Train the model
    model = train_model(model, dataloaders, criterion, optimizer, num_epochs=10)
    
    # Evaluate the model on the validation set
    print("Validation Set Performance:")
    evaluate_model(model, dataloaders['val'], criterion, class_names)
    
    # Save the best model
    torch.save(model.state_dict(), "best_model.pth")
"""