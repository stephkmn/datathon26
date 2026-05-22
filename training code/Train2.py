import pandas as pd
import onnx
import cv2
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

BATCH_SIZE = 512
EPOCHS = 100

# Define Convolutional Neural Network Class
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # Block 1: 3 input channels (RGB), 32 filters, 3x3 kernel
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2) # Image size: 32x32 -> 16x16
        
        # Block 2: 32 input channels, 64 filters, 3x3 kernel
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2) # Image size: 16x16 -> 8x8
        
        # Fully Connected Layers
        self.flatten_size = 64 * 8 * 8
        self.fc1 = nn.Linear(self.flatten_size, 128)
        self.dropout = nn.Dropout(0.5) # Prevents overfitting
        self.fc_out = nn.Linear(128, 1)
        
    def forward(self, x):
        x = self.pool1(nn.functional.relu(self.bn1(self.conv1(x))))
        x = self.pool2(nn.functional.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, self.flatten_size) 
        x = self.dropout(nn.functional.relu(self.fc1(x)))

        # Apply Sigmoid to get a single confidence score [0, 1]
        confidence = torch.sigmoid(self.fc_out(x))
        return confidence

def main():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)  

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    model = SimpleCNN().to(device)
    
    # Parse Data
    train_transforms = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_ds = datasets.ImageFolder("train", transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    test_ds = datasets.ImageFolder("test", transform=test_transforms)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Train Variables
    criterion = nn.BCELoss() # Binary Cross Entropy
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()

    # Train Model
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        q=0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            q+=1
            print(f"Batch: -{q}/{len(train_loader)}-", end="\r")

        print(f"Epoch {epoch}, Loss: {running_loss/len(train_loader):.4f}")
        if (epoch+1) % 5 == 0:
            # Inside your validation loop (every 5 epochs)
            correct = 0
            total = 0
            val_loss = 0.0

            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.float().unsqueeze(1).to(device)

                output = model(images)
                loss = criterion(output, labels)
                val_loss += loss.item()

                # Calculate accuracy: since output is Sigmoid [0,1], threshold at 0.5
                predictions = (output >= 0.5).float()
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

            test_accuracy = (correct / total) * 100
            print(f"Test Loss: {val_loss/len(test_loader):.4f}, Test Accuracy: {test_accuracy:.2f}%")
    
    model.eval() 
    torch.onnx.export(
        model, 
        torch.randn(1, 3, 32, 32).to(device),
        "model2.onnx", 
        export_params=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    torch.save(model.state_dict(), "model2.pt")

if __name__ == "__main__":
    main()