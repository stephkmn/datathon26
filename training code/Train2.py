import pandas as pd
import onnx
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

BATCH_SIZE = 512
EPOCHS = 20

class SpectralGate(nn.Module):
    def __init__(self, channels):
        super(SpectralGate, self).__init__()
        self.conv_r = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv_i = nn.Conv2d(channels, channels, kernel_size=1)
        
        self.gate = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        # Change rfft2 to fft2. No 's' parameter needed since it preserves size.
        x_fft = torch.fft.fft2(x, norm='ortho')
        r, i = self.conv_r(x_fft.real), self.conv_i(x_fft.imag)
        x_fft = torch.complex(r, i)
        
        # Change irfft2 to ifft2. real extracts the final spatial features.
        out = torch.fft.ifft2(x_fft, norm='ortho').real
        
        return self.bn(self.gate * out)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # Block 1: 3 -> 32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.spectral = SpectralGate(32) # Applied BEFORE pooling
        self.pool1 = nn.MaxPool2d(2, 2)  # 32x32 -> 16x16
        
        # Block 2: 32 -> 64
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)  # 16x16 -> 8x8
        
        # Block 3: 64 -> 128 (Added depth to capture hierarchical AI artifacts)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)  # 8x8 -> 4x4
        
        # Fully Connected Layers
        self.flatten_size = 128 * 4 * 4
        self.fc1 = nn.Linear(self.flatten_size, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc_out = nn.Linear(256, 1)
        
    def forward(self, x):
        # Block 1 with Spectral Residual
        x_conv1 = nn.functional.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x_conv1 + self.spectral(x_conv1))
        
        # Block 2 & 3
        x = self.pool2(nn.functional.relu(self.bn2(self.conv2(x))))
        x = self.pool3(nn.functional.relu(self.bn3(self.conv3(x))))
        
        # Flatten and Classification
        x = x.view(-1, self.flatten_size) 
        x = self.dropout(nn.functional.relu(self.fc1(x)))
        
        # Return RAW LOGITS (No Sigmoid). Required for BCEWithLogitsLoss.
        return self.fc_out(x)

# Wrapper to add Sigmoid back for the Chrome Extension ONNX export
class ExportModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model
    def forward(self, x):
        return torch.sigmoid(self.base(x))

def main():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)  

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    model = SimpleCNN().to(device)
    
    # Data Augmentation: Added Blur and Jitter to simulate web compression
    train_transforms = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.GaussianBlur(3)], p=0.2),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2)], p=0.2),
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

    # Numerically stable loss + L2 Regularization (weight_decay)
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Learning Rate Scheduler (cools down the learning rate over time)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Train Model
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        q = 0

        for images, labels in train_loader:
            images = images.to(device)
            # Label Smoothing: 0 -> 0.05, 1 -> 0.95
            labels = labels.float().unsqueeze(1).to(device)
            smoothed_labels = labels * 0.9 + 0.05 

            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, smoothed_labels)
            
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            q += 1
            print(f"Batch: -{q}/{len(train_loader)}-", end="\r")

        scheduler.step() # Update learning rate

        print(f"Epoch {epoch}, Loss: {running_loss/len(train_loader):.4f}")
        
        if (epoch+1) % 5 == 0:
            model.eval()
            correct = 0
            total = 0
            val_loss = 0.0

            with torch.no_grad(): # Disable gradients for validation speed
                for images, labels in test_loader:
                    images = images.to(device)
                    labels = labels.float().unsqueeze(1).to(device)

                    output = model(images)
                    loss = criterion(output, labels) # Use strict labels for val loss
                    val_loss += loss.item()

                    # Apply sigmoid here since model outputs logits now
                    predictions = (torch.sigmoid(output) >= 0.5).float()
                    correct += (predictions == labels).sum().item()
                    total += labels.size(0)

            test_accuracy = (correct / total) * 100
            print(f"Test Loss: {val_loss/len(test_loader):.4f}, Test Accuracy: {test_accuracy:.2f}%")
    
    # Export: Wrap the model to ensure the Chrome Extension still gets a [0, 1] probability
    export_model = ExportModel(model)
    export_model.eval() 
    
    # Create a clean mock input with a strict batch size of 1
    dummy_input = torch.randn(1, 3, 32, 32).to(device)
    
    torch.onnx.export(
        export_model, 
        dummy_input,
        "model3.onnx", 
        export_params=True,
        input_names=['input'],
        output_names=['output'],
        opset_version=17  # Explicitly setting a modern opset handles FFT operations cleaner
    )
    torch.save(model.state_dict(), "model3.pt")
    print("Export complete: model2.onnx generated with fixed batch size (1, 3, 32, 32).")

if __name__ == "__main__":
    main()