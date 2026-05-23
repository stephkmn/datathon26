import os
import glob
import pandas as pd
import onnxruntime as ort
from torchvision import datasets, transforms
from PIL import Image

def evaluate(ort_session, path):
    # Pass the path directly, open with PIL
    img = Image.open(path).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img_tensor = transform(img).unsqueeze(0)
    img_numpy = img_tensor.numpy()
    outputs = ort_session.run(None, {'input': img_numpy})
    
    confidence = outputs[0][0][0]
    return confidence

def generate_predictions_csv(ort_session, test_dir="test"):
    """
    Finds all images in the test folder, extracts ground truth from parent folders,
    computes predictions, and writes them out to predictions.csv.
    """
    print("Gathering test images and generating predictions...")
    results = []
    
    # Supported image extensions
    extensions = ('*.jpg', '*.jpeg', '*.png')
    
    # Walk through the test directory to capture images from all subfolders
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(test_dir, "**", ext), recursive=True))
        
    if not image_paths:
        print(f"No images found in directory: {test_dir}")
        return

    # Map your folder names to the corresponding integer labels.
    # Change these if your model interprets 1 as FAKE and 0 as REAL.
    class_mapping = {
        "REAL": 1,
        "FAKE": 0
    }

    for idx, path in enumerate(image_paths):
        # Extract the name of the immediate parent folder (e.g., "REAL" or "FAKE")
        parent_folder = os.path.basename(os.path.dirname(path))
        
        # Get ground truth integer label. Falls back to the folder name string if not in mapping.
        y_true = class_mapping.get(parent_folder.upper(), parent_folder)
        
        confidence = evaluate(ort_session, path)
        y_pred = 1 if confidence >= 0.5 else 0
        
        # Save structural details along with true vs predicted labels
        results.append({
            "filename": os.path.basename(path),
            "filepath": path,
            "folder_label": parent_folder,
            "y_true": y_true,
            "y_pred": y_pred,
            "confidence": confidence
        })
        
        if (idx + 1) % 1000 == 0 or (idx + 1) == len(image_paths):
            print(f"Processed {idx + 1}/{len(image_paths)} images...", end="\r")
            
    # Convert list of dicts to pandas DataFrame and export to CSV
    df = pd.DataFrame(results)
    
    # Reorder columns to place y_true and y_pred front and center
    column_order = ["filename", "y_true", "y_pred", "confidence", "folder_label", "filepath"]
    df = df[column_order]
    
    df.to_csv("predictions.csv", index=False)
    print("\nSuccessfully generated 'predictions.csv' with y_true vs y_pred!")

def main():
    session = ort.InferenceSession("model3.onnx")
    generate_predictions_csv(session, test_dir="test")

if __name__ == "__main__":
    main()