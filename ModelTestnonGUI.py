import torch
import os
from PIL import Image
import torchvision.transforms as transforms
import random
import torchvision.models as models


# This code loops over the images randmly and logs the Predicted class information
# of the image in result.txt file.

# Step 1: Load the best saved model
def load_model(model_path, num_classes):
    model = models.resnet18(pretrained=False)  # We set pretrained to False as we're loading a custom trained model
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, num_classes)  # Adjust final layer to match number of classes
    
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    return model

# Step 2: Define transformation (should match your validation set transformations)
def get_image_transform():
    return transforms.Compose([
        transforms.Resize(512),                    # Resize the image to 512x512
        transforms.CenterCrop(512),                # Crop from the center
        transforms.ToTensor(),                     # Convert the image to a tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize with standard values
    ])

# Step 3: Get a random image and process it
def get_random_image(data_dir):
    # Walk through the dataset folder and collect all image paths
    all_images = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith((".jpg", ".jpeg", ".png")):  # Add extensions as per your dataset
                all_images.append(os.path.join(root, file))
    
    # Randomly select an image
    random_image_path = random.choice(all_images)
    image = Image.open(random_image_path).convert('RGB')  # Load the image and ensure it's in RGB format
    
    return random_image_path, image

# Step 4: Predict the class of the image
def predict_image_class(model, image_path, image, transform, class_names):
    transformed_image = transform(image).unsqueeze(0)  # Apply transformations and add a batch dimension
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    transformed_image = transformed_image.to(device)
    
    # Run the image through the model
    with torch.no_grad():
        outputs = model(transformed_image)
        _, preds = torch.max(outputs, 1)  # Get the class with the highest score
    
    predicted_class = class_names[preds.item()]
    f = open('ModelPredictionResult.csv', 'a')
    # Print the image name, path, and predicted class
    print(f"Image Path: {image_path}")
    print(f"Image Name: {os.path.basename(image_path)}")
    print(f"Predicted Class: {predicted_class}")
    output = f"{image_path},{predicted_class}\n"
    f.write(output)

# Example Usage
if __name__ == "__main__":
    data_dir = "./test"  # Path to validation set or any dataset
    model_path = "./train/best_model.pth"  # Path to the saved best model
    num_classes = 10  # Replace with the number of classes in your dataset
    class_names = ['Butler Hall', 'Carpenter Hall', 'Lee Hall', 'McCain Hall', 'McCool Hall', 'Old Main', 'Simrall Hall', 'Student Union', 'Swalm Hall', 'Walker Hall']  # Replace with your actual class names
    
    
    # Load model
    model = load_model(model_path, num_classes)
    f = open('ModelPredictionResult.csv', 'w')
    f.write('')
    for i in range(1000):
        # Get a random image
        random_image_path, random_image = get_random_image(data_dir)
        
        # Get the image transformation
        transform = get_image_transform()
        
        # Predict the class of the image
        predict_image_class(model, random_image_path, random_image, transform, class_names)

