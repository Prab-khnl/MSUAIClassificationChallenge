
import sys
import os
import torch
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QLabel, QApplication, QVBoxLayout, QWidget
from PyQt5.QtCore import Qt
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models

# Load the best saved model
def load_model(model_path, num_classes):
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, num_classes)
    
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Define transformation
def get_image_transform():
    return transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(512),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# Predict the class of the image
def predict_image_class(model, image, transform, class_names):
    transformed_image = transform(image).unsqueeze(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    transformed_image = transformed_image.to(device)
    
    with torch.no_grad():
        outputs = model(transformed_image)
        _, preds = torch.max(outputs, 1)
    
    predicted_class = class_names[preds.item()]
    return predicted_class

class ImageDropArea(QWidget):
    def __init__(self, model, transform, class_names):
        super().__init__()
        self.setAcceptDrops(True)
        
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Image label
        self.image_label = QLabel("Drag and drop an image here")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("QLabel { border: 2px dashed #aaa; }")
        self.layout.addWidget(self.image_label)

        # Class name label
        self.class_label = QLabel("")
        self.class_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.class_label)

        self.model = model
        self.transform = transform
        self.class_names = class_names

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                file_path = url.toLocalFile()
                if os.path.isfile(file_path) and file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.process_image(file_path)

    def process_image(self, file_path):
        image = Image.open(file_path).convert('RGB')
        class_name = predict_image_class(self.model, image, self.transform, self.class_names)

        # Update GUI to show the image and the predicted class
        pixmap = QtGui.QPixmap(file_path).scaled(512, 512, Qt.KeepAspectRatio)
        self.image_label.setPixmap(pixmap)
        self.class_label.setText(f"Predicted Class: {class_name}")

class App(QWidget):
    def __init__(self, model, transform, class_names):
        super().__init__()
        self.setWindowTitle("Image Class Predictor")
        self.setGeometry(100, 100, 600, 600)

        self.drop_area = ImageDropArea(model, transform, class_names)
        layout = QVBoxLayout()
        layout.addWidget(self.drop_area)
        self.setLayout(layout)

if __name__ == "__main__":
    # Load model
    model_path = "./train/best_model.pth"  # Path to the saved best model
    num_classes = 10  # Replace with the number of classes in your dataset
    class_names = ['Butler Hall', 'Carpenter Hall', 'Lee Hall', 'McCain Hall', 'McCool Hall', 'Old Main', 'Simrall Hall', 'Student Union', 'Swalm Hall', 'Walker Hall']  # Replace with your actual class names

    transform = get_image_transform()  # Get the image transformation
    model = load_model(model_path, num_classes)  # Load the model

    app = QApplication(sys.argv)
    window = App(model, transform, class_names)
    window.show()
    sys.exit(app.exec_())

