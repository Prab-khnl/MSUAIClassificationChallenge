Group Name :- Anton

In this project,ResNet18 model was trained for image classification using the PyTorch framework. The process involved several key steps:

 Data Preprocessing and Augmentation: TO create diverse set of images, I applied transformations such as random cropping, flipping, rotation,
				 Gaussian blurring, and normalization to the training data for data augmentation. The validation data underwent
				  only resizing, cropping, and normalization. This step was crucial for improving generalization.

Data Loading: Using PyTorch's DataLoader class, I loaded the data from the specified directory, creating loaders for both the training and validation sets.

Model Creation: I utilized a pre-trained ResNet18 model and customized it by replacing the final fully connected layer 
   		    with a new one matching the number of classes in my dataset. I also added a dropout layer for regularization.

Model Training: The training loop included forward passes, loss computation using cross-entropy, backward propagation, and weight updates via the SGD optimizer. 
    		Training was performed for 20 epochs, with an added learning rate scheduler to adjust the learning rate every 7 epochs.

Model Evaluation: After training, I evaluated the model's performance on the validation data by computing metrics such as accuracy, precision, recall, F1 score, and log loss.

 Model Saving: Finally, I saved the best-performing model's weights.

The model showed improved accuracy and stability due to data augmentation, regularization, and dynamic learning rate adjustments. This workflow demonstrates an end-to-end supervised learning pipeline using PyTorch and highlights techniques to enhance model generalization.