import os
import random

import torch
import torchvision
from torchvision import datasets, transforms

# Define transformations for the dataset
transform = transforms.Compose(
    [
        transforms.ToTensor(),  # Convert images to tensors
    ]
)

# Load the MNIST training dataset
mnist_train = datasets.MNIST(root="dataset", train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(root="dataset", train=False, download=True, transform=transform)

# Calculate the number of samples per class for the validation set
num_classes = 10  # Number of classes in MNIST
samples_per_class = len(mnist_train) // num_classes
validation_size = samples_per_class * 0.2  # 20% of samples per class for validation

# Initialize lists to store samples for the validation set
mnist_val_samples = []

# Initialize counters for samples per class
class_counters = [0] * num_classes

# Iterate over the training dataset and randomly select samples for the validation set
mnist_train_samples = []
for img, label in mnist_train:
    if class_counters[label] < validation_size:
        mnist_val_samples.append((img, label))
        class_counters[label] += 1
    else:
        mnist_train_samples.append((img, label))


# Randomly shuffle the training samples
random.shuffle(mnist_train_samples)

# Save the validation dataset to a file
torch.save(mnist_train_samples, "dataset/MNIST/mnist_train.pth")
torch.save(mnist_val_samples, "dataset/MNIST/mnist_validation.pth")

mnist_test_samples = []
for img, label in mnist_test:
    mnist_test_samples.append((img, label))

torch.save(mnist_test_samples, "dataset/MNIST/mnist_test.pth")

# # Define a function to save images
# def save_images(dataset, directory):
#     os.makedirs(directory, exist_ok=True)
#     for img, label in dataset:
#         label_dir = os.path.join(directory, str(label))
#         os.makedirs(label_dir, exist_ok=True)
#         img_path = os.path.join(label_dir, f"{len(os.listdir(label_dir))}.png")
#         torchvision.utils.save_image(img, img_path)


# # Save the validation images to the local drive
# save_images(mnist_val, directory="./mnist_validation")
