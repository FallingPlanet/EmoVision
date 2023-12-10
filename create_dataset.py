import os
from PIL import Image
from torchvision import transforms
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
from transformers import ViTImageProcessor

def load_and_preprocess_images(image_root_folder, image_size, exclude_label):
    feature_extractor = ViTImageProcessor(size=image_size)
    all_images = []
    all_labels = []
    label_dict = {}

    for root, dirs, files in os.walk(image_root_folder):
        for file in files:
            if file.endswith((".jpg", ".png", ".jpeg")):
                label = os.path.basename(root)
                if label != exclude_label:
                    image_path = os.path.join(root, file)
                    image = Image.open(image_path).convert("RGB")
                    image_tensor = feature_extractor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
                    all_images.append(image_tensor)
                    int_label = label_dict.setdefault(label, len(label_dict))
                    all_labels.append(int_label)

    return all_images, all_labels

def save_batches(images, labels, output_folder, batch_size):
    batch_counter = 0
    for i in range(0, len(images), batch_size):
        batch_images = images[i:i+batch_size]
        batch_labels = labels[i:i+batch_size]
        batch_output_file = os.path.join(output_folder, f'batch_{batch_counter}.pt')
        torch.save((torch.stack(batch_images), torch.tensor(batch_labels)), batch_output_file)
        batch_counter += 1

def split_and_batch_dataset(image_root_folder, output_folders, batch_size, image_size, exclude_label):
    all_images, all_labels = load_and_preprocess_images(image_root_folder, image_size, exclude_label)
    X_train, X_temp, y_train, y_temp = train_test_split(all_images, all_labels, train_size=0.7)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, train_size=0.5)
    save_batches(X_train, y_train, output_folders['train'], batch_size)
    save_batches(X_val, y_val, output_folders['val'], batch_size)
    save_batches(X_test, y_test, output_folders['test'], batch_size)

# Example usage
file_path = r"E:\facial_recognition_datasets\emotion"
output_folders = {
    'train': r"E:\facial_recognition_datasets\train_set",
    'val': r"E:\facial_recognition_datasets\val_set",
    'test': r"E:\facial_recognition_datasets\test_set"
}
split_and_batch_dataset(file_path, output_folders, batch_size=1000, image_size=224, exclude_label="contempt")
   