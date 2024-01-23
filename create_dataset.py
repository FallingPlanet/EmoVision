import os
from PIL import Image
from torchvision import transforms as T
import torch
from sklearn.model_selection import train_test_split
from transformers import ViTImageProcessor

def get_label_from_path(file_path):
    labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
    for label in labels:
        if label.lower() in file_path.lower():
            return label
    return None  # Return None if no label is found

def load_and_preprocess_images(file_paths, image_size, output_folder,label_dict, num_augmented_copies=2, is_train=False):
    feature_extractor = ViTImageProcessor(size=image_size, do_rescale=False)

    pil_transforms = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomRotation(5),
        T.RandomResizedCrop(image_size, scale=(0.85, 1.0)),
        T.RandomAffine(degrees=0, translate=(0.02, 0.02), scale=(0.98, 1.02), shear=5),
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.01),
        T.GaussianBlur(kernel_size=(3, 5), sigma=(0.1, 2))
    ])

    tensor_transforms = T.Compose([
        T.RandomErasing(p=0.05, scale=(0.01, 0.05), ratio=(0.3, 3.3), value=0, inplace=False)
    ])

    for idx, file_path in enumerate(file_paths):
        label = get_label_from_path(file_path)
        if label is not None:
            int_label = label_dict[label]  # Get integer label from label_dict
            image = Image.open(file_path).convert("RGB")
            img_tensor = T.ToTensor()(image).to(torch.float32)
            processed_original = feature_extractor(images=img_tensor.unsqueeze(0), return_tensors="pt")["pixel_values"].squeeze(0)

            save_image_with_label(processed_original, int_label, os.path.join(output_folder, f'image_{idx}.pt'))

            if is_train:
                for aug_idx in range(num_augmented_copies):
                    aug_img = pil_transforms(image)
                    img_tensor = T.ToTensor()(aug_img).to(torch.float32)
                    img_tensor = tensor_transforms(img_tensor)
                    processed_aug = feature_extractor(images=img_tensor.unsqueeze(0), return_tensors="pt")["pixel_values"].squeeze(0)
                    save_image_with_label(processed_aug, int_label, os.path.join(output_folder, f'image_{idx}_aug_{aug_idx}.pt'))  # Use int_label

def save_image_with_label(image_tensor, label, file_path):
    label_tensor = torch.tensor(label, dtype=torch.long)  # Convert label to tensor
    torch.save({"image": image_tensor, "label": label_tensor}, file_path)


def split_dataset(image_root_folder, exclude_label):
    all_file_paths = []
    all_labels = []
    label_dict = {}

    for root, dirs, files in os.walk(image_root_folder):
        for file in files:
            if file.endswith((".jpg", ".png", ".jpeg")):
                label = get_label_from_path(os.path.join(root, file))
                if label and label != exclude_label.lower():
                    file_path = os.path.join(root, file)
                    all_file_paths.append(file_path)
                    int_label = label_dict.setdefault(label, len(label_dict))
                    all_labels.append(int_label)

    return train_test_split(all_file_paths, all_labels, train_size=0.7, random_state=42)

def split_and_batch_dataset(image_root_folder, output_folders, image_size, exclude_label):
    labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
    label_dict = {label: i for i, label in enumerate(labels)}

    train_paths, test_val_paths, train_labels, test_val_labels = split_dataset(image_root_folder, exclude_label)
    val_paths, test_paths, val_labels, test_labels = train_test_split(test_val_paths, test_val_labels, train_size=0.5, random_state=42)

    load_and_preprocess_images(train_paths, image_size, output_folders['train'], label_dict, is_train=True, num_augmented_copies=3)
    load_and_preprocess_images(val_paths, image_size, output_folders['val'], label_dict, is_train=False)
    load_and_preprocess_images(test_paths, image_size, output_folders['test'], label_dict, is_train=False)



# Example usage
file_path = r"E:\facial_recognition_datasets\emotion"
output_folders = {
    'train': r"E:\facial_recognition_datasets\train_set_augmented",
    'val': r"E:\facial_recognition_datasets\val_set_augmented",
    'test': r"E:\facial_recognition_datasets\test_set_augmented"
}

split_and_batch_dataset(file_path, output_folders, image_size=224, exclude_label="contempt")
   