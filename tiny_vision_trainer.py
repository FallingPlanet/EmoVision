# Standard library imports (if any)
import os
import gc
# Third-party library imports
import torch
import torch.nn as nn
from transformers import BertForSequenceClassification, BertTokenizerFast
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset, Dataset
from torch.cuda.amp import GradScaler, autocast
from transformers import ViTModel
from sklearn.preprocessing import LabelEncoder
# Local application/library s
from FallingPlanet.orbit.utils.Metrics import AdvancedMetrics
from FallingPlanet.orbit.utils.Metrics import TinyEmoBoard
import torchmetrics
from tqdm import tqdm
from FallingPlanet.orbit.utils.callbacks import EarlyStopping
from FallingPlanet.orbit.models import DeitFineTuneTiny
from itertools import islice



class EmoVision(nn.Module):
    def __init__(self, num_labels, image_size=224, from_saved_weights=None):
        super(EmoVision, self).__init__()
        self.vit = ViTModel.from_pretrained('facebook/deit-tiny-patch16-224')

        if from_saved_weights:
            self.vit.load_state_dict(torch.load(from_saved_weights))

        self.classifier = nn.Sequential(
            nn.Linear(192, 512),  # Adjusted to match DeiT-Tiny's output
            nn.ReLU(),            # Activation function
            nn.Linear(512, 256),  # Second additional dense layer
            nn.ReLU(),            # Activation function
            nn.Linear(256, num_labels)  # Final layer for classification
        )
    
    
    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        pooled_output = outputs.last_hidden_state[:, 0]  # Adjusted to use the correct output
        logits = self.classifier(pooled_output)
        return logits

    
class Classifier:
    def __init__(self,model, device, num_labels, log_dir):
        self.model = model.to(device)
        self.device = device
        self.loss_criterion = CrossEntropyLoss()
        self.writer = TinyEmoBoard(log_dir=log_dir)
        self.scaler = GradScaler()
        
        
        self.accuracy = torchmetrics.Accuracy(num_classes=num_labels, task='multiclass').to(device)
        self.precision = torchmetrics.Precision(average = 'macro' ,num_classes=num_labels, task='multiclass').to(device)
        self.recall = torchmetrics.Recall(average = 'macro', num_classes=num_labels, task='multiclass').to(device)
        self.f1= torchmetrics.F1Score(average = 'macro', num_classes=num_labels, task = 'multiclass').to(device)
        self.mcc = torchmetrics.MatthewsCorrCoef(num_classes=num_labels,task = 'multiclass').to(device)
        self.top2_acc = torchmetrics.Accuracy(top_k=2, num_classes=num_labels,task='multiclass').to(device)
        
    def compute_loss(self,logits, labels):
        loss = self.loss_criterion(logits,labels)
        return loss
    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    from tqdm import tqdm

    def train_step(self, dataloader, optimizer, epoch, accumulation_steps=4):
        self.model.train()
        total_loss = 0.0
        total_accuracy = 0.0
        total_precision = 0.0
        total_recall = 0.0
        total_f1 = 0.0
        total_mcc = 0.0
        total_batches = 0.0

        pbar = tqdm(dataloader, desc=f"Training Epoch {epoch}")
        optimizer.zero_grad()

        for batch in pbar:
            pixel_values, labels = batch
            pixel_values = pixel_values.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(pixel_values)
            loss = self.compute_loss(outputs, labels) / accumulation_steps

            # Check for NaN values in outputs and loss
            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                print("NaN or Inf found in model outputs, skipping batch")
                continue

            if torch.isnan(loss) or torch.isinf(loss):
                print("NaN or Inf found in loss, skipping batch")
                continue

            loss.backward()

            if (total_batches + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * accumulation_steps

            # Detach and accumulate metrics
            total_accuracy += self.accuracy(outputs.detach(), labels).item()
            total_precision += self.precision(outputs.detach(), labels).item()
            total_recall += self.recall(outputs.detach(), labels).item()
            total_f1 += self.f1(outputs.detach(), labels).item()
            total_mcc += self.mcc(outputs.detach(), labels).item()

            pbar.set_postfix(loss=total_loss / (total_batches + 1), accuracy=total_accuracy / (total_batches + 1))

            # Memory cleanup
            del pixel_values, labels, outputs, loss
            gc.collect()
            torch.cuda.empty_cache()  # Free up memory on the GPU

            total_batches += 1

        pbar.close()

        # Calculate average metrics
        avg_train_loss = total_loss / total_batches
        avg_accuracy = total_accuracy / total_batches
        avg_precision = total_precision / total_batches
        avg_recall = total_recall / total_batches
        avg_f1 = total_f1 / total_batches
        avg_mcc = total_mcc / total_batches

        return avg_train_loss

       



        


    def val_step(self, dataloader, epoch):
        self.model.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        total_precision = 0.0
        total_recall = 0.0
        total_f1 = 0.0
        total_mcc = 0.0
        num_batches = 0

        with torch.no_grad():
            pbar = tqdm(dataloader, desc=f"Validation Epoch {epoch}")
            for batch in pbar:
                pixel_values, labels = batch
                pixel_values = pixel_values.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(pixel_values)
                loss = self.compute_loss(outputs, labels)

                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    print("NaN or Inf found in validation outputs, skipping batch")
                    continue

                if torch.isnan(loss) or torch.isinf(loss):
                    print("NaN or Inf found in validation loss, skipping batch")
                    continue

                total_loss += loss.item()
                total_accuracy += self.accuracy(outputs.argmax(dim=1), labels).item()
                total_precision += self.precision(outputs.argmax(dim=1), labels).item()
                total_recall += self.recall(outputs.argmax(dim=1), labels).item()
                total_f1 += self.f1(outputs, labels).item()
                total_mcc += self.mcc(outputs.argmax(dim=1), labels).item()

                num_batches += 1

                current_loss = total_loss / num_batches
                current_accuracy = total_accuracy / num_batches
                pbar.set_postfix(loss=current_loss, accuracy=current_accuracy)

                del pixel_values, labels, outputs, loss  # Free up memory
                gc.collect()  # Invoke garbage collector

            pbar.close()

        avg_val_loss = total_loss / num_batches if num_batches > 0 else 0
        avg_accuracy = total_accuracy / num_batches if num_batches > 0 else 0
        avg_precision = total_precision / num_batches if num_batches > 0 else 0
        avg_recall = total_recall / num_batches if num_batches > 0 else 0
        avg_f1 = total_f1 / num_batches if num_batches > 0 else 0
        avg_mcc = total_mcc / num_batches if num_batches > 0 else 0

        self.writer.log_scalar('Validation/Average Loss', avg_val_loss, epoch)
        self.writer.log_scalar('Validation/Average Accuracy', avg_accuracy, epoch)
        self.writer.log_scalar('Validation/Average Precision', avg_precision, epoch)
        self.writer.log_scalar('Validation/Average Recall', avg_recall, epoch)
        self.writer.log_scalar('Validation/Average F1', avg_f1, epoch)
        self.writer.log_scalar('Validation/Average MCC', avg_mcc, epoch)

        return avg_val_loss


 
        
    def test_step(self, dataloader):
        self.model.eval()
        total_accuracy = 0.0
        total_precision = 0.0
        total_recall = 0.0
        total_f1 = 0.0
        total_mcc = 0.0
        total_top_2_acc = 0.0
        num_batches = 0

        with torch.no_grad():
            pbar = tqdm(dataloader, desc="Testing")
            for batch in pbar:
                # Unpack the batch
                pixel_values, labels = batch
                pixel_values = pixel_values.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(pixel_values)

                # Update and accumulate metrics
                total_accuracy += self.accuracy(outputs.argmax(dim=1), labels).item()
                total_precision += self.precision(outputs.argmax(dim=1), labels).item()
                total_recall += self.recall(outputs.argmax(dim=1), labels).item()
                total_f1 += self.f1(outputs, labels).item()
                total_mcc += self.mcc(outputs.argmax(dim=1), labels).item()
                total_top_2_acc += self.top2_acc(outputs, labels).item()

                num_batches += 1

                # Update tqdm description with current metrics
                pbar.set_postfix({
                    'Accuracy': total_accuracy / num_batches,
                    'MCC': total_mcc / num_batches
                })

        # Calculate average metrics
        avg_accuracy = total_accuracy / num_batches
        avg_precision = total_precision / num_batches
        avg_recall = total_recall / num_batches
        avg_f1 = total_f1 / num_batches
        avg_mcc = total_mcc / num_batches
        avg_top_2_acc = total_top_2_acc / num_batches

        aggregated_metrics = {
            'accuracy': avg_accuracy,
            'precision': avg_precision,
            'recall': avg_recall,
            'f1': avg_f1,
            'mcc': avg_mcc,
            'top_2_accuracy': avg_top_2_acc
        }

        return aggregated_metrics



class CustomDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        data = torch.load(file_path)  # Load data directly from file
        image = data['image']
        label_tensor = data['label']
        return image, label_tensor

def create_combined_dataloader(file_paths, batch_size=4):
    dataset = CustomDataset(file_paths)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)  # Consider adding num_workers for parallel loading
    return dataloader


def main(mode="full"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_folder = "E:\\facial_recognition_datasets\\train_set_augmented"
    val_folder = "E:\\facial_recognition_datasets\\val_set_augmented"
    test_folder = "E:\\facial_recognition_datasets\\test_set_augmented"
    
    train_files = [os.path.join(train_folder, file) for file in os.listdir(train_folder) if file.endswith('.pt')]
    val_files = [os.path.join(val_folder, file) for file in os.listdir(val_folder) if file.endswith('.pt')]
    test_files = [os.path.join(test_folder, file) for file in os.listdir(test_folder) if file.endswith('.pt')]

    train_dataloader = create_combined_dataloader(train_files, batch_size=32)
    val_dataloader = create_combined_dataloader(val_files, batch_size=32)
    test_dataloader = create_combined_dataloader(test_files, batch_size=32)
    
    NUM_EMOTION_LABELS = 8
    LOG_DIR = r"EmoVision\logging"


    model = DeitFineTuneTiny(num_tasks=1 ,num_labels=[NUM_EMOTION_LABELS])
    optimizer = torch.optim.AdamW(model.parameters(),lr =1e-4, weight_decay=1e-10)
    classifier = Classifier(model, device,  NUM_EMOTION_LABELS, LOG_DIR)
    total_params = classifier.count_parameters()
    print(f"Total trainable parameters: {total_params}")
    checkpoint_file = 'latest_checkpoint.pth'  # Same file name used for saving checkpoints
    
    if mode == "test_checkpoint":
        if os.path.exists(checkpoint_file):
            checkpoint = torch.load(checkpoint_file)
            classifier.model.load_state_dict(checkpoint['model_state_dict'])
            print("Testing using the latest checkpoint...")
            test_results = classifier.test_step(test_dataloader)
            print("Test Results:", test_results)
        else:
            print("Checkpoint file not found.")
        return
    if os.path.exists(checkpoint_file):
        checkpoint = torch.load(checkpoint_file)
        classifier.model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        start_step = checkpoint['step']
        print(f"Resuming from epoch {start_epoch}, step {start_step}")
    else:
        start_epoch = 0
        start_step = 0
    if mode in ["train", "full"]:
        # Your training logic here
        early_stopping = EarlyStopping(patience=5, min_delta=1e-11)  # Initialize Early Stopping
        num_epochs = 3
        for epoch in range(num_epochs):
            classifier.train_step(train_dataloader, optimizer, epoch)
            val_loss = classifier.val_step(val_dataloader, epoch)

            if early_stopping.step(val_loss, classifier.model):
                print("Early stopping triggered. Restoring best model weights.")
                classifier.model.load_state_dict(early_stopping.best_state)
                break

        if early_stopping.best_state is not None:
            torch.save(early_stopping.best_state, 'EmoVision_augmented-tiny.pth')
    if mode in ["test", "full"]:
        if os.path.exists('EmoVision_augmented-tiny.pth'):
            classifier.model.load_state_dict(torch.load('EmoVision_augmented-tiny.pth'))
            # Assuming you have test_step implemented in classifier
            test_results = classifier.test_step(test_dataloader)
            print("Test Results:", test_results)
            
    
if __name__ == "__main__":
    main(mode="test")  # or "train" or "test"  