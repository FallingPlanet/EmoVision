# Standard library imports (if any)
import os
# Third-party library imports
import torch
import torch.nn as nn
from transformers import BertForSequenceClassification, BertTokenizerFast
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, TensorDataset
# Local application/library s
from FallingPlanet.orbit.utils.Metrics import AdvancedMetrics
from FallingPlanet.orbit.utils.Metrics import TinyEmoBoard
import torchmetrics
from tqdm import tqdm
from FallingPlanet.orbit.utils.callbacks import EarlyStopping
from FallingPlanet.orbit.models import DeitFineTuneTiny
from itertools import islice

class Classifier:
    def __init__(self,model, device, num_labels, log_dir):
        self.model = model.to(device)
        self.device = device
        self.loss_criterion = CrossEntropyLoss()
        self.writer = TinyEmoBoard(log_dir=log_dir)
        
        
        self.accuracy = torchmetrics.Accuracy(num_classes=num_labels, task='multiclass').to(device)
        self.precision = torchmetrics.Precision(num_classes=num_labels, task='multiclass').to(device)
        self.recall = torchmetrics.Recall(num_classes=num_labels, task='multiclass').to(device)
        self.f1= torchmetrics.F1Score(num_classes=num_labels, task = 'multiclass').to(device)
        self.mcc = torchmetrics.MatthewsCorrCoef(num_classes=num_labels,task = 'multiclass').to(device)
        self.top2_acc = torchmetrics.Accuracy(top_k=2, num_classes=num_labels,task='multiclass').to(device)
        
    def compute_loss(self,logits, labels):
        loss = self.loss_criterion(logits,labels)
        return loss
    
    def train_step(self, dataloader, optimizer, epoch):
        self.model.train()
        total_loss = 0.0
        # Initialize metric accumulators
        total_accuracy = 0.0
        total_precision = 0.0
        total_recall = 0.0
        total_f1 = 0.0
        total_mcc = 0.0
        total_batches = 0.0

        for data in dataloader:
            pbar = tqdm(data, desc=f"Training Epoch {epoch}")

            for batch in pbar:
                input_ids, attention_masks, labels = [x.to(self.device) for x in batch]

                optimizer.zero_grad()
                outputs = self.model(input_ids, attention_masks)
                loss = self.compute_loss(outputs, labels)
                loss.backward()
                optimizer.step()


                total_loss += loss.item()

                # Update and accumulate metrics
                total_accuracy += self.accuracy(outputs.argmax(dim=1), labels).item()
                total_precision += self.precision(outputs.argmax(dim=1), labels).item()
                total_recall += self.recall(outputs.argmax(dim=1), labels).item()
                total_f1 += self.f1(outputs, labels).item()
                total_mcc += self.mcc(outputs.argmax(dim=1), labels).item()

                # Update tqdm description with current loss and metrics
                pbar.set_postfix(loss=total_loss / (pbar.n + 1))
                total_batches+=1
            pbar.close()

        # Calculate averages
        num_batches = total_batches
        avg_accuracy = total_accuracy / num_batches
        avg_precision = total_precision / num_batches
        avg_recall = total_recall / num_batches
        avg_f1 = total_f1 / num_batches
        avg_mcc = total_mcc / num_batches
        avg_train_loss = total_loss / num_batches

        # Log metrics to TensorBoard
        self.writer.log_scalar('Training/Average Loss', avg_train_loss, epoch)
        self.writer.log_scalar('Training/Average Accuracy', avg_accuracy, epoch)
        self.writer.log_scalar('Training/Average Precision', avg_precision, epoch)
        self.writer.log_scalar('Training/Average Recall', avg_recall, epoch)
        self.writer.log_scalar('Training/Average F1', avg_f1, epoch)
        self.writer.log_scalar('Training/Average MCC', avg_mcc, epoch)

        


    def val_step(self, dataloader, epoch):
        self.model.eval()
        total_loss = 0.0
        # Initialize metric accumulators
        total_accuracy = 0.0
        total_precision = 0.0
        total_recall = 0.0
        total_f1 = 0.0
        total_mcc = 0.0
        num_batches = 0.0

        with torch.no_grad():
            for data in dataloader:
                pbar = tqdm(data, desc=f"Validation Epoch {epoch}")
                for batch in pbar:
                    input_ids, attention_masks, labels = [x.to(self.device) for x in batch]
                    
                    outputs = self.model(input_ids, attention_masks)
                    loss = self.compute_loss(outputs, labels)

                    total_loss += loss.item()

                    # Update and accumulate metrics
                    total_accuracy += self.accuracy(outputs.argmax(dim=1), labels).item()
                    total_precision += self.precision(outputs.argmax(dim=1), labels).item()
                    total_recall += self.recall(outputs.argmax(dim=1), labels).item()
                    total_f1 += self.f1(outputs, labels).item()
                    total_mcc += self.mcc(outputs.argmax(dim=1), labels).item()

                    # Update tqdm description with current loss and metrics
                    pbar.set_postfix(loss=total_loss / (pbar.n + 1))
                    num_batches +=1
                pbar.close()

        # Calculate averages
        num_batches = num_batches
        avg_val_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        avg_precision = total_precision / num_batches
        avg_recall = total_recall / num_batches
        avg_f1 = total_f1 / num_batches
        avg_mcc = total_mcc / num_batches

        # Log metrics to TensorBoard
        self.writer.log_scalar('Validation/Average Loss', avg_val_loss, epoch)
        self.writer.log_scalar('Validation/Average Accuracy', avg_accuracy, epoch)
        self.writer.log_scalar('Validation/Average Precision', avg_precision, epoch)
        self.writer.log_scalar('Validation/Average Recall', avg_recall, epoch)
        self.writer.log_scalar('Validation/Average F1', avg_f1, epoch)
        self.writer.log_scalar('Validation/Average MCC', avg_mcc, epoch)

        
        return avg_val_loss
 
        
    def test_step(self, dataloader):
        self.model.eval()
        # Initialize aggregated metrics
        aggregated_metrics = {
            'total_accuracy': 0.0,
            'total_precision': 0.0,
            'total_recall': 0.0,
            'total_f1': 0.0,
            'total_mcc': 0.0,
            'total_top_2_acc': 0.0
        }

        with torch.no_grad():
            for data in dataloader:
                pbar = tqdm(data, desc="Testing")
                for batch in pbar:
                    input_ids, attention_masks, labels = [x.to(self.device) for x in batch]
                    outputs = self.model(input_ids, attention_masks)

                    # Update and accumulate metrics
                    aggregated_metrics['total_accuracy'] += self.accuracy(outputs.argmax(dim=1), labels).item()
                    aggregated_metrics['total_precision'] += self.precision(outputs.argmax(dim=1), labels).item()
                    aggregated_metrics['total_recall'] += self.recall(outputs.argmax(dim=1), labels).item()
                    aggregated_metrics['total_f1'] += self.f1(outputs, labels).item()
                    aggregated_metrics['total_mcc'] += self.mcc(outputs.argmax(dim=1), labels).item()
                    aggregated_metrics['total_top_2_acc'] += self.top2_acc(outputs, labels).item()

                    # Update tqdm description with current metrics
                    pbar.set_postfix({
                        'Accuracy': aggregated_metrics['total_accuracy'] / (pbar.n + 1),
                        'MCC': aggregated_metrics['total_mcc'] / (pbar.n + 1)
                    })

        # Calculate average metrics
        num_batches = len(dataloader)
        for key in aggregated_metrics:
            aggregated_metrics[key] /= num_batches

        return aggregated_metrics

def create_dataloaders(file_path, batch_size=64):
    dataloaders = []
    for filename in os.listdir(file_path):
        if filename.endswith('.pt'):
            full_path = os.path.join(file_path, filename)
            data, labels = torch.load(full_path)
            dataset = TensorDataset(data, labels)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            dataloaders.append(dataloader)
    return dataloaders
    
def main(mode = "full"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_file_path = r"E:\facial_recognition_datasets\train_set"
    test_file_path = r"E:\facial_recognition_datasets\test_set"
    val_file_path = r"E:\facial_recognition_datasets\val_set"
    
    train_dataloaders = create_dataloaders(train_file_path,batch_size=64)
    val_dataloaders = create_dataloaders(val_file_path,batch_size=64)
    test_dataloaders= create_dataloaders(test_file_path,batch_size=64)


    
    

   
    
    
    
 
    NUM_EMOTION_LABELS = 9
    LOG_DIR = r"EmoBERTv2-tiny\logging"
    

    model = DeitFineTuneTiny(num_tasks=1, num_labels=[9])
    optimizer = torch.optim.AdamW(model.parameters(),lr =1e-5, weight_decay=1e-6)
    classifier = Classifier(model, device,  NUM_EMOTION_LABELS, LOG_DIR)

    if mode in ["train", "full"]:
        # Your training logic here
        early_stopping = EarlyStopping(patience=50, min_delta=1e-8)  # Initialize Early Stopping
        num_epochs = 75
        for epoch in range(num_epochs):
            classifier.train_step(train_dataloaders, optimizer, epoch)
            val_loss = classifier.val_step(val_dataloaders, epoch)

            if early_stopping.step(val_loss, classifier.model):
                print("Early stopping triggered. Restoring best model weights.")
                classifier.model.load_state_dict(early_stopping.best_state)
                break

        if early_stopping.best_state is not None:
            torch.save(early_stopping.best_state, 'EmoVision-tiny.pth')

    if mode in ["test", "full"]:
        if os.path.exists('EmoVision-tiny.pth'):
            classifier.model.load_state_dict(torch.load('EmoVision-tiny.pth'))
    # Assuming you have test_step implemented in classifier
    test_results = classifier.test_step(test_dataloaders)
    print("Test Results:", test_results)


if __name__ == "__main__":
    main(mode="full")  # or "train" or "test"  