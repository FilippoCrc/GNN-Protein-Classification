

# main2.py
import argparse
import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
import gc # For garbage collection
from pathlib import Path # Added for path manipulation
from sklearn.metrics import f1_score 

# Load utility functions from cloned repository (assuming script is run from 'hackaton' directory)
# Ensure 'src' directory is accessible, e.g. in /kaggle/working/src or added to sys.path
# For Kaggle, if 'src' is uploaded as part of your kernel files, it might be in '../input/your-dataset-name/src'
# or if you upload 'hackaton' folder to /kaggle/working/, then 'src' would be '/kaggle/working/src'
# import sys
# sys.path.append('/kaggle/working/') # If main1.py and src are in /kaggle/working/
from src.loadData import GraphDataset
from src.utils import set_seed
from src.models import GNN

# Set the random seed (as in the notebook)
set_seed()

# we experiment that starting with 1 somehow leads to better results
def add_zeros(data):
    data.x = torch.zeros(data.num_nodes, dtype=torch.long)
    return data

def train(data_loader, model, optimizer, scheduler, criterion, device, save_checkpoints, checkpoint_path_prefix, current_epoch):
    model.train()
    total_loss = 0
    # correct = 0 # Not needed if using all_predictions/all_true_labels for accuracy
    # total = 0   # Not needed if using all_predictions/all_true_labels for accuracy
    all_predictions = []
    all_true_labels = []

    for data in tqdm(data_loader, desc="Iterating training graphs", unit="batch"):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()

        if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        # correct += (pred == data.y).sum().item() # Accumulate for direct accuracy calculation
        # total += data.y.size(0)                  # Accumulate for direct accuracy calculation
        all_predictions.extend(pred.cpu().numpy())
        all_true_labels.extend(data.y.cpu().numpy())


    if save_checkpoints:
        checkpoint_file = f"{checkpoint_path_prefix}_epoch_{current_epoch + 1}.pth"
        torch.save(model.state_dict(), checkpoint_file)
        print(f"Checkpoint saved at {checkpoint_file}")

    # Calculate accuracy and F1 score from collected predictions and labels
    train_accuracy = 0.0
    calculated_train_f1 = 0.0 # Renamed to avoid conflict with list name later if not careful
    if len(all_true_labels) > 0:
        correct_preds = sum(p == t for p, t in zip(all_predictions, all_true_labels))
        train_accuracy = correct_preds / len(all_true_labels)
        calculated_train_f1 = f1_score(all_true_labels, all_predictions, average='weighted', zero_division=0) # Use zero_division
    
    return total_loss / len(data_loader), train_accuracy, calculated_train_f1 # Now returns 3 values

def evaluate(model, data_loader, device, criterion=None, calculate_accuracy=False):
    model.eval()
    total_loss = 0
    num_loss_batches = 0  # To correctly average loss
    correct = 0
    total_samples_with_labels = 0 
    all_predictions = []
    all_true_labels = [] 

    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            output = model(data)
            
            # Calculate loss if criterion is provided and labels are available
            if criterion is not None and hasattr(data, 'y') and data.y is not None:
                loss = criterion(output, data.y)
                total_loss += loss.item()
                num_loss_batches += 1
            
            pred = output.argmax(dim=1)
            all_predictions.extend(pred.cpu().numpy())

            if calculate_accuracy and hasattr(data, 'y') and data.y is not None:
                all_true_labels.extend(data.y.cpu().numpy())
                correct += (pred == data.y).sum().item()
                total_samples_with_labels += data.y.size(0)
        
    avg_loss = total_loss / num_loss_batches if num_loss_batches > 0 else 0.0
    
    accuracy = 0.0
    f1 = 0.0
    
    if calculate_accuracy and total_samples_with_labels > 0:
        accuracy = correct / total_samples_with_labels
        f1 = f1_score(all_true_labels, all_predictions, average='weighted', zero_division=0) # Added zero_division
    elif calculate_accuracy: # This case means total_samples_with_labels is 0
        logging.warning("Accuracy and F1-score could not be calculated because no labeled samples were found for evaluation, despite calculate_accuracy=True.")
            
    return avg_loss, all_predictions, accuracy, f1 # Returns 4 values

def save_predictions(predictions, test_set_identifier):
    submission_folder = "/kaggle/working/submission"
    os.makedirs(submission_folder, exist_ok=True)
    
    output_csv_path = os.path.join(submission_folder, f"testset_{test_set_identifier}.csv")
    
    test_graph_ids = list(range(len(predictions)))
    output_df = pd.DataFrame({
        "id": test_graph_ids,
        "pred": predictions
    })
    
    output_df.to_csv(output_csv_path, index=False)
    print(f"Predictions saved to {output_csv_path}")

def plot_training_progress(losses, accuracies, f1_scores, plot_output_dir, plot_type="Training"): # Added f1_scores
    epochs = range(1, len(losses) + 1)
    plt.figure(figsize=(18, 6)) # Adjusted figsize for 3 plots

    plt.subplot(1, 3, 1)
    plt.plot(epochs, losses, label=f"{plot_type} Loss", color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{plot_type} Loss per Epoch')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs, accuracies, label=f"{plot_type} Accuracy", color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'{plot_type} Accuracy per Epoch')
    plt.legend()

    plt.subplot(1, 3, 3) # New subplot for F1 score
    plt.plot(epochs, f1_scores, label=f"{plot_type} F1 Score", color='red')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title(f'{plot_type} F1 Score per Epoch')
    plt.legend()

    os.makedirs(plot_output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_output_dir, f"{plot_type.lower()}_progress.png"))
    plt.close()

class GeneralizedCrossEntropy(torch.nn.Module):
    def __init__(self, q=0.7):
        super().__init__()
        self.q = q

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        target_probs = probs[torch.arange(targets.size(0)), targets]
        loss = (1 - (target_probs + 1e-7)**self.q) / self.q
        return loss.mean()

class NoisyCrossEntropyLoss(torch.nn.Module):
    def __init__(self, p_noisy):
        super().__init__()
        self.p = p_noisy
        self.ce = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, targets):
        losses = self.ce(logits, targets)
        weights = (1 - self.p) + self.p * (1 - torch.nn.functional.one_hot(targets, num_classes=logits.size(1)).float().sum(dim=1))
        return (losses * weights).mean()

def main(args):
    # Determine a directory name component from the test_path for organizing outputs
    # e.g., for "datasets/A/test.json.gz", we want "A"
    # e.g., for "/kaggle/input/mycompetition/test_public/data.csv", we want "test_public"
    temp_dir_name_from_path = os.path.basename(os.path.dirname(args.test_path))
    if not temp_dir_name_from_path: # If dirname is empty (e.g. test_path is "file.csv" at root)
        # Attempt to use the filename stem as a fallback, e.g. "file"
        temp_dir_name_from_path = Path(args.test_path).stem
        if ".json" in temp_dir_name_from_path: # For cases like "file.json.gz" -> "file.json" -> "file"
            temp_dir_name_from_path = Path(temp_dir_name_from_path).stem
    
    # Final fallback if still empty (e.g. if test_path was just ".gz")
    test_set_identifier = temp_dir_name_from_path if temp_dir_name_from_path else "output"
    print(f"Using test set identifier: {test_set_identifier}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    num_checkpoints_to_save = args.num_checkpoints if args.num_checkpoints is not None else 0
    
    # Define model
    if args.gnn == 'gin':
        model = GNN(gnn_type='gin', num_class=args.num_classes, num_layer=args.num_layer, emb_dim=args.emb_dim, drop_ratio=args.drop_ratio, virtual_node=False).to(device)
    elif args.gnn == 'gin-virtual':
        model = GNN(gnn_type='gin', num_class=args.num_classes, num_layer=args.num_layer, emb_dim=args.emb_dim, drop_ratio=args.drop_ratio, virtual_node=True).to(device)
    elif args.gnn == 'gcn':
        model = GNN(gnn_type='gcn', num_class=args.num_classes, num_layer=args.num_layer, emb_dim=args.emb_dim, drop_ratio=args.drop_ratio, virtual_node=False).to(device)
    elif args.gnn == 'gcn-virtual':
        model = GNN(gnn_type='gcn', num_class=args.num_classes, num_layer=args.num_layer, emb_dim=args.emb_dim, drop_ratio=args.drop_ratio, virtual_node=True).to(device)
    else:
        raise ValueError('Invalid GNN type')
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay) 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=args.scheduler_patience, verbose=True, min_lr=1e-7)

    # Define criterion
    if args.baseline_mode == 1:
        print(f"Using CrossEntropyLoss")
        criterion = torch.nn.CrossEntropyLoss()
    elif args.baseline_mode == 2:
        print(f"Using GeneralizedCrossEntropy with q={args.loss_param}")
        criterion = GeneralizedCrossEntropy(q=args.loss_param)
    elif args.baseline_mode == 3: 
        print(f"Using NoisyCrossEntropyLoss with noise_rate={args.loss_param}")
        criterion = NoisyCrossEntropyLoss(p_noisy=args.loss_param)
    else:
        raise ValueError(f"Invalid baseline_mode: {args.baseline_mode}. Choose 1 (CE), 2 (GCE), or 3 (NCE).")

    # Kaggle specific paths
    log_dir = "/kaggle/working/logs"
    plot_dir_base = "/kaggle/working/plots" 
    plot_dir_specific = os.path.join(plot_dir_base, test_set_identifier) 
    checkpoints_base_dir = "/kaggle/working/checkpoints"
    
    log_file = os.path.join(log_dir, "training.log") 
    os.makedirs(log_dir, exist_ok=True)
    
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filemode='a')
    logging.getLogger().addHandler(logging.StreamHandler())

    checkpoints_specific_dir = os.path.join(checkpoints_base_dir, test_set_identifier)
    best_model_checkpoint_path = os.path.join(checkpoints_base_dir, f"model_{test_set_identifier}_best.pth")
    
    os.makedirs(checkpoints_base_dir, exist_ok=True) 
    os.makedirs(checkpoints_specific_dir, exist_ok=True)

    if os.path.exists(best_model_checkpoint_path) and not args.train_path:
        print(f"Loading best model from {best_model_checkpoint_path} for inference.")
        logging.info(f"Loading best model from {best_model_checkpoint_path} for inference.")
        model.load_state_dict(torch.load(best_model_checkpoint_path, map_location=device))
    
    if args.train_path:
        logging.info("Starting training process...")
        full_dataset = GraphDataset(args.train_path, transform=add_zeros)
        val_size = int(0.2 * len(full_dataset))
        train_size = len(full_dataset) - val_size
        
        generator = torch.Generator().manual_seed(12)
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        num_epochs = args.epochs
        best_val_f1 = 0.0   # Changed from best_val_accuracy to best_val_f1
        logging.info(f"Dataset split: {train_size} training samples, {val_size} validation samples")

        train_losses_history, train_accuracies_history, train_f1_scores_history = [], [], []
        val_losses_history, val_accuracies_history, val_f1_scores_history = [], [], []


        checkpoint_intervals = []
        if num_checkpoints_to_save > 0 and num_epochs > 0:
            if num_checkpoints_to_save >= num_epochs:
                 checkpoint_intervals = list(range(1, num_epochs + 1))
            else:
                 checkpoint_intervals = [int(i * num_epochs / num_checkpoints_to_save) for i in range(1, num_checkpoints_to_save + 1)]
                 if num_epochs not in checkpoint_intervals:
                     checkpoint_intervals = sorted(list(set(checkpoint_intervals + [num_epochs])))

        for epoch in range(num_epochs):
            should_save_checkpoint_this_epoch = (epoch + 1) in checkpoint_intervals
            intermediate_checkpoint_prefix = os.path.join(checkpoints_specific_dir, f"model_{test_set_identifier}")

            # train() returns: loss, accuracy, f1_score
            current_epoch_train_loss, current_epoch_train_acc, current_epoch_train_f1 = train(
                train_loader, model, optimizer, scheduler,
                criterion, device,
                save_checkpoints=should_save_checkpoint_this_epoch,
                checkpoint_path_prefix=intermediate_checkpoint_prefix,
                current_epoch=epoch
            )

            # evaluate() returns: avg_loss, all_predictions, accuracy, f1
            current_epoch_val_loss, _, current_epoch_val_acc, current_epoch_val_f1 = evaluate(
                model, val_loader, device, criterion=criterion, calculate_accuracy=True
            )


            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(current_epoch_val_acc) # scheduler steps on validation accuracy

            log_msg = (f"Epoch {epoch + 1}/{num_epochs}, TestSetID: {test_set_identifier}, Train Loss: {current_epoch_train_loss:.4f}, Train Acc: {current_epoch_train_acc:.4f}, Train F1: {current_epoch_train_f1:.4f}, "
                       f"Val Loss: {current_epoch_val_loss:.4f}, Val Acc: {current_epoch_val_acc:.4f}, Val F1: {current_epoch_val_f1:.4f}")
            print(log_msg)
            logging.info(log_msg)

            train_losses_history.append(current_epoch_train_loss)
            train_accuracies_history.append(current_epoch_train_acc)
            train_f1_scores_history.append(current_epoch_train_f1)
            
            val_losses_history.append(current_epoch_val_loss)
            val_accuracies_history.append(current_epoch_val_acc)
            val_f1_scores_history.append(current_epoch_val_f1)
            
            if current_epoch_val_f1 > best_val_f1:
                best_val_f1 = current_epoch_val_f1
                torch.save(model.state_dict(), best_model_checkpoint_path)
                print(f"Best model updated and saved at {best_model_checkpoint_path} (Val F1: {best_val_f1:.4f})")
                logging.info(f"Best model updated for {test_set_identifier} and saved. Val F1: {best_val_f1:.4f}")
        
        plot_training_progress(train_losses_history, train_accuracies_history, train_f1_scores_history, plot_dir_specific, plot_type="Training")
        plot_training_progress(val_losses_history, val_accuracies_history, val_f1_scores_history, plot_dir_specific, plot_type="Validation")
        logging.info(f"Training finished for {test_set_identifier}.")

        del train_dataset, val_dataset, train_loader, val_loader, full_dataset
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logging.info(f"Loading test dataset {args.test_path} for final evaluation/prediction...")
    test_dataset = GraphDataset(args.test_path, transform=add_zeros)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    if os.path.exists(best_model_checkpoint_path):
        print(f"Loading best model from {best_model_checkpoint_path} for test set {test_set_identifier} prediction.")
        logging.info(f"Loading best model from {best_model_checkpoint_path} for test set {test_set_identifier} prediction.")
        model.load_state_dict(torch.load(best_model_checkpoint_path, map_location=device))
    else:
        print(f"No best model checkpoint found at {best_model_checkpoint_path}. Using the current model state for predictions.")
        logging.warning(f"No best model checkpoint found for test set {test_set_identifier} predictions.")

    # evaluate() returns: avg_loss, all_predictions, accuracy, f1
    # For test set, we primarily need all_predictions. Loss, acc, f1 might be 0 if no labels/criterion.
    _, test_set_predictions, _, _ = evaluate(model, test_loader, device, criterion=None, calculate_accuracy=False)
    save_predictions(test_set_predictions, test_set_identifier) 
    logging.info(f"Predictions saved for test set {test_set_identifier}.")
    print("Script finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate GNN models on graph datasets with optional robust loss functions.")
    
    parser.add_argument("--train_path", type=str, help="Path to the training dataset (optional). If not provided, only testing is performed if a model exists.")
    parser.add_argument("--test_path", type=str, required=True, help="Path to the test dataset.")
    
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train (default: 200).')
    parser.add_argument('--batch_size', type=int, default=64, help='Input batch size for training (default: 32).')
    parser.add_argument("--num_checkpoints", type=int, default=5, help="Number of intermediate checkpoints to save. 0 for no intermediate.")
    
    parser.add_argument('--gnn', type=str, default='gin-virtual', choices=['gin', 'gin-virtual', 'gcn', 'gcn-virtual'], help='GNN architecture (default: gin).')
    parser.add_argument('--num_layer', type=int, default=2, help='Number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=128, help='Dimensionality of hidden units (default: 300).') 
    parser.add_argument('--drop_ratio', type=float, default=0.3, help='Dropout ratio (default: 0.5).') 
    parser.add_argument('--num_classes', type=int, default=6, help="Number of classes (default: 6).")

    parser.add_argument('--baseline_mode', type=int, default=2, choices=[1, 2, 3], help="Loss mode: 1 (CrossEntropy), 2 (GeneralizedCrossEntropy), 3 (NoisyCrossEntropy) (default: 2).")
    parser.add_argument('--loss_param', type=float, default=0.9, help="Parameter for GCE (q value) or NCE (noise_rate) (default: 0.7 for GCE, NCE needs its own default if different).") # Adjusted GCE default to 0.9 as per notebook
    
    parser.add_argument('--device', type=str, default='0', help="CUDA device index ('0', '1', ..) or 'cpu' (default: '0').") 
    #parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for DataLoader (default: 0)')

    parser.add_argument('--num_features', type=int, default=1,help="Dimensionality of input node features. Default is 1 if features are auto-generated as zeros.")

    parser.add_argument('--lr', type=float, default=5e-3, help='Learning rate (default: 0.001).')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay (L2 penalty) (default: 1e-5).')
    parser.add_argument('--scheduler_patience', type=int, default=15, help='Patience for ReduceLROnPlateau scheduler (default: 10).')

    args = parser.parse_args()
    
    # Example for local testing if needed (uncomment and adjust)
    # args.train_path = "datasets/A/train.json.gz"
    # args.test_path = "datasets/A/test.json.gz"
    # args.epochs = 2 
    # args.num_checkpoints = 1
    # args.device = 'cpu' # Forcing CPU if no GPU or for testing
    # args.gnn = 'gin'
    # args.baseline_mode = 1
    # args.num_classes = 6 # Adjust if your dataset A has different number of classes

    main(args)