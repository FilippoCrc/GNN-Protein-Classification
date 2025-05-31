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
set_seed(777)

def add_zeros(data):
    data.x = torch.zeros(data.num_nodes, dtype=torch.long)
    return data

def train(data_loader, model, optimizer, criterion, device, save_checkpoints, checkpoint_path_prefix, current_epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for data in tqdm(data_loader, desc="Iterating training graphs", unit="batch"):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += (pred == data.y).sum().item()
        total += data.y.size(0)

    if save_checkpoints:
        # checkpoint_path_prefix will be like /kaggle/working/checkpoints/A/model_A
        checkpoint_file = f"{checkpoint_path_prefix}_epoch_{current_epoch + 1}.pth"
        torch.save(model.state_dict(), checkpoint_file)
        print(f"Checkpoint saved at {checkpoint_file}")

    return total_loss / len(data_loader), correct / total

def evaluate(data_loader, model, device, calculate_accuracy=False):
    model.eval()
    correct = 0
    total = 0
    predictions = []
    total_loss = 0
    if calculate_accuracy:
        criterion_eval = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for data in tqdm(data_loader, desc="Iterating eval graphs", unit="batch"):
            data = data.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            
            if calculate_accuracy:
                if hasattr(data, 'y') and data.y is not None:
                    correct += (pred == data.y).sum().item()
                    total += data.y.size(0)
                    total_loss += criterion_eval(output, data.y).item()
                else: 
                    pass 
            else:
                predictions.extend(pred.cpu().numpy())
    
    if calculate_accuracy:
        if total == 0:
             return 0.0, 0.0 
        accuracy = correct / total
        return total_loss / len(data_loader), accuracy
    return predictions

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

def plot_training_progress(losses, accuracies, plot_output_dir, plot_type="Training"):
    epochs = range(1, len(losses) + 1)
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, losses, label=f"{plot_type} Loss", color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{plot_type} Loss per Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracies, label=f"{plot_type} Accuracy", color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'{plot_type} Accuracy per Epoch')
    plt.legend()

    os.makedirs(plot_output_dir, exist_ok=True) # Ensure the specific plot directory exists
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

""" class ForwardCorrectionLoss(torch.nn.Module):
    # ... (omitted for brevity, unchanged)
"""

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

    # Setup device
    if args.device == 'cpu':
        device = torch.device("cpu")
    else:
        try:
            device_id = int(args.device)
            if torch.cuda.is_available() and device_id >= 0 and device_id < torch.cuda.device_count():
                device = torch.device(f"cuda:{device_id}")
            else:
                print(f"CUDA device {args.device} not available or invalid. Using CPU.")
                device = torch.device("cpu")
        except ValueError:
            print(f"Invalid device_id '{args.device}'. Using CPU.")
            device = torch.device("cpu")
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
        
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Define criterion
    if args.baseline_mode == 1:
        print(f"Using CrossEntropyLoss")
        criterion = torch.nn.CrossEntropyLoss()
    elif args.baseline_mode == 2:
        print(f"Using GeneralizedCrossEntropy with q={args.loss_param}")
        criterion = GeneralizedCrossEntropy(q=args.loss_param)
    # elif args.baseline_mode == 3: # If ForwardCorrectionLoss is re-enabled
    #     print(f"Using ForwardCorrectionLoss with noise_rate={args.loss_param}")
    #     criterion = ForwardCorrectionLoss(noise_rate=args.loss_param, num_classes=args.num_classes)
    else:
        raise ValueError(f"Invalid baseline_mode: {args.baseline_mode}. Choose 1 (CE) or 2 (GCE).") # Adjusted message if FC is out

    # Kaggle specific paths
    log_dir = "/kaggle/working/logs"
    plot_dir_base = "/kaggle/working/plots" # Base directory for all plots
    plot_dir_specific = os.path.join(plot_dir_base, test_set_identifier) # Plots for this specific test set
    checkpoints_base_dir = "/kaggle/working/checkpoints"
    
    # Log file path
    log_file = os.path.join(log_dir, "training.log") # Single log file, appends
    os.makedirs(log_dir, exist_ok=True)
    
    # Setup logging
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filemode='a')
    logging.getLogger().addHandler(logging.StreamHandler())

    # Checkpoint paths
    # Intermediate checkpoints: /kaggle/working/checkpoints/{test_set_identifier}/model_{test_set_identifier}_epoch_X.pth
    checkpoints_specific_dir = os.path.join(checkpoints_base_dir, test_set_identifier)
    # Best model checkpoint: /kaggle/working/checkpoints/model_{test_set_identifier}_best.pth
    best_model_checkpoint_path = os.path.join(checkpoints_base_dir, f"model_{test_set_identifier}_best.pth")
    
    os.makedirs(checkpoints_base_dir, exist_ok=True) # Ensure base checkpoints dir exists
    os.makedirs(checkpoints_specific_dir, exist_ok=True) # Ensure specific test set's checkpoints dir exists
    # plot_dir_specific will be created by plot_training_progress if needed

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

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        num_epochs = args.epochs
        best_val_accuracy = 0.0   

        train_losses, train_accuracies = [], []
        val_losses, val_accuracies = [], []

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
            
            # Prefix for intermediate checkpoints, e.g. /kaggle/working/checkpoints/A/model_A
            intermediate_checkpoint_prefix = os.path.join(checkpoints_specific_dir, f"model_{test_set_identifier}")

            train_loss, train_acc = train(
                train_loader, model, optimizer, criterion, device,
                save_checkpoints=should_save_checkpoint_this_epoch,
                checkpoint_path_prefix=intermediate_checkpoint_prefix,
                current_epoch=epoch
            )

            val_loss, val_acc = evaluate(val_loader, model, device, calculate_accuracy=True)

            log_msg = (f"Epoch {epoch + 1}/{num_epochs}, TestSetID: {test_set_identifier}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                       f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print(log_msg)
            logging.info(log_msg)

            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                torch.save(model.state_dict(), best_model_checkpoint_path)
                print(f"Best model updated and saved at {best_model_checkpoint_path} (Val Acc: {best_val_accuracy:.4f})")
                logging.info(f"Best model updated for {test_set_identifier} and saved. Val Acc: {best_val_accuracy:.4f}")
        
        # Ensure the specific plot directory for this test_set_identifier is created by plot_training_progress
        plot_training_progress(train_losses, train_accuracies, plot_dir_specific, plot_type="Training")
        plot_training_progress(val_losses, val_accuracies, plot_dir_specific, plot_type="Validation")
        logging.info(f"Training finished for {test_set_identifier}.")

        del train_dataset, val_dataset, train_loader, val_loader, full_dataset
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logging.info(f"Loading test dataset {args.test_path} for final evaluation/prediction...")
    test_dataset = GraphDataset(args.test_path, transform=add_zeros)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    if os.path.exists(best_model_checkpoint_path):
        print(f"Loading best model from {best_model_checkpoint_path} for test set {test_set_identifier} prediction.")
        logging.info(f"Loading best model from {best_model_checkpoint_path} for test set {test_set_identifier} prediction.")
        model.load_state_dict(torch.load(best_model_checkpoint_path, map_location=device))
    else:
        print(f"No best model checkpoint found at {best_model_checkpoint_path}. Using the current model state for predictions.")
        logging.warning(f"No best model checkpoint found for test set {test_set_identifier} predictions.")

    predictions = evaluate(test_loader, model, device, calculate_accuracy=False)
    save_predictions(predictions, test_set_identifier) # Pass test_set_identifier
    logging.info(f"Predictions saved for test set {test_set_identifier}.")
    print("Script finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate GNN models on graph datasets with optional robust loss functions.")
    
    parser.add_argument("--train_path", type=str, help="Path to the training dataset (optional). If not provided, only testing is performed if a model exists.")
    parser.add_argument("--test_path", type=str, required=True, help="Path to the test dataset.")
    
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train (default: 200).') # Default was 25, changed to 200 as in notebook
    parser.add_argument('--batch_size', type=int, default=32, help='Input batch size for training (default: 32).')
    parser.add_argument("--num_checkpoints", type=int, default=3, help="Number of intermediate checkpoints to save. 0 for no intermediate.")
    
    parser.add_argument('--gnn', type=str, default='gin', choices=['gin', 'gin-virtual', 'gcn', 'gcn-virtual'], help='GNN architecture (default: gin).')
    parser.add_argument('--num_layer', type=int, default=5, help='Number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300, help='Dimensionality of hidden units (default: 300).') # Default was 300, common examples use 100 or 64
    parser.add_argument('--drop_ratio', type=float, default=0.5, help='Dropout ratio (default: 0.5).') # Default was 0.0
    parser.add_argument('--num_classes', type=int, default=6, help="Number of classes (default: 6).")

    parser.add_argument('--baseline_mode', type=int, default=2, choices=[1, 2], help="Loss mode: 1 (CrossEntropy), 2 (GeneralizedCrossEntropy) (default: 2).") # Removed 3 if FC is commented out
    parser.add_argument('--loss_param', type=float, default=0.7, help="Parameter for GCE (q value) (default: 0.7).")
    
    parser.add_argument('--device', type=str, default='0', help="CUDA device index ('0', '1', ..) or 'cpu' (default: '0').") # Default to '0' for GPU if available
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for DataLoader (default: 0)')

    args = parser.parse_args()
    
    # Example for local testing if needed (uncomment and adjust)
    # args.train_path = "datasets/A/train.json.gz"
    # args.test_path = "datasets/A/test.json.gz"
    # args.epochs = 2 
    # args.num_checkpoints = 1
    # args.device = 'cpu'
    # args.gnn = 'gin'
    # args.baseline_mode = 1
    # args.num_classes = 6 # Adjust if your dataset A has different number of classes

    main(args)
