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

# Load utility functions from cloned repository (assuming script is run from 'hackaton' directory)
from src.loadData import GraphDataset
from src.utils import set_seed
from src.models import GNN

# Set the random seed (as in the notebook)
set_seed(777)

def add_zeros(data):
    data.x = torch.zeros(data.num_nodes, dtype=torch.long)
    return data

def train(data_loader, model, optimizer, criterion, device, save_checkpoints, checkpoint_path, current_epoch):
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
        checkpoint_file = f"{checkpoint_path}_epoch_{current_epoch + 1}.pth"
        torch.save(model.state_dict(), checkpoint_file)
        print(f"Checkpoint saved at {checkpoint_file}")

    return total_loss / len(data_loader), correct / total

def evaluate(data_loader, model, device, calculate_accuracy=False):
    model.eval()
    correct = 0
    total = 0
    predictions = []
    total_loss = 0
    # Re-initialize criterion for evaluation if y is present, to calculate loss
    # This is only used if calculate_accuracy is True and data.y exists
    if calculate_accuracy:
        criterion_eval = torch.nn.CrossEntropyLoss() 
        # Note: If a robust loss was used for training, val loss might ideally use the same.
        # However, for accuracy comparison, CE is standard. The notebook uses CE for eval loss.

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
                else: # Should not happen if loader provides y for val/train
                    pass 
            else:
                predictions.extend(pred.cpu().numpy())
    
    if calculate_accuracy:
        if total == 0: # Avoid division by zero if no labels were processed
             return 0.0, 0.0 # loss, accuracy
        accuracy = correct / total
        return total_loss / len(data_loader), accuracy
    return predictions

def save_predictions(predictions, test_path):
    script_dir = os.path.dirname(os.path.abspath(__file__)) # Use __file__ for .py script
    submission_folder = os.path.join(script_dir, "submission")
    test_dir_name = os.path.basename(os.path.dirname(test_path))
    
    os.makedirs(submission_folder, exist_ok=True)
    
    output_csv_path = os.path.join(submission_folder, f"testset_{test_dir_name}.csv")
    
    # Assuming test graphs are sequentially numbered if IDs are not explicitly available
    test_graph_ids = list(range(len(predictions)))
    output_df = pd.DataFrame({
        "id": test_graph_ids,
        "pred": predictions
    })
    
    output_df.to_csv(output_csv_path, index=False)
    print(f"Predictions saved to {output_csv_path}")

def plot_training_progress(losses, accuracies, output_dir, plot_type="Training"):
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

    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{plot_type.lower()}_progress.png"))
    plt.close()

# Custom Loss Functions from Notebook Cell 17
class GeneralizedCrossEntropy(torch.nn.Module):
    def __init__(self, q=0.7):
        super().__init__()
        self.q = q

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        # Get the probabilities of the target class
        target_probs = probs[torch.arange(targets.size(0)), targets]
        # Add epsilon for stability, especially if target_probs can be 0 and q is small
        loss = (1 - (target_probs + 1e-7)**self.q) / self.q
        return loss.mean()

""" class ForwardCorrectionLoss(torch.nn.Module):
    def __init__(self, noise_rate, num_classes):
        super().__init__()
        self.p = noise_rate
        self.num_classes = num_classes
        self.noise_transition_matrix = self._build_noise_matrix().float() # Ensure float

    def _build_noise_matrix(self):
        # T_ij = P(noisy_label=j | true_label=i)
        if self.num_classes == 1: # Avoid division by zero if num_classes is 1
            return torch.eye(1)
        matrix = torch.full((self.num_classes, self.num_classes), self.p / (self.num_classes - 1))
        for i in range(self.num_classes):
            matrix[i, i] = 1 - self.p
        return matrix

    def forward(self, logits, targets):
        # logits: (batch_size, num_classes)
        # targets: (batch_size,)
        if self.noise_transition_matrix.device != logits.device:
            self.noise_transition_matrix = self.noise_transition_matrix.to(logits.device)

        softmax_preds = torch.softmax(logits, dim=1) # P_model(y_true | x)
        # P(y_noisy | x) = P_model(y_true | x) @ T
        noisy_preds = softmax_preds @ self.noise_transition_matrix
        
        # Add a small epsilon to prevent log(0)
        noisy_preds_clipped = torch.clamp(noisy_preds, min=1e-7, max=1.0)

        # Standard NLL loss on the corrected predictions
        loss = torch.nn.functional.nll_loss(torch.log(noisy_preds_clipped), targets, reduction='mean')
        return loss """

def main(args):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    if args.device == 'cpu':
        device = torch.device("cpu")
    else:
        # Try to use the specified CUDA device, fallback to CPU if not available or error
        try:
            device_id = int(args.device)
            if torch.cuda.is_available() and device_id >= 0 and device_id < torch.cuda.device_count():
                device = torch.device(f"cuda:{device_id}")
            else:
                print(f"CUDA device {args.device} not available. Using CPU.")
                device = torch.device("cpu")
        except ValueError:
            print(f"Invalid device_id '{args.device}'. Using CPU.")
            device = torch.device("cpu")
    print(f"Using device: {device}")

    num_checkpoints_to_save = args.num_checkpoints if args.num_checkpoints is not None else 0
    
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
        
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # lr is hardcoded as in notebook
    
    if args.baseline_mode == 1:
        print(f"Using CrossEntropyLoss")
        criterion = torch.nn.CrossEntropyLoss()
    elif args.baseline_mode == 2:
        print(f"Using GeneralizedCrossEntropy with q={args.loss_param}")
        criterion = GeneralizedCrossEntropy(q=args.loss_param)
    else:
        raise ValueError(f"Invalid baseline_mode: {args.baseline_mode}. Choose 1 (CE), 2 (GCE), or 3 (FC).")

    test_dir_name = os.path.basename(os.path.dirname(args.test_path))
    logs_folder = os.path.join(script_dir, "logs", test_dir_name)
    #log_file = os.path.join(logs_folder, "training.log")
    log_file = "/kaggle/working/logs/training.log"
    
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Setup logging
    # Remove existing handlers before adding new ones to avoid duplicate logs if main is called multiple times
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filemode='a')
    logging.getLogger().addHandler(logging.StreamHandler()) # Console output as well

    # Checkpoint paths
    checkpoints_folder_base = os.path.join(script_dir, "checkpoints")
    checkpoints_folder_specific = os.path.join(checkpoints_folder_base, test_dir_name) # For intermediate checkpoints
    best_model_checkpoint_path = os.path.join(checkpoints_folder_base, f"model_{test_dir_name}_best.pth")
    os.makedirs(checkpoints_folder_specific, exist_ok=True)


    if os.path.exists(best_model_checkpoint_path) and not args.train_path:
        print(f"Loading best model from {best_model_checkpoint_path} for inference.")
        model.load_state_dict(torch.load(best_model_checkpoint_path, map_location=device))
    
    if args.train_path:
        logging.info("Starting training process...")
        full_dataset = GraphDataset(args.train_path, transform=add_zeros)
        # Splitting dataset into training and validation
        val_size = int(0.2 * len(full_dataset)) # 20% for validation
        train_size = len(full_dataset) - val_size
        
        # Use a generator for reproducible splits
        generator = torch.Generator().manual_seed(12) # Seed from notebook
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        num_epochs = args.epochs
        best_val_accuracy = 0.0   

        train_losses, train_accuracies = [], []
        val_losses, val_accuracies = [], []

        checkpoint_intervals = []
        if num_checkpoints_to_save > 0 and num_epochs > 0:
            if num_checkpoints_to_save >= num_epochs: # Save every epoch if more checkpoints than epochs
                 checkpoint_intervals = list(range(1, num_epochs + 1))
            else:
                # Ensure at least one checkpoint if num_checkpoints_to_save = 1 (last epoch)
                 checkpoint_intervals = [int(i * num_epochs / num_checkpoints_to_save) for i in range(1, num_checkpoints_to_save + 1)]
                 if num_epochs not in checkpoint_intervals: # Ensure last epoch is a checkpoint if not already
                     checkpoint_intervals = sorted(list(set(checkpoint_intervals + [num_epochs])))


        for epoch in range(num_epochs):
            should_save_checkpoint_this_epoch = (epoch + 1) in checkpoint_intervals
            
            train_loss, train_acc = train(
                train_loader, model, optimizer, criterion, device,
                save_checkpoints=should_save_checkpoint_this_epoch,
                # Path for intermediate checkpoints (e.g., checkpoints/A/model_A_epoch_1.pth)
                checkpoint_path=os.path.join(checkpoints_folder_specific, f"model_{test_dir_name}"),
                current_epoch=epoch
            )

            val_loss, val_acc = evaluate(val_loader, model, device, calculate_accuracy=True)

            log_msg = (f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
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
                logging.info(f"Best model updated and saved. Val Acc: {best_val_accuracy:.4f}")
        
        plot_training_progress(train_losses, train_accuracies, os.path.join(logs_folder, "plots_train"), plot_type="Training")
        plot_training_progress(val_losses, val_accuracies, os.path.join(logs_folder, "plots_validation"), plot_type="Validation")
        logging.info("Training finished.")

        # Garbage collection
        del train_dataset, val_dataset, train_loader, val_loader, full_dataset
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Prepare test dataset and loader for final evaluation / prediction
    logging.info("Loading test dataset for final evaluation/prediction...")
    test_dataset = GraphDataset(args.test_path, transform=add_zeros)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    # Load the best model for testing
    if os.path.exists(best_model_checkpoint_path):
        print(f"Loading best model from {best_model_checkpoint_path} for test set prediction.")
        model.load_state_dict(torch.load(best_model_checkpoint_path, map_location=device))
    else:
        print("No best model checkpoint found. Using the current model state for predictions (if any training was done).")
        logging.warning("No best model checkpoint found for test predictions.")

    predictions = evaluate(test_loader, model, device, calculate_accuracy=False)
    save_predictions(predictions, args.test_path)
    logging.info(f"Predictions saved for test set {test_dir_name}.")
    print("Script finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate GNN models on graph datasets with optional robust loss functions.")
    
    # Paths
    parser.add_argument("--train_path", type=str, help="Path to the training dataset (optional). If not provided, only testing is performed if a model exists.")
    parser.add_argument("--test_path", type=str, required=True, help="Path to the test dataset.")
    
    # Training Configuration
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train (default: 25).')
    parser.add_argument('--batch_size', type=int, default=32, help='Input batch size for training (default: 32).')
    parser.add_argument("--num_checkpoints", type=int, default=3, help="Number of intermediate checkpoints to save during training (e.g., 3 means 3 checkpoints spread across epochs, plus the best model). 0 for no intermediate ones.")
    
    # Model Configuration
    parser.add_argument('--gnn', type=str, default='gin', choices=['gin', 'gin-virtual', 'gcn', 'gcn-virtual'], help='GNN architecture (default: gin).')
    parser.add_argument('--num_layer', type=int, default=5, help='Number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=100, help='Dimensionality of hidden units in GNNs (default: 300).')
    parser.add_argument('--drop_ratio', type=float, default=0.5, help='Dropout ratio (default: 0.0).')
    parser.add_argument('--num_classes', type=int, default=6, help="Number of classes for the dataset (default: 6).")

    # Loss Configuration
    parser.add_argument('--baseline_mode', type=int, default=2, choices=[1, 2, 3], help="Loss mode: 1 (CrossEntropy), 2 (GeneralizedCrossEntropy), 3 (ForwardCorrection) (default: 1).")
    parser.add_argument('--loss_param', type=float, default=0.7, help="Parameter for robust loss (q for GCE, noise_rate for FC) (default: 0.7).")
    
    # System Configuration
    parser.add_argument('--device', type=str, default='cpu', help="CUDA device index (e.g., '0' or '1') or 'cpu' (default: '0').")
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for DataLoader. 0 for main process. (default: 0)')


    # Print usage if no arguments are provided (or only -h/--help)
    # import sys
    # if len(sys.argv) == 1:
    #     parser.print_help(sys.stderr)
    #     sys.exit(1)

    args = parser.parse_args()
    
    # Example: Manually set args for testing if not running from command line
    # args.train_path = "datasets/A/train.json.gz" # Or None
    # args.test_path = "datasets/A/test.json.gz"
    # args.epochs = 1 # For quick test
    # args.num_checkpoints = 1
    # args.device = 'cpu'
    # args.gnn = 'gin-virtual'
    # args.drop_ratio = 0.5
    # args.num_layer = 4
    # args.emb_dim = 100
    # args.baseline_mode = 1
    # args.loss_param = 0.2
    # args.num_classes = 6
    
    main(args)