import argparse
import os
import torch
from torch_geometric.loader import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm
from sklearn.metrics import f1_score 

from src.loadData import GraphDataset
from src.utils import set_seed
from src.loss_functions import *
from src.models import GNN 

set_seed()

def add_zeros(data):
    data.x = torch.zeros(data.num_nodes, dtype=torch.long)
    return data

def train(data_loader, model, optimizer, criterion, device, save_checkpoints, checkpoint_path, current_epoch):
    model.train()
    total_loss = 0
    for data in tqdm(data_loader, desc="Iterating training graphs", unit="batch"):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Save checkpoints if required
    if save_checkpoints:
        checkpoint_file = f"{checkpoint_path}_epoch_{current_epoch + 1}.pth"
        torch.save(model.state_dict(), checkpoint_file)
        print(f"Checkpoint saved at {checkpoint_file}")

    return total_loss / len(data_loader)

def evaluate(data_loader, model, device, calculate_accuracy=False):
    model.eval()
    # Use distinct names for lists to avoid confusion if 'predictions' were a return var name
    collected_predictions = [] 
    collected_true_labels = [] # Only populated if calculate_accuracy is True
    
    # For accuracy calculation
    correct = 0 
    total = 0   

    with torch.no_grad():
        for data in tqdm(data_loader, desc="Iterating eval graphs", unit="batch"):
            data = data.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            collected_predictions.extend(pred.cpu().numpy()) # The list `collected_predictions` is populated

            if calculate_accuracy:
                collected_true_labels.extend(data.y.cpu().numpy())
                correct += (pred == data.y).sum().item()
                total += data.y.size(0)
        
    if calculate_accuracy:
        accuracy = 0.0
        if total > 0:
            accuracy = correct / total
        
        f1 = 0.0
        if collected_true_labels: # Check if list is not empty to prevent error with f1_score
            f1 = f1_score(collected_true_labels, collected_predictions, average='weighted', zero_division=0)
        return accuracy, f1
    else:
        # When not calculating accuracy, return only the list of predictions
        return collected_predictions

def save_predictions(predictions, test_path):
    script_dir = "/kaggle/working" #os.path.dirname(os.path.abspath(__file__))
    submission_folder = os.path.join(script_dir, "submission")
    test_dir_name = os.path.basename(os.path.dirname(test_path))
    
    os.makedirs(submission_folder, exist_ok=True)
    
    output_csv_path = os.path.join(submission_folder, f"testset_{test_dir_name}.csv")
    
    test_graph_ids = list(range(len(predictions)))
    output_df = pd.DataFrame({
        "id": test_graph_ids,
        "pred": predictions
    })
    
    output_df.to_csv(output_csv_path, index=False)
    print(f"Predictions saved to {output_csv_path}")


def plot_training_progress(train_losses, train_accuracies, output_dir):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 6))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Training Loss", color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss per Epoch')

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label="Training Accuracy", color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy per Epoch')

    # Save plots in the current directory
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_progress.png"))
    plt.close()

def main(args):
    # Get the directory where the main script is located
    script_dir = "/kaggle/working" #os.path.dirname(os.path.abspath(__file__))
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    num_checkpoints = args.num_checkpoints if args.num_checkpoints else 5
    
    

    if args.gnn == 'gin':
        model = GNN(gnn_type = 'gin', num_class = 6, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False).to(device)
    elif args.gnn == 'gin-virtual':
        model = GNN(gnn_type = 'gin', num_class = 6, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = True).to(device)
    elif args.gnn == 'gcn':
        model = GNN(gnn_type = 'gcn', num_class = 6, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False).to(device)
    elif args.gnn == 'gcn-virtual':
        model = GNN(gnn_type = 'gcn', num_class = 6, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = True).to(device)
    else:
        raise ValueError('Invalid GNN type')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    if args.loss == 'ce':
        criterion = torch.nn.CrossEntropyLoss()
    elif args.loss == 'gce':
        criterion = GeneralizedCrossEntropy(q=args.q)
    elif args.loss == 'sce':
        criterion = SymmetricCrossEntropy(alpha=args.alpha, beta=args.beta)
    elif args.loss == 'NCE':
        criterion = NoisyCrossEntropyLoss(p_noisy=args.noise_prob)  # Example value for p_noisy 
    else:
        raise ValueError("Invalid loss function. Choose from: ce, gce, sce")

    # Identify dataset folder (A, B, C, or D)
    test_dir_name = os.path.basename(os.path.dirname(args.test_path))
    
    # Setup logging
    logs_folder = os.path.join(script_dir, "logs", test_dir_name)
    log_file = os.path.join(logs_folder, "training.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')
    logging.getLogger().addHandler(logging.StreamHandler())  # Console output as well


    # Define checkpoint path relative to the script's directory
    checkpoint_path = os.path.join(script_dir, "checkpoints", f"model_{test_dir_name}_best.pth")
    checkpoints_folder = os.path.join(script_dir, "checkpoints", test_dir_name)
    os.makedirs(checkpoints_folder, exist_ok=True)

    # Load pre-trained model for inference
    if os.path.exists(checkpoint_path) and not args.train_path:
        model.load_state_dict(torch.load(checkpoint_path))
        print(f"Loaded best model from {checkpoint_path}")

    # Prepare test dataset and loader
    test_dataset = GraphDataset(args.test_path, transform=add_zeros)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # If train_path is provided, train the model
    if args.train_path:
        train_dataset = GraphDataset(args.train_path, transform=add_zeros)
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=True)

        
        # Training loop
        num_epochs = args.epochs
        best_f1 = 0.0
        train_losses = []
        train_accuracies = []

        # Calculate intervals for saving checkpoints
        if num_checkpoints > 1:
            checkpoint_intervals = [int((i + 1) * num_epochs / num_checkpoints) for i in range(num_checkpoints)]
        else:
            checkpoint_intervals = [num_epochs]

        for epoch in range(num_epochs):
            train_loss = train(
                train_loader, model, optimizer, criterion, device,
                save_checkpoints=(epoch + 1 in checkpoint_intervals),
                checkpoint_path=os.path.join(checkpoints_folder, f"model_{test_dir_name}"),
                current_epoch=epoch
            )
            
            # Corrected call to evaluate: (data_loader, model, ...)
            # Corrected unpacking: expects (accuracy, f1_score)
            train_acc, train_f1 = evaluate(train_loader, model, device, calculate_accuracy=True)
            val_acc, val_f1 = evaluate(val_loader, model, device, calculate_accuracy=True)

            print(f"Epoch {epoch + 1} - Training Loss: {train_loss:.6f}, "
                  f"Training Accuracy: {train_acc:.6f}, Training F1: {train_f1:.6f}, "
                  f"Validation Accuracy: {val_acc:.6f}, Validation F1: {val_f1:.6f}")
            
            # Save logs for training progress
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)

            
            logging.info(f"Epoch {epoch + 1} - Training Loss: {train_loss:.4f}, "
                  f"Training Accuracy: {train_acc:.4f}, Training F1: {train_f1:.4f}, "
                  f"Validation Accuracy: {val_acc:.4f}, Validation F1: {val_f1:.4f}")

            # Save best model
            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Best model updated and saved at {checkpoint_path}")

        # Plot training progress in current directory
        plot_training_progress(train_losses, train_accuracies, os.path.join(logs_folder, "plots"))

    # Generate predictions for the test set using the best model
    if not os.path.exists(checkpoint_path):
        print(f"Warning: Checkpoint {checkpoint_path} not found. Predictions might be from an untrained model if training was not run.")
    else:
        model.load_state_dict(torch.load(checkpoint_path))
        
    predictions = evaluate(test_loader, model, device, calculate_accuracy=False)
    save_predictions(predictions, args.test_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate GNN models on graph datasets.")

    parser.add_argument("--num_checkpoints", type=int, help="Number of checkpoints to save during training.")
    parser.add_argument('--device', type=int, default=0, help='which gpu to use if any (default: 0)')

    parser.add_argument("--train_path", type=str, help="Path to the training dataset (optional).")
    parser.add_argument("--test_path", type=str, required=True, help="Path to the test dataset.")
    
    parser.add_argument('--loss', type=str, default='NCE',help='ce, gce, sce, NCE (default: ce)')
    parser.add_argument('--q', type=float, default=0.7, help='q of gce (default: 0.7)')
    parser.add_argument('--alpha', type=float, default=0.5, help='alpha of sce related to ce (default: 0.3)')
    parser.add_argument('--beta', type=float, default=0.5, help='beta of sce related to rce (default: 1.0)')
    parser.add_argument('--noise_prob', type=float, default=0.2, help='NCE noise parameter A,B,C=0.2 D=0.4 (default: 0.1)')

    parser.add_argument('--gnn', type=str, default='gin-virtual', help='GNN gin, gin-virtual, or gcn, gcn-virtual (default: gin-virtual)')
    parser.add_argument('--drop_ratio', type=float, default=0.5, help='dropout ratio (default: 0.2)')
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate A=0.001 B,C,D=0.005')
    parser.add_argument('--wd', type=float, default=1e-4, help='weight decay (default: 1e-4)')

    parser.add_argument('--num_layer', type=int, default=5, help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=300, help='dimensionality of hidden units in GNNs (default: 64)')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training A=16, B,C,D=64')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train (default: 100)')
    
    args = parser.parse_args()
    main(args)


#cd "C:\Users\simon\OneDrive\Desktop\Progetti Ingegneria\DEEP"
#python main.py --train_path "dataset/B/train.json.gz" --test_path "dataset/B/test.json.gz" --loss "gce" --drop_ratio 0.7 --emb_dim 200 --num_layer 4 --epochs 20 --batch_size 64