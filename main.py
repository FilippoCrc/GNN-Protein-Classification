import argparse
import os
import torch
from torch_geometric.loader import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm
from sklearn.metrics import f1_score

from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

import warnings

from loadData import GraphDataset
from utils import set_seed
from loss_functions import *
from models import GNN

set_seed()

warnings.filterwarnings("ignore", category=UserWarning, module="torch.optim.lr_scheduler")

def add_zeros(data):
    data.x = torch.zeros(data.num_nodes, dtype=torch.long)
    return data


def train(data_loader, model, optimizer, criterion, device, save_checkpoints, checkpoint_path, current_epoch, grad_clip_value=None):

    model.train()
    total_loss = 0
    for data in tqdm(data_loader, desc="Iterating training graphs", unit="batch"):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        if isinstance(criterion, GCODLoss):
            batch_size = data.y.size(0)
            device = data.y.device
            u_params = torch.ones(batch_size, device=device, requires_grad=True)
            training_accuracy = 0.5
            total_loss_val, L1, L2, L3 = criterion(output, data.y, u_params, training_accuracy)
            loss = total_loss_val
            loss.backward()

        else:
            loss = criterion(output, data.y)
            loss.backward()


        if grad_clip_value:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)

        optimizer.step()
        total_loss += loss.item()

    # Save checkpoints if required
    if save_checkpoints:
        checkpoint_file = f"{checkpoint_path}_epoch_{current_epoch + 1}.pth"
        torch.save(model.state_dict(), checkpoint_file)
        print(f"Checkpoint saved at {checkpoint_file}")

    return total_loss / len(data_loader)


def evaluate(data_loader, model, criterion, device, calculate_accuracy=False):

    model.eval()
    collected_predictions = []
    collected_true_labels = []
    
    correct = 0
    total = 0
    total_eval_loss = 0

    with torch.no_grad():
        for data in tqdm(data_loader, desc="Iterating eval graphs", unit="batch"):
            data = data.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            collected_predictions.extend(pred.cpu().numpy())

            if calculate_accuracy:
                if criterion:
                    loss = criterion(output, data.y)
                    total_eval_loss += loss.item()
                collected_true_labels.extend(data.y.cpu().numpy())
                correct += (pred == data.y).sum().item()
                total += data.y.size(0)
        
    if calculate_accuracy:
        accuracy = 0.0
        if total > 0:
            accuracy = correct / total
        
        f1 = 0.0
        if collected_true_labels:
            f1 = f1_score(collected_true_labels, collected_predictions, average='weighted', zero_division=0)
        
        eval_loss = 0.0
        if criterion and len(data_loader) > 0 : # Check if data_loader is not empty
            eval_loss = total_eval_loss / len(data_loader)
        return eval_loss, accuracy, f1

    else:
        return collected_predictions

def save_predictions(predictions, test_path):
    script_dir = os.path.dirname(os.path.abspath(__file__)) #"/kaggle/working"
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

    losses = [l.item() if torch.is_tensor(l) else l for l in train_losses]
    accuracies = [a.item() if torch.is_tensor(a) else a for a in train_accuracies]

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, losses, label="Training Loss", color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss per Epoch')

    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracies, label="Training Accuracy", color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy per Epoch')

    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_progress.png"))
    plt.close()

def main(args):
    script_dir = os.path.dirname(os.path.abspath(__file__)) #"/kaggle/working"
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    num_checkpoints = args.num_checkpoints if args.num_checkpoints else 3

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


    scheduler = None
    if args.scheduler == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='max' if args.scheduler_monitor_metric in ['val_f1', 'val_acc'] else 'min',
                                      factor=args.lr_factor, patience=args.lr_patience, verbose=True, min_lr=args.min_lr)
    elif args.scheduler == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr, verbose=True)

    if args.loss == 'ce':
        criterion = torch.nn.CrossEntropyLoss()
    elif args.loss == 'gce':
        criterion = GeneralizedCrossEntropy(q=args.q)
    elif args.loss == 'sce':
        criterion = SymmetricCrossEntropy(alpha=args.alpha, beta=args.beta)
    elif args.loss == 'gcod':
        criterion = GCODLoss(num_classes=6, alpha_train=0.01, lambda_r=0.1)
    elif args.loss == 'nce':
        criterion = NoisyCrossEntropyLoss(p_noisy = 0.2)
    else:
        raise ValueError("Invalid loss function. Choose from: ce, gce, sce")

    test_dir_name = os.path.basename(os.path.dirname(args.test_path))
    
    logs_folder = os.path.join(script_dir, "logs", test_dir_name)
    log_file = os.path.join(logs_folder, "training.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')
    logging.getLogger().addHandler(logging.StreamHandler())

    checkpoint_path = os.path.join(script_dir, "checkpoints", f"model_{test_dir_name}_best.pth")
    checkpoints_folder = os.path.join(script_dir, "checkpoints", test_dir_name)
    os.makedirs(checkpoints_folder, exist_ok=True)

    if os.path.exists(checkpoint_path) and not args.train_path:
        model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
        print(f"Loaded best model from {checkpoint_path}")

    test_dataset = GraphDataset(args.test_path, transform=add_zeros)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    if args.train_path:
        train_dataset = GraphDataset(args.train_path, transform=add_zeros)
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        
        if val_size == 0 and train_size > 0:
            print("Warning: Not enough samples for validation split, using entire training set for validation as well.")
            val_subset = train_subset = torch.utils.data.Subset(train_dataset, range(len(train_dataset)))
        elif val_size == 0 and train_size == 0:
            raise ValueError("Training dataset is empty.")
        else:
            train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False)
        
        num_epochs = args.epochs
        best_f1 = 0.0
        train_losses = []
        train_accuracies = []

        if num_checkpoints > 1:
            checkpoint_intervals = [int((i + 1) * num_epochs / num_checkpoints) for i in range(num_checkpoints)]
        else:
            checkpoint_intervals = [num_epochs] 

        for epoch in range(num_epochs):
            train_loss = train(
                train_loader, model, optimizer, criterion, device,
                save_checkpoints=(epoch + 1 in checkpoint_intervals),
                checkpoint_path=os.path.join(checkpoints_folder, f"model_{test_dir_name}"),
                current_epoch=epoch,
                grad_clip_value=args.grad_clip_value
            )
            
            if isinstance(criterion, GCODLoss):
                eval_criterion = torch.nn.CrossEntropyLoss()
                _, train_acc, train_f1 = evaluate(train_loader, model, eval_criterion, device, calculate_accuracy=True)
                val_loss, val_acc, val_f1 = evaluate(val_loader, model, eval_criterion, device, calculate_accuracy=True)

            else:
                _, train_acc, train_f1 = evaluate(train_loader, model, criterion, device, calculate_accuracy=True)
                val_loss, val_acc, val_f1 = evaluate(val_loader, model, criterion, device, calculate_accuracy=True)


            '''print(f"Epoch {epoch + 1} - Training Loss: {train_loss:.6f}, "
                  f"Training Accuracy: {train_acc:.6f}, Training F1: {train_f1:.6f}, "
                  f"Validation Loss: {val_loss:.6f}, Validation Accuracy: {val_acc:.6f}, Validation F1: {val_f1:.6f}")'''

            
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            

            logging.info(f"Epoch {epoch + 1} - Training Loss: {train_loss:.4f}, "
                  f"Training Accuracy: {train_acc:.4f}, Training F1: {train_f1:.4f}, "
                  f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}, Validation F1: {val_f1:.4f}")


            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Best model updated and saved at {checkpoint_path}")


            if scheduler:
                if args.scheduler == 'ReduceLROnPlateau':
                    if args.scheduler_monitor_metric == 'val_loss':
                        scheduler.step(val_loss)
                    elif args.scheduler_monitor_metric == 'val_acc':
                        scheduler.step(val_acc)
                    else: 
                        scheduler.step(val_f1)
                else: 
                    scheduler.step()


        plot_training_progress(train_losses, train_accuracies, os.path.join(logs_folder, "plots"))

    if not os.path.exists(checkpoint_path):
        print(f"Warning: Checkpoint {checkpoint_path} not found. Predictions might be from an untrained model if training was not run.")
    else:
        model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
        

    predictions = evaluate(test_loader, model, None, device, calculate_accuracy=False)

    save_predictions(predictions, args.test_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate GNN models on graph datasets.")

    parser.add_argument("--num_checkpoints", type=int, default=5, help="Number of checkpoints to save during training. (default: 3)")
    parser.add_argument('--device', type=int, default=0, help='which gpu to use if any (default: 0)')

    parser.add_argument("--train_path", type=str, help="Path to the training dataset (optional).")
    parser.add_argument("--test_path", type=str, required=True, help="Path to the test dataset.")
    
    parser.add_argument('--loss', type=str, default='ce',help='ce, gce, sce, gcod, nce (default: ce)')
    parser.add_argument('--q', type=float, default=0.7, help='q of gce (default: 0.7)')
    parser.add_argument('--alpha', type=float, default=0.3, help='alpha of sce related to ce (default: 0.3)')
    parser.add_argument('--beta', type=float, default=1.0, help='beta of sce related to rce (default: 1.0)')

    parser.add_argument('--gnn', type=str, default='gin', help='GNN gin, gin-virtual, or gcn, gcn-virtual (default: gin)')
    parser.add_argument('--drop_ratio', type=float, default=0.4, help='dropout ratio (default: 0.3)')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.001)')
    parser.add_argument('--wd', type=float, default=1e-4, help='weight decay (default: 1e-4)')

    parser.add_argument('--num_layer', type=int, default=5, help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=300, help='dimensionality of hidden units in GNNs (default: 300)') 
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)') 
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 100)') 
    

    parser.add_argument('--scheduler', type=str, default='ReduceLROnPlateau', choices=['none', 'ReduceLROnPlateau', 'CosineAnnealingLR'], help='LR scheduler type (default: ReduceLROnPlateau)')
    parser.add_argument('--lr_patience', type=int, default=20, help='Patience for ReduceLROnPlateau (default: 10)')
    parser.add_argument('--lr_factor', type=float, default=0.5, help='Factor by which LR is reduced for ReduceLROnPlateau (default: 0.1)')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='Minimum learning rate for schedulers (default: 1e-6)')
    parser.add_argument('--scheduler_monitor_metric', type=str, default='val_f1', choices=['val_loss', 'val_f1', 'val_acc'], help='Metric to monitor for ReduceLROnPlateau (default: val_f1)')


    parser.add_argument('--grad_clip_value', type=float, default=None, help='Value for gradient clipping norm (e.g., 1.0). If None, no clipping. (default: None)')

    
    args = parser.parse_args()
    main(args)


# cd "C:\Users\simon\OneDrive\Desktop\Progetti Ingegneria\hackaton"

# C
### python main.py --test_path "dataset/C/test.json.gz" --gnn "gin-virtual" --emb_dim 300 --num_layer 5

# D
### python main.py --test_path "dataset/D/test.json.gz" --gnn gin --emb_dim 200 --num_layer 3

# B
### python main.py --test_path "dataset/B/test.json.gz" --gnn gin --emb_dim 200 --num_layer 4

# A
### python main.py --test_path "dataset/A/test.json.gz" --gnn "gin-virtual" --emb_dim 300 --num_layer 5
 


