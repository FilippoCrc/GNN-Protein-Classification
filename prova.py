import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d, Embedding
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool
from torch_geometric.data import Dataset, Data, DataLoader
from torch_geometric.utils import degree

import json
import gzip
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

# --- 1. Custom Dataset Class ---
class PPADataset(Dataset):
    def __init__(self, root, filename, is_train=True, transform=None, pre_transform=None):
        self.filename = filename
        self.is_train = is_train
        self.raw_filepath = os.path.join(root, self.filename)
        super().__init__(root, transform, pre_transform)
        self.data_list = self._load_data()
        
        if self.data_list and self.data_list[0].x is not None:
            self._num_node_features = self.data_list[0].x.size(1)
        else: # Fallback if no features or empty dataset
            self._num_node_features = 7 

    @property
    def raw_file_names(self):
        return [self.filename]

    @property
    def processed_file_names(self):
        # We are processing on the fly, so no processed files saved by PyG's convention
        return []

    def download(self):
        # Data is assumed to be pre-downloaded
        pass

    def _load_data(self):
        data_list = []
        try:
            with gzip.open(self.raw_filepath, 'rt', encoding='utf-8') as f:
                raw_graphs = json.load(f)
        except FileNotFoundError:
            print(f"Warning: File not found {self.raw_filepath}")
            return []
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from {self.raw_filepath}")
            return []

        for i, graph_data in enumerate(raw_graphs):
            num_nodes = graph_data.get('num_nodes')
            
            # --- MODIFICATION FOR EDGES ---
            # Assuming 'edge_index' in JSON is a list of pairs like [[u,v], [u,v], ...]
            # If 'edge_index' is already [2, num_edges], the .t().contiguous() might be wrong.
            edge_data_raw = graph_data.get('edge_index', []) # Get 'edge_index'

            if num_nodes is None or num_nodes == 0 :
                continue

            if not edge_data_raw:
                 edge_index = torch.empty((2, 0), dtype=torch.long)
            else:
                # If edge_data_raw is ALREADY [2, num_edges] format:
                # edge_index = torch.tensor(edge_data_raw, dtype=torch.long).contiguous()
                # If edge_data_raw is list of pairs [[u,v], ...]:
                edge_index = torch.tensor(edge_data_raw, dtype=torch.long).t().contiguous()


            # --- MODIFICATION FOR NODE FEATURES (checking for 'node_feat', 'x', or fallback) ---
            node_features_raw = graph_data.get('node_feat', graph_data.get('x')) # Check for 'node_feat' or 'x'
            
            if node_features_raw is not None and len(node_features_raw) > 0:
                x = torch.tensor(node_features_raw, dtype=torch.float)
                if x.shape[0] != num_nodes:
                    continue
            else:
                print(f"Warning: No 'node_feat' or 'x' for graph {i} in {self.filename}. Using node degrees as features.")
                if edge_index.numel() > 0:
                    deg = degree(edge_index[0], num_nodes=num_nodes, dtype=torch.float).unsqueeze(1)
                else:
                    deg = torch.zeros((num_nodes, 1), dtype=torch.float)
                x = deg

            if self.is_train:
                # --- MODIFICATION FOR LABELS ---
                label_val = graph_data.get('y', graph_data.get('label')) # Get 'y' or fallback to 'label'
                if label_val is None:
                    continue
                y = torch.tensor([label_val], dtype=torch.long)
                data = Data(x=x, edge_index=edge_index, y=y, num_nodes=num_nodes)
            else:
                data = Data(x=x, edge_index=edge_index, num_nodes=num_nodes)
            
            # Optionally load edge_attr if your model uses it
            # edge_attr_raw = graph_data.get('edge_attr')
            # if edge_attr_raw is not None:
            #    data.edge_attr = torch.tensor(edge_attr_raw, dtype=torch.float)

            data_list.append(data)
        return data_list

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]

    @property
    def num_node_features(self):
        return self._num_node_features


# --- 2. GIN Model with Virtual Node ---
class GINWithVirtualNode(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.hidden_channels = hidden_channels

        # Virtual node embedding
        self.virtual_node_embedding = Embedding(1, hidden_channels) # A single virtual node
        
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.vn_mlps = torch.nn.ModuleList() # MLPs to update virtual node

        # Initial layer
        self.initial_mlp = Sequential(Linear(in_channels, hidden_channels), ReLU(), BatchNorm1d(hidden_channels))
        
        for _ in range(num_layers):
            nn = Sequential(Linear(hidden_channels, hidden_channels), ReLU(), Linear(hidden_channels, hidden_channels))
            self.convs.append(GINConv(nn, train_eps=True)) # train_eps=True is common for GIN
            self.batch_norms.append(BatchNorm1d(hidden_channels))
            self.vn_mlps.append(Sequential(Linear(hidden_channels, hidden_channels), ReLU(), Linear(hidden_channels, hidden_channels)))

        self.fc1 = Linear(hidden_channels, hidden_channels)
        self.fc2 = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        # Initial node projection
        x = self.initial_mlp(x)

        # Initialize virtual node embedding for the batch
        # virtual_node_feat = self.virtual_node_embedding(torch.zeros(batch.max().item() + 1, dtype=torch.long, device=x.device))
        # A simpler way for a single global VN type, then expanded per graph in batch
        vn_emb = self.virtual_node_embedding(torch.tensor([0], device=x.device)) # Shape [1, hidden_channels]
        
        for i in range(len(self.convs)):
            # 1. Aggregate from real nodes to virtual node
            # Pool real node features per graph, then add current virtual node state
            pooled_x_for_vn = global_add_pool(x, batch) + vn_emb.repeat(batch.max().item() + 1, 1)
            
            # 2. Update virtual node embedding
            vn_emb_updated = self.vn_mlps[i](pooled_x_for_vn)
            
            # 3. Add updated virtual node embedding to all real nodes
            # Expand vn_emb_updated to match node batching
            x = x + vn_emb_updated[batch]
            
            # 4. GINConv layer
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            vn_emb = vn_emb_updated # Carry over for next layer's VN input or for final pooling

        # Use the final virtual node embedding for graph representation
        # or combine with pooled real node features
        graph_emb = global_add_pool(x, batch) + vn_emb # Example: combine both

        x_out = F.relu(self.fc1(graph_emb))
        x_out = F.dropout(x_out, p=self.dropout, training=self.training)
        x_out = self.fc2(x_out)
        return F.log_softmax(x_out, dim=-1)

# --- 3. Training Function ---
def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for data in tqdm(loader, desc="Training", leave=False):
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)

# --- 4. Prediction Function ---
def predict_epoch(model, loader, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for data in tqdm(loader, desc="Predicting", leave=False):
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=-1)
            predictions.extend(pred.cpu().tolist())
    return predictions

# --- 5. Main Loop ---
if __name__ == '__main__':
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")

    DATASET_BASE_PATH = 'datasets'  # Your folder containing A, B, C, D
    SUBFOLDERS = ['A', 'B', 'C', 'D']
    # SUBFOLDERS = ['A'] # For quick testing

    # Hyperparameters
    NUM_CLASSES = 6  # As per problem description (6 out of 37)
    HIDDEN_DIM = 128 # Can be tuned
    NUM_LAYERS = 4   # Can be tuned
    LEARNING_RATE = 0.0005 # Can be tuned
    EPOCHS = 30      # Start with this, can be tuned
    BATCH_SIZE = 64  # Can be tuned
    DROPOUT_RATE = 0.3 # Can be tuned

    for subfolder_name in SUBFOLDERS:
        print(f"\n===== Processing dataset: {subfolder_name} =====")
        subfolder_path = os.path.join(DATASET_BASE_PATH, subfolder_name)

        if not os.path.exists(subfolder_path):
            print(f"Subfolder {subfolder_path} not found. Skipping.")
            continue

        train_file = 'train.json.gz'
        test_file = 'test.json.gz'

        # Load data
        print("Loading training data...")
        train_dataset = PPADataset(root=subfolder_path, filename=train_file, is_train=True)
        if not train_dataset.data_list:
            print(f"No data loaded for training in {subfolder_name}. Skipping.")
            continue
        
        node_feature_dim = train_dataset.num_node_features
        print(f"Inferred node feature dimension: {node_feature_dim}")

        print("Loading test data...")
        test_dataset = PPADataset(root=subfolder_path, filename=test_file, is_train=False)
        if not test_dataset.data_list and os.path.exists(os.path.join(subfolder_path, test_file)):
             print(f"Warning: Test data file exists but no test data loaded for {subfolder_name}. Predictions will be empty if test file is truly empty.")
        elif not test_dataset.data_list and not os.path.exists(os.path.join(subfolder_path, test_file)):
            print(f"Test file {test_file} not found in {subfolder_name}. Skipping prediction for this subfolder.")
            # continue # Or handle as needed - maybe you still want to train if test is missing

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True if len(train_dataset)>BATCH_SIZE else False)
        if test_dataset.data_list:
            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        else:
            test_loader = None # No test data to predict on

        # Model, optimizer
        model = GINWithVirtualNode(
            in_channels=node_feature_dim,
            hidden_channels=HIDDEN_DIM,
            out_channels=NUM_CLASSES,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT_RATE
        ).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5) # Added small weight decay

        print(f"Training model for {subfolder_name}...")
        for epoch in range(1, EPOCHS + 1):
            loss = train_epoch(model, train_loader, optimizer, DEVICE)
            print(f'Subfolder: {subfolder_name}, Epoch: {epoch:02d}, Loss: {loss:.4f}')
            # Add validation here if you create a validation split

        # Prediction
        if test_loader:
            print(f"Generating predictions for {subfolder_name}...")
            predictions = predict_epoch(model, test_loader, DEVICE)

            # Save predictions
            output_dir = f"predictions"
            os.makedirs(output_dir, exist_ok=True)
            subfolder_output_dir = os.path.join(output_dir, subfolder_name)
            os.makedirs(subfolder_output_dir, exist_ok=True)
            
            pred_df = pd.DataFrame({'prediction': predictions})
            # It's common for hackathons to require an ID column. If your test set graphs have implicit IDs (0 to N-1):
            pred_df.index.name = 'graph_id' 
            pred_df.to_csv(os.path.join(subfolder_output_dir, 'predictions.csv'))
            print(f"Predictions saved to {os.path.join(subfolder_output_dir, 'predictions.csv')}")
        else:
            print(f"No test data or loader for {subfolder_name}, skipping prediction.")
            
    print("\nAll processing finished.")