## GNN-Protein-Classification

**Teaser Diagram:**
The process begins with an input graph, which is preprocessed (e.g., node features are initialized). This graph is then fed into a GNN backbone, which can consist of layers like GIN or GCN, potentially incorporating a virtual node for global context, edge dropout for regularization, and batch normalization. After several GNN layers, node embeddings are produced. These node embeddings are aggregated using a graph pooling mechanism (such as mean, sum, max, or attention pooling) to create a single vector representation for the entire graph. This graph embedding is then passed through a classifier (typically a linear layer) to predict the final class label. The model is trained using one of several available loss functions, including Cross-Entropy (CE), Generalized Cross-Entropy (GCE), Symmetric Cross-Entropy (SCE), or Graph Centroid Outlier Discounting (GCOD).
This repository contains a PyTorch-based solution for graph classification tasks using Graph Neural Networks (GNNs). The framework is designed to be flexible, supporting various GNN architectures, loss functions, and training configurations.
![Editor _ Mermaid Chart-2025-05-31-150612](https://github.com/user-attachments/assets/cd54ec0d-a56e-4caa-b3d6-36c4079741d6)


## Overview of the Method

This solution tackles graph classification by learning discriminative representations of graph structures using Graph Neural Networks (we mostly used gin with virtual node). The core methodology involves several key stages:

1.  **Data Loading and Preprocessing:**
    *   Graphs are loaded from `json.gz` files, each containing edge indices, edge attributes (if any), number of nodes, and a target label.
    *   Node features (`data.x`) are initialized uniformly (e.g., to zeros), as the GNNs are designed to learn representations primarily from graph topology and edge attributes.

2.  **Graph Neural Network Architecture:**
    *   The framework supports configurable GNN architectures, primarily **Graph Isomorphism Network (GIN)** and **Graph Convolutional Network (GCN)** layers.
    *   **Virtual Nodes:** An option to include a virtual node is provided. This special node connects to all other nodes in the graph, acting as a global information aggregator and disseminator, enhancing the expressive power of the GNN for graph-level tasks.
    *   **Node Embeddings:** Multiple GNN layers are stacked to iteratively update node representations by aggregating information from their local neighborhoods.
    *   **Regularization:** Techniques like dropout on node embeddings and edge dropout (randomly removing edges during training) are employed to prevent overfitting. Batch Normalization is used after GNN layers to stabilize training.

3.  **Graph-Level Representation:**
    *   After obtaining node embeddings, a **graph pooling** layer (e.g., global mean, sum, max, attention, or Set2Set pooling) aggregates these node features into a single vector representing the entire graph.

4.  **Classification:**
    *   The resulting graph embedding is fed into a final linear layer followed by a softmax (implicitly, via the loss function) to produce class probabilities for the graph.

5.  **Training Paradigm:**
    *   The model is trained end-to-end using an Adam optimizer.
    *   **Loss Functions:** A variety of loss functions are supported to handle potentially noisy labels or imbalanced datasets:
        *   Standard Cross-Entropy (CE)
        *   Generalized Cross-Entropy (GCE)
        *   Symmetric Cross-Entropy (SCE)
        *   Graph Centroid Outlier Discounting (GCOD), a more advanced loss designed for learning with noisy labels by dynamically weighting samples.
    *   **Learning Rate Scheduling:** Options like `ReduceLROnPlateau` and `CosineAnnealingLR` are available to adapt the learning rate during training.
    *   **Gradient Clipping:** Can be used to prevent exploding gradients.
    *   **Validation and Model Selection:** The training data is split into training and validation subsets. The model's performance (e.g., F1-score) on the validation set is monitored, and the model with the best validation F1-score is saved.
    *   **Checkpointing:** Model checkpoints are saved periodically during training and the best performing model is stored.

6.  **Evaluation and Prediction:**
    *   The trained model is evaluated on a test set using metrics like accuracy and F1-score.
    *   Predictions for the test set are saved in a CSV format suitable for submission.
  
    
The system is highly configurable through command-line arguments, allowing for easy experimentation with different GNN types, number of layers, embedding dimensions, dropout rates, loss functions, and other hyperparameters. Training progress, including loss and accuracy, is logged and can be visualized.

# Parameters used for each dataset
## For dataset A
gin-virtual / noisy cross entropy loss / 0.001 lr / neuron drop ratio 0.5 / 300 embeded dimension / 5 layers / 200 epochs / 32 batch size / scheduler = ReduceLRonPlateau

## For dataset B
gin / symmetric cross entropy loss / 0.003 / alpha 0.4 / beta 0.8 / neuron drop ratio 0.6 / 200 embeded dimension / 4 layers / 150 epochs / 32 batch size / scheduler = ReduceLRonPlateau

## For dataset C 
gin-virtual / noisy cross entropy loss / 0.0025 lr / neuron drop ratio 0.5 / 300 embeded dimension / 5 layers / 200 epochs / 32 batch size / scheduler = ReduceLRonPlateau

## For dataset D
gin / symmetric cross entropy loss / 0.005 lr / alpha 0.3 / beta 1 / neuron drop ration 0.6 / embeded dimension 200 / 3 layers / 200 epochs / 64 batch size / scheduler = ReduceLRonPlateau 

# Test the model
## For dataset A
```bash
python main.py --test_path <path_to_test.json.gz> --gnn "gin-virtual" --emb_dim 300 --num_layer 5
```
## For dataset B
```bash
python main.py --test_path <path_to_test.json.gz> --gnn "gin" --emb_dim 200 --num_layer 4
```
## For dataset C
```bash
python main.py --test_path <path_to_test.json.gz> --gnn "gin-virtual" --emb_dim 300 --num_layer 5
```
## For dataset D
```bash
python main.py --test_path <path_to_test.json.gz> --gnn "gin" --emb_dim 200 --num_layer 3
```
