
**Explanation of the Teaser Diagram:**
The process begins with an input graph, which is preprocessed (e.g., node features are initialized). This graph is then fed into a GNN backbone, which can consist of layers like GIN or GCN, potentially incorporating a virtual node for global context, edge dropout for regularization, and batch normalization. After several GNN layers, node embeddings are produced. These node embeddings are aggregated using a graph pooling mechanism (such as mean, sum, max, or attention pooling) to create a single vector representation for the entire graph. This graph embedding is then passed through a classifier (typically a linear layer) to predict the final class label. The model is trained using one of several available loss functions, including Cross-Entropy (CE), Generalized Cross-Entropy (GCE), Symmetric Cross-Entropy (SCE), or Graph Centroid Outlier Discounting (GCOD).
This repository contains a PyTorch-based solution for graph classification tasks using Graph Neural Networks (GNNs). The framework is designed to be flexible, supporting various GNN architectures, loss functions, and training configurations.

graph TD
    A[Input: batched_data (x, edge_index, batch)] --> B{GNN Node Embedding};

    subgraph GNN Node Embedding Module (self.gnn_node)
        direction LR
        B_Input[Input: batched_data] --> B_InitEmb(Initial Node Features);

        B_VN_Choice{virtual_node?}
        B_InitEmb --> B_VN_Choice;

        B_VN_Choice -- Yes --> B_VN_Init[Initialize Virtual Node Embedding];
        B_VN_Init --> B_Loop_Start;
        B_VN_Choice -- No --> B_Loop_Start;

        B_Loop_Start(Loop for num_layer) --> B_Layer_i;

        subgraph Layer_i
            direction TB
            B_L_Input[Node Embeddings H_prev<br>+ (Optional) Virtual Node Emb_v] --> B_L_Conv[GNN Conv (gnn_type)];
            B_L_Conv --> B_L_BN[BatchNorm1d];
            B_L_BN --> B_L_ReLU[ReLU];
            B_L_ReLU --> B_L_Drop[Dropout (drop_ratio)];
            B_L_Drop --> B_L_Res_Choice{residual?};
            B_L_Res_Choice -- Yes --> B_L_Res_Add["Add H_prev (Residual)"];
            B_L_Res_Add --> B_L_Output[Node Embeddings H_i];
            B_L_Res_Choice -- No --> B_L_Output;

            B_L_Output --> B_L_VN_Update_Choice{virtual_node?};
            B_L_VN_Update_Choice -- Yes --> B_L_VN_Aggregate["Aggregate H_i"];
            B_L_VN_Aggregate --> B_L_VN_MLP["MLP for VN update"];
            B_L_VN_MLP --> B_L_VN_Updated_Emb[Updated Virtual Node Emb_v];
            B_L_VN_Update_Choice -- No --> B_L_End_VN_Update;
            B_L_VN_Updated_Emb --> B_L_End_VN_Update;
        end
        B_Layer_i --> B_Loop_End{End Loop?};
        B_Loop_End -- No --> B_Layer_i;
        B_Loop_End -- Yes --> B_JK[Jumping Knowledge (JK)];
        B_JK --> B_h_node[Output Node Embeddings: h_node];
    end

    B --> C{Graph Pooling (graph_pooling)};

    subgraph Graph Pooling Module (self.pool)
        direction LR
        C_Input[h_node, batch_data.batch] --> C_Pool_Choice{Pooling Type};
        C_Pool_Choice -- sum --> C_Sum[global_add_pool];
        C_Pool_Choice -- mean --> C_Mean[global_mean_pool];
        C_Pool_Choice -- max --> C_Max[global_max_pool];
        C_Pool_Choice -- attention --> C_Att[GlobalAttention];
        C_Pool_Choice -- set2set --> C_S2S[Set2Set];
        C_Sum --> C_h_graph[Graph Embedding: h_graph];
        C_Mean --> C_h_graph;
        C_Max --> C_h_graph;
        C_Att --> C_h_graph;
        C_S2S --> C_h_graph_S2S[Graph Embedding: h_graph (2*emb_dim)];
    end

    C --> D[Graph Prediction Linear Layer (self.graph_pred_linear)];
    D_Note["Input: emb_dim (or 2*emb_dim if Set2Set)"];
    D_Note --> D;
    D --> E[Output: Logits (num_class)];

    style B_VN_Init fill:#f9d,stroke:#333,stroke-width:2px
    style B_L_VN_Updated_Emb fill:#f9d,stroke:#333,stroke-width:2px
    style B_VN_Choice fill:#lightgrey,stroke:#333,stroke-width:2px
    style B_L_Res_Choice fill:#lightgrey,stroke:#333,stroke-width:2px
    style B_L_VN_Update_Choice fill:#lightgrey,stroke:#333,stroke-width:2px
    style C_Pool_Choice fill:#lightgrey,stroke:#333,stroke-width:2px
## Overview of the Method

This solution tackles graph classification by learning discriminative representations of graph structures using Graph Neural Networks (GNNs). The core methodology involves several key stages:

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


