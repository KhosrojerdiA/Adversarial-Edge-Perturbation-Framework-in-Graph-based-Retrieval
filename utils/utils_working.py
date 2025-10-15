

import sys
import os

#os.system("nvidia-smi")
#os.environ["CUDA_VISIBLE_DEVICES"] = "3"


project_path = './'
sys.path.append(project_path)
# Set a seed for reproducibility
#torch.manual_seed(3708) 


from model.retrieval import * 
from model.per_node_attack import *  

import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.nn import VGAE, APPNP
from torch_geometric.utils import negative_sampling, remove_self_loops
import torch_geometric.transforms as T
from torch_geometric.data import Data
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import roc_auc_score, average_precision_score
import math
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.notebook import tqdm
from torch_geometric.utils import degree, remove_self_loops
import random
from torch_geometric.utils import remove_self_loops
from torch_geometric.data import Data
from torch.nn import Linear
from torch_geometric.nn import GCNConv
from networkx import pagerank
import pandas as pd
import time
torch.backends.cudnn.deterministic = True  # Force deterministic behavior
torch.backends.cudnn.benchmark = False  # Disable auto-tuning for convolution algorithms
from torch_geometric.nn import GINConv
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.loader import DataLoader
#____________________________________________________________________________________________________________________________________________

class GCNEncoder(nn.Module):
    """
    Encodes node features into latent embeddings using a simple 2-layer GCN.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        # First GCN layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # Second GCN layer
        x = self.conv2(x, edge_index)
        return x


class EdgePredictor(nn.Module):
    """
    Takes node embeddings from the Above and predicts the label
    (e.g. change_position) for a given pair of nodes (an edge).
    """
    def __init__(self, node_embedding_dim):
        super(EdgePredictor, self).__init__()
        # A simple linear layer on top of two concatenated node embeddings
        self.mlp = nn.Sequential(
            nn.Linear(node_embedding_dim * 2, node_embedding_dim),
            nn.ReLU(),
            nn.Linear(node_embedding_dim, 1)  # 1-dimensional output (e.g. regression or binary logit)
        )

    def forward(self, node_embeddings, edges):
        """
        node_embeddings: Tensor of shape [num_nodes, node_embedding_dim]
        edges: Tensor of shape [num_edges, 2] specifying which node pairs
        """
        # Extract embeddings for source and target nodes
        src = node_embeddings[edges[:, 0]]  # [num_edges, node_embedding_dim]
        dst = node_embeddings[edges[:, 1]]  # [num_edges, node_embedding_dim]

        # Concatenate source and target embeddings
        edge_repr = torch.cat([src, dst], dim=-1)  # [num_edges, 2*node_embedding_dim]

        # Predict
        return self.mlp(edge_repr).squeeze(-1)  # [num_edges]
    
#____________________________________________________________________________________________________________________________________________

from sklearn.preprocessing import StandardScaler

def train_model_tr(data, training_set, hidden_dim, num_epochs, lr):


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    # Flatten the training_set to get a list of edges and labels
    all_edges = []
    all_labels = []
    for _, edges_dict in training_set.items():
        for (src, dst), label in edges_dict.items():
            all_edges.append((src, dst))
            all_labels.append(label)
        
    scaler = StandardScaler()
    scaled_labels = scaler.fit_transform(np.array(all_labels).reshape(-1, 1)).flatten()

    # Convert edges and labels to Tensors
    edge_tensor = torch.tensor(all_edges, dtype=torch.long, device=device)  # [num_training_edges, 2]
    labels_tensor = torch.tensor(scaled_labels, dtype=torch.float, device=device)  # [num_training_edges]


    # Initialize the GCNEncoder and EdgePredictor
    input_dim = data.x.size(1)  # number of node features
    node_embedding_dim = hidden_dim  # you can use a separate dimension if you like
    gcn_encoder = GCNEncoder(input_dim, hidden_dim, node_embedding_dim).to(device)
    edge_predictor = EdgePredictor(node_embedding_dim).to(device)

    # Define an optimizer (for both the GCN encoder and edge predictor)
    optimizer = torch.optim.Adam(
        list(gcn_encoder.parameters()) + list(edge_predictor.parameters()),
        lr=lr
    )

    # Define a loss function (Binary Cross-Entropy if your label is 0/1, 
    # or MSELoss if it's a regression)
    # Adjust based on your actual label range.
    criterion = nn.MSELoss()

    # Training loop
    for epoch in range(num_epochs):
        gcn_encoder.train()
        edge_predictor.train()
        optimizer.zero_grad()

        # Forward pass: get node embeddings, then predict on edges
        node_embeddings = gcn_encoder(data.x, data.edge_index)  # [num_nodes, node_embedding_dim]
        preds = edge_predictor(node_embeddings, edge_tensor)     # [num_training_edges]

        # Compute loss
        loss = criterion(preds, labels_tensor)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    # You might want to return the trained modules
    return gcn_encoder, edge_predictor


#____________________________________________________________ Loop ____________________________________________________________

def tr_model_prediction(data, gcn_encoder, edge_predictor, test_edges_list):
    """
    data: same data object as before
    gcn_encoder: trained GCNEncoder module
    edge_predictor: trained EdgePredictor module
    test_set: dict of edges -> labels (or just edges). 
              For example:
              {
                (u1, v1): label1,
                (u2, v2): label2,
                ...
              }
    Returns: a dict of {(u, v): predicted_value} for each edge in the test set.
    """

    device = next(gcn_encoder.parameters()).device  # same device the model is on
    data = data.to(device)

    # Convert test edges to Tensors
    
    edge_tensor = torch.tensor(test_edges_list, dtype=torch.long, device=device)

    gcn_encoder.eval()
    edge_predictor.eval()

    with torch.no_grad():
        # Forward pass: get node embeddings, then predict
        node_embeddings = gcn_encoder(data.x, data.edge_index)
        preds = edge_predictor(node_embeddings, edge_tensor)  # [num_test_edges]

    # Convert predictions back to a dictionary
    predictions = {}
    for (edge, pred_value) in zip(test_edges_list, preds.cpu().numpy()):
        predictions[edge] = float(pred_value)
    
    sorted_predictions = dict(sorted(predictions.items(), key=lambda item: item[1], reverse=True))

    return sorted_predictions
#____________________________________________________________________________________________________________________________________________


def mark_edges_above_threshold(data, threshold, promotion_mode):
    updated_data = {}

    for node, edge_dict in data.items():
        updated_edge_dict = {
            edge: 1 if value > threshold else 0
            for edge, value in edge_dict.items()
        }
        updated_data[node] = updated_edge_dict

    return updated_data



#____________________________________________________________________________________________________________________________________________

def prepare_edge_dataset(per_node_data):
    edge_list = []
    labels = []

    for edge_dict in per_node_data.values():
        for edge, label in edge_dict.items():
            edge_list.append(edge)
            labels.append(label)

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_labels = torch.tensor(labels, dtype=torch.float)

    return edge_index, edge_labels

#____________________________________________________________________________________________________________________________________________

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCNLinkPredictor(nn.Module):

    def __init__(self, in_channels, hidden_channels, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.dropout = dropout

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

    def decode(self, x, edge_pairs):
        src, dst = edge_pairs
        # Dot product decoder: <h_src, h_dst>
        return torch.sigmoid((x[src] * x[dst]).sum(dim=1))

    def forward(self, x, edge_index, edge_pairs):
        x = self.encode(x, edge_index)
        return self.decode(x, edge_pairs)

    
#____________________________________________________________________________________________________________________________________________

from sklearn.model_selection import train_test_split

def split_edges(edge_index, labels, val_ratio=0.2):
    idx = torch.arange(len(labels))
    train_idx, val_idx = train_test_split(idx, test_size=val_ratio, stratify=labels.cpu())

    return (
        edge_index[:, train_idx], labels[train_idx],
        edge_index[:, val_idx], labels[val_idx]
    )
#____________________________________________________________________________________________________________________________________________






#____________________________________________________________________________________________________________________________________________

def undersample_labels(edge_index, labels):
    # Indices of class 1 and class 0
    pos_indices = (labels == 1).nonzero(as_tuple=True)[0]
    neg_indices = (labels == 0).nonzero(as_tuple=True)[0]

    # Randomly sample from negatives to match number of positives
    num_pos = len(pos_indices)
    sampled_neg_indices = neg_indices[torch.randperm(len(neg_indices))[:num_pos]]

    # Combine and shuffle
    balanced_indices = torch.cat([pos_indices, sampled_neg_indices])
    shuffled = balanced_indices[torch.randperm(len(balanced_indices))]

    # Subset edge_index and labels
    edge_index_balanced = edge_index[:, shuffled]
    labels_balanced = labels[shuffled]

    return edge_index_balanced, labels_balanced


#____________________________________________________________________________________________________________________________________________


def link_prediction_predict(model, data, edge_list):
    edge_tensor = torch.tensor(edge_list, dtype=torch.long).t().contiguous().to(data.x.device)

    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index, edge_tensor)
        probs = torch.sigmoid(logits).cpu().tolist()

    return {edge: prob for edge, prob in zip(edge_list, probs)}





#____________________________________________________________________________________________________________________________________________


def evaluate_accuracy_of_ones(preds, ground_truth_scores, top_n):
    # Step 1: Convert ground truth scores to binary labels
    sorted_edges = sorted(ground_truth_scores.items(), key=lambda x: x[1], reverse=True)
    ground_truth = {
        edge: 1 if i < top_n else 0
        for i, (edge, _) in enumerate(sorted_edges)
    }

    # Step 2: Accuracy for class 1 only
    true_positives = 0
    total_positives = 0

    for edge, true_label in ground_truth.items():
        if true_label == 1:
            total_positives += 1
            pred_label = preds.get(edge, 0)  # default to 0 if not in prediction
            if pred_label == 1:
                true_positives += 1

    accuracy_ones = true_positives / total_positives if total_positives > 0 else 0.0
    return accuracy_ones, true_positives, total_positives, ground_truth


#____________________________________________________________________________________________________________________________________________



from sklearn.metrics import recall_score, precision_score, f1_score

def find_best_threshold(val_probs, val_true):
    best_f1 = 0
    best_thresh = 0.5
    best_recall = 0
    best_precision = 0

    for thresh in [i / 100 for i in range(10, 90)]:
        preds = (val_probs > thresh).astype(int)
        f1 = f1_score(val_true, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            best_recall = recall_score(val_true, preds)
            best_precision = precision_score(val_true, preds)
    return best_thresh, best_f1, best_precision, best_recall
#____________________________________________________________________________________________________________________________________________
def train_link_prediction_model(
    model, data,
    train_edges, train_labels,
    val_edges, val_labels,
    optimizer, loss_fn,
    scheduler=None,
    num_epochs=200
):
    best_f1 = 0
    best_epoch = 0
    best_stats = {}
    best_model_state = None

    for epoch in range(1, num_epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, train_edges)
        loss = loss_fn(out, train_labels)
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        model.eval()
        with torch.no_grad():
            val_out = model(data.x, data.edge_index, val_edges)
            val_probs = torch.sigmoid(val_out).cpu().numpy()
            val_true = val_labels.cpu().numpy()

            # Find best threshold for current epoch
            best_thresh, f1, precision, recall = find_best_threshold(val_probs, val_true)

            if f1 > best_f1:
                best_f1 = f1
                best_epoch = epoch
                best_stats = {
                    "f1": f1,
                    "precision": precision,
                    "recall": recall,
                    "threshold": best_thresh
                }
                # Save model state dict
                best_model_state = model.state_dict()

        print(f"Epoch {epoch:03}, Loss: {loss.item():.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Threshold: {best_thresh:.2f}")

    print(f"\n✅ Best F1 Score: {best_stats['f1']:.4f} at Epoch {best_epoch}")
    print(f"   ↳ Precision: {best_stats['precision']:.4f}, Recall: {best_stats['recall']:.4f}, Threshold: {best_stats['threshold']:.2f}")

    # Load best model before returning
    model.load_state_dict(best_model_state)
    return model, best_stats['threshold']



#____________________________________________________________________________________________________________________________________________



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class GraphSAGEEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, dropout=0.5):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv3(x, edge_index)
        return x

class EdgeClassifier(nn.Module):
    def __init__(self, in_channels, hidden_channels, dropout=0.5):
        super().__init__()
        self.encoder = GraphSAGEEncoder(in_channels, hidden_channels, dropout)
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_channels, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x, graph_edge_index, edge_pairs):
        node_embeddings = self.encoder(x, graph_edge_index)
        src, dst = edge_pairs
        edge_features = torch.cat([node_embeddings[src], node_embeddings[dst]], dim=1)
        return self.mlp(edge_features).squeeze()



class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        return F_loss.mean() if self.reduction == 'mean' else F_loss
    


#____________________________________________________________________________________________________________________________________________
