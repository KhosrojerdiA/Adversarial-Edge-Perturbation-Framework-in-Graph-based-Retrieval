import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import remove_self_loops
from torch_geometric.transforms import NormalizeFeatures
import random
import numpy as np
import pickle
import os
from model.edge_performance import calculate_edge_performance_all_data
from utils.utils import gold_remove_edges
from torch_geometric.nn import SAGEConv  # Switched from GCNConv to SAGEConv
from utils.utils import *



#____________________________________________________________________________________________________________________________________________

def load_sort_and_create_edge_performance_dataset(data, graph_model, data_name, top_k, n, trained_models_path):

    #top k = data.num_nodes = data.x.size(0)
    # top k defult is data.num_nodes
    #n = 3 # Number of top edges to save for each node

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    file_path = f'{trained_models_path}/change_position_{data_name}_{graph_model}_edge_performance.pkl'  #./CiteSeer_gcn_edge_performance.pkl

    if os.path.exists(file_path):
        print("_____________________________________")
        print(f"Edge Performance found in here {file_path}. Loading edge performance data...")
        print("_____________________________________")
        with open(file_path, "rb") as file:
            if torch.cuda.is_available() == True:
                edge_performance = pickle.load(file)
            else:
                edge_performance = torch.load(file, map_location=torch.device('cpu'))

        print("_____________________________________")
        print("Edge performance loaded successfully and ready to sort and create dataset!")
        print("_____________________________________")
    else:
        print("_____________________________________")
        print(f"Edge Performance not found at {file_path}. Calculating edge performance for all nodes...")
        print("_____________________________________")
        edge_performance = calculate_edge_performance_all_data(data, graph_model, data_name, trained_models_path)
        print("Edge performance is calculated and ready to sort and create dataset!")
        print("_____________________________________")

    edge_performance_original = edge_performance
    target_edges_dataset = {}


    #for node, edges in edge_performance.items():
    #    if promo_mode:
    #        sorted_edges = sorted(edges.items(), key=lambda x: x[1][2][0])  # Ascending order (lowest scores)
    #    else:
    #        sorted_edges = sorted(edges.items(), key=lambda x: x[1][2][0], reverse=True)  # Descending order (highest scores)
    #    
    #    top_n_edges = sorted_edges[:n]
    #    target_edges_dataset[node] = {edge[0]: edge[1] for edge in top_n_edges}  
    target_edges_dataset = edge_performance_original



    return edge_performance_original, target_edges_dataset  #format of citeseer, cora and pubmed

#____________________________________________________________________________________________________________________________________________

def extract_edge_changes(edge_performance):
    """
    Extracts node, edge, and position change from edge_performance.
    
    Args:
        edge_performance (dict): Dictionary with node as keys and edges with their details as values.
    
    Returns:
        dict: A new dictionary with only node, edge, and position change.
    """
    simplified_performance = {}
    
    for node, edges in edge_performance.items():
        simplified_performance[node] = {}
        for edge, details in edges.items():
            position_change = details[2][0]  # Extract position change
            simplified_performance[node][edge] = position_change
    
    return simplified_performance

#____________________________________________________________________________________________________________________________________________

def create_simplified_performance(data, graph_model, data_name, top_k, n, trained_models_path, promo_mode):

    edge_performance_original, edge_performance = load_sort_and_create_edge_performance_dataset(data, graph_model, data_name, top_k, n, trained_models_path, promo_mode)
    simplified_performance = extract_edge_changes(edge_performance)

    return simplified_performance 
 
#____________________________________________________________________________________________________________________________________________
#____________________________________________________________________________________________________________________________________________
#____________________________________________________________________________________________________________________________________________


def create_train_target_node(data, graph_model, data_name, selected_nodes, trained_models_path, top_k, n, promo_mode):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#if not os.path.exists(model_path): # Not targeted_node in the folder     ./targeted_node_CiteSeer_top_3327_n_10_gcn.pth
    print("_____________________________________")

    edge_performance_original, edge_performance = load_sort_and_create_edge_performance_dataset(data, graph_model, data_name ,top_k, n, 
                                                                                                trained_models_path) #format of plantoid                                                                                       trained_models_path, promotion_mode) 
    simplified_performance = extract_edge_changes(edge_performance) #just edge: change_position
    simplified_performance_link_prediction = mark_edges_above_threshold(simplified_performance, 1) #change_position above thresh will be 1.

      
    training_set = {node: edges for node, edges in simplified_performance_link_prediction.items() if node not in selected_nodes.tolist()}
    train_edge_index, train_labels = prepare_edge_dataset(training_set)
    train_edge_index = train_edge_index.to(device)
    train_labels = train_labels.to(device)

    data.x = data.x.to(device)
    data.edge_index = data.edge_index.to(device)
    num_ones = (train_labels == 1).sum().item()
    num_zeros = (train_labels == 0).sum().item()
    train_edges, train_labels, val_edges, val_labels = split_edges(train_edge_index, train_labels, val_ratio=0.2)
    

    model = EdgeClassifier(
    in_channels=data.x.size(1),
    hidden_channels=128,
    dropout=0.5
    ).to(device)

    pos_weight = torch.tensor([num_zeros / num_ones], dtype=torch.float).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    #loss_fn = FocalLoss(alpha=pos_weight.item(), gamma=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Optional LR scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.7)

    trained_model, best_thresh = train_link_prediction_model(
        model, data,
        train_edges, train_labels,
        val_edges, val_labels,
        optimizer, loss_fn,
        scheduler=scheduler,
        num_epochs=200
    )

    print(f"Targeted Node model is trained!")
    print("_____________________________________")
    return trained_model, best_thresh, simplified_performance_link_prediction
    
#____________________________________________________________________________________________________________________________________________
 