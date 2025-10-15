import sys
import os


project_path = '/mnt/data/khosro/Graph-Pruning'
sys.path.append(project_path)

from EPAGCL.model_2 import *  

import torch
import math
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import torch
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from torch.nn import Linear
from torch_geometric.nn import GCNConv
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.nn import VGAE, APPNP
from torch_geometric.utils import negative_sampling, remove_self_loops
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.utils import remove_self_loops, to_undirected
import torch_geometric.transforms as T
import math
from torch_geometric.data import Data
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from tqdm.notebook import tqdm
import torch
from torch_geometric.utils import degree, remove_self_loops
import copy
from torch_geometric.utils import train_test_split_edges
# Set a seed for reproducibility
#torch.manual_seed(3708)
import pickle
import os.path as osp
from torch_geometric.utils import negative_sampling
import random
import torch
from torch_geometric.transforms import RandomLinkSplit, NormalizeFeatures
import torch
from torch_geometric.utils import remove_self_loops
from torch_geometric.data import Data
from torch.nn import Linear
from torch_geometric.nn import GCNConv
from networkx import pagerank
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from tqdm import tqdm
from tqdm.notebook import tqdm
from torch_geometric.utils import subgraph

#____________________________________________________________________________________________________________________________

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
#_______________________________________________________________________________________________________________________________________________

from torch_geometric.utils import subgraph, degree, to_undirected

def remove_isolated_nodes(data):
    """
    Removes nodes with zero degree and rebuilds the edge_index so that:
    - Node indices are contiguous [0, num_nodes_new)
    - No degree-0 nodes remain
    - Ensures edge_index and all weights align perfectly
    """
    print(">> Checking for isolated or invalid nodes...", flush=True)

    # --- Step 1: Ensure edge_index is undirected
    edge_index = to_undirected(data.edge_index)

    # --- Step 2: Identify connected nodes (degree > 0)
    deg = degree(edge_index[0], num_nodes=data.num_nodes)
    connected_nodes = (deg > 0).nonzero(as_tuple=True)[0]
    num_isolated = data.num_nodes - connected_nodes.numel()

    if num_isolated > 0:
        print(f"⚠️  Found {num_isolated} isolated nodes. Removing them...", flush=True)

        # --- Step 3: Build new subgraph containing only connected nodes
        new_edge_index, mapping = subgraph(connected_nodes, edge_index, relabel_nodes=True)

        # --- Step 4: Reindex features and labels
        data.x = data.x[connected_nodes]
        if hasattr(data, "y"):
            data.y = data.y[connected_nodes]
        data.edge_index = new_edge_index
        data.num_nodes = data.x.size(0)

        # --- Step 5: Remove duplicate edges & self-loops
        data.edge_index = to_undirected(data.edge_index)
        data.edge_index = data.edge_index[:, data.edge_index[0] != data.edge_index[1]]

        # --- Step 6: Validate alignment
        num_edges = data.edge_index.size(1)
        print(f"✅ Removed isolated nodes. New num_nodes={data.num_nodes}, edges={num_edges}", flush=True)

    else:
        print("✅ No isolated nodes found. Data is consistent.", flush=True)

    # --- Final check for downstream consistency
    assert data.edge_index.max() < data.num_nodes, "❌ edge_index has invalid node ids!"
    assert data.edge_index.size(0) == 2, "❌ edge_index must be shape [2, num_edges]!"
    assert data.edge_index.size(1) > 0, "❌ Graph has no edges after cleanup!"
    return data

#____________________________________________________________________________________________________________________________

def attacked_embedding_v2(updated_data, one_node_selected_nodes, trained_model, device="cuda"):
    """
    Generate attacked embeddings without retraining the model.

    Args:
        updated_data (torch_geometric.data.Data): Graph after edge removals.
        one_node_selected_nodes (torch.Tensor): Node(s) to isolate and compute query embeddings for.
        trained_model (torch.nn.Module): Pre-trained model (already trained before attack).
        device (str): 'cuda' or 'cpu'.
    
    Returns:
        attacked_dataset_embeddings (torch.Tensor): Embeddings for all nodes in the attacked graph.
        one_node_selected_node_embeddings (torch.Tensor): Embeddings for selected nodes.
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    updated_data = updated_data.to(device)
    trained_model = trained_model.to(device)
    trained_model.eval()

    # --- Full graph embeddings (after attack)
    with torch.no_grad():
        attacked_dataset_embeddings, _ = trained_model(updated_data.x, updated_data.edge_index)

    print("- Attacked Embedding is Generated!", flush=True)
    print("____________________________________________", flush=True)

    # --- Isolate selected node(s) by removing edges connected to them
    selected_nodes_set = set(one_node_selected_nodes.tolist())
    mask = [(edge[0].item() not in selected_nodes_set) and (edge[1].item() not in selected_nodes_set)
            for edge in updated_data.edge_index.t()]
    mask = torch.tensor(mask, dtype=torch.bool, device=device)
    isolated_edge_index = updated_data.edge_index[:, mask]

    # --- Embeddings for isolated nodes (query nodes)
    with torch.no_grad():
        isolated_embeddings, _ = trained_model(updated_data.x, isolated_edge_index)
    one_node_selected_node_embeddings = isolated_embeddings[one_node_selected_nodes]

    print("- Attacked Query Embedding is Generated!", flush=True)
    print("____________________________________________", flush=True)

    return attacked_dataset_embeddings, one_node_selected_node_embeddings



#________________________________________________________________________________________________________________________________________________

def compute_position_change(node, top_k_indices, new_top_k_indices):
    # Convert tensors to CPU for easier indexing
    top_k_indices_cpu = top_k_indices
    new_top_k_indices_cpu = new_top_k_indices
    
    # Find the position of the node in both lists
    old_position_tensor = (top_k_indices_cpu == node).nonzero(as_tuple=True)[1]
    new_position_tensor = (new_top_k_indices_cpu == node).nonzero(as_tuple=True)[1]

    # If node is missing in either list, return None
    if old_position_tensor.numel() == 0 or new_position_tensor.numel() == 0:
        return 1

    old_position = old_position_tensor.item()
    new_position = new_position_tensor.item()

    # Calculate the change in position
    change_position = new_position - old_position

    return change_position

#________________________________________________________________________________________________________________________________________________

def similarity(top_k, new_node_embeddings, dataset_embeddings):

    new_node_embeddings = new_node_embeddings.to(device)
    dataset_embeddings = dataset_embeddings.to(device)

    cosine_sim = torch.nn.functional.cosine_similarity(
        new_node_embeddings.unsqueeze(1),  # Add dimension for pairwise comparison
        dataset_embeddings.unsqueeze(0),   # Add dimension for pairwise comparison
        dim=-1                             # Specify the dimension for reduction
    )

    # Get the indices of the top-k most similar vectors
    top_k_indices = torch.argsort(cosine_sim, dim=1, descending=True)[:, :top_k]

    return top_k_indices

#____________________________________________________________________________________________________________________________________

def compare_positions(top_k_indice_at_k_origin, attacked_at_num_node_dict_tensor_only): 

    total_promoted = 0
    total_demoted = 0
    total_unchanged = 0
    total_items_in_both_lists = 0
    total_items_in_promoted_list = 0
    total_items_in_demoted_list = 0

    for i in range(len(top_k_indice_at_k_origin)):
        list1 = top_k_indice_at_k_origin[i]
        list2 = attacked_at_num_node_dict_tensor_only[i]

        for idx, item in enumerate(list1):
            # Check if item is in list2 and find its index using PyTorch
            matches = (list2 == item).nonzero(as_tuple=True)[0]
            if len(matches) > 0:
                total_items_in_both_lists += 1
                index_in_list2 = matches.item()

                if index_in_list2 < idx:
                    total_items_in_promoted_list += 1
                    total_promoted += (idx - index_in_list2)
                elif index_in_list2 > idx:
                    total_items_in_demoted_list += 1
                    total_demoted += (index_in_list2 - idx)
                else:
                    total_unchanged += 1

    if total_items_in_both_lists > 0:
        avg_promoted = total_promoted / total_items_in_promoted_list if total_items_in_promoted_list != 0 else 0
        avg_demoted = total_demoted / total_items_in_demoted_list if total_items_in_demoted_list != 0 else 0
        avg_changed = (total_promoted + total_demoted) / (total_items_in_promoted_list + total_items_in_demoted_list) if (total_items_in_promoted_list + total_items_in_demoted_list) != 0 else 0
        avg_unchanged = total_unchanged / total_items_in_both_lists
    else:
        avg_promoted = avg_demoted = avg_changed = avg_unchanged = 0

    return avg_promoted, avg_demoted, avg_changed, avg_unchanged, total_items_in_both_lists


#________________________________________________________________________________________________________________________________________________

def calculate_edge_performance_v2(data, graph_model, data_name, trained_models_path, main_seed): 
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data = data.to(device)  # Move data to the appropriate device
    num_nodes = data.x.size(0)

#________________________

    save_dir = "/mnt/data/khosro/Graph-Pruning/embeddings"
    epochs=2000
    file_prefix = f"{data_name}_{graph_model}_epochs_{epochs}"
    emb_path = os.path.join(save_dir, f"{file_prefix}_embeddings.pt")
    model_path = os.path.join(save_dir, f"{file_prefix}_model.pt")
    print(f">> Found cached EPAGCL model and embeddings for {data_name}! Loading from disk...", flush=True)
    dataset_embeddings = torch.load(emb_path, map_location="cpu")
    model = torch.load(model_path, map_location=device)
    model.eval()
    print("- Loaded existing EPAGCL embeddings and model successfully!", flush=True)
    print("____________________________________________")

    edge_performance = {}


    #node = 12
 
    for node in tqdm(range(num_nodes)):

        #Start of Loop
        edge_performance[node] = {}

        edges = data.edge_index.t().tolist()
        selected_edges = [edge for edge in edges if edge[0] == node or edge[1] == node]

        #Isolate the node by removing all its edges (Query)
        mask = (data.edge_index[0] != node) & (data.edge_index[1] != node)
        isolated_edge_index = data.edge_index[:, mask]

        # Get query embedding
        #model.eval()
        with torch.no_grad():
            isolated_embeddings, _ = model(data.x, isolated_edge_index)
        target_node_embedding = isolated_embeddings[node].unsqueeze(0)

        top_k_indices = similarity(num_nodes, target_node_embedding, dataset_embeddings) #20 or num_nodes

        print("______________________________________________________________", flush=True)
        print("Node: ", node)
        print("______________________________________________________________", flush=True)
        #print("Top K Indices: ", top_k_indices)
        #print("______________________________________________________________")

        for se in selected_edges: #se is edge to be removed
            new_data = data.clone()
            #Remove se edge from the graph
            new_edges = [edge for edge in edges if edge != se]
            new_edges = torch.tensor(new_edges, dtype=torch.long, device=device).t().contiguous()
            new_data.edge_index = new_edges

            # Get the Attacked Data embeddings
            attacked_dataset_embeddings, attacked_one_node_selected_node_embeddings = attacked_embedding_v2(new_data, torch.tensor([node]), model, device=device) #attacked_embedding(new_data, torch.tensor([node]), graph_model)


            new_top_k_indices = similarity(num_nodes, attacked_one_node_selected_node_embeddings, attacked_dataset_embeddings) #20 or num_nodes

            #print("______________________________________________________________")
            #print("Edge removed: ", se)
            #print("______________________________________________________________")
            #print("New Top K Indices: ", new_top_k_indices)
            #print("______________________________________________________________")

            if node in top_k_indices:
                position_before = torch.nonzero(top_k_indices[0] == node, as_tuple=True)[0].item()
                if node in new_top_k_indices:
                    position_after = torch.nonzero(new_top_k_indices[0] == node, as_tuple=True)[0].item()
                    target_demote = position_after - position_before
                else:
                    target_demote = len(new_top_k_indices[0]) + 1 - position_before
            else:
                target_demote = 0

            avg_promoted, avg_demoted, avg_changed, avg_unchanged, total_items = compare_positions(top_k_indices, new_top_k_indices)
            position_change = compute_position_change(node, top_k_indices, new_top_k_indices)
            edge_performance[node].update({tuple(se): [top_k_indices, new_top_k_indices, [position_change, avg_promoted, avg_demoted, avg_changed, avg_unchanged, total_items]]})
            

            #End of Loop

    #print("______________________________________________________________")
    #print("edge_performance: ", edge_performance)
    #print("______________________________________________________________")

    file_path = os.path.join(trained_models_path, f'change_position_{data_name}_{graph_model}_edge_performance.pkl') #./CiteSeer_gcn_edge_performance.pkl
    #file_path = os.path.join(trained_models_path, f'{data_name}_{graph_model}_edge_performance.pkl') #./CiteSeer_gcn_edge_performance.pkl

    with open(file_path, "wb") as file:
        pickle.dump(edge_performance, file)
    
    print(f"Edge performance Dataset saved at: {file_path}", flush=True)
    
#________________________________________________________________________________________________________________________________________________
#________________________________________________________________________________________________________________________________________________

trained_models_path = f"/mnt/data/khosro/Graph-Pruning/edge_performance_dataset_v2"
dataset_subgraph_path = "/mnt/data/khosro/Graph-Pruning/data/pubmed_subgraph.pt"

data_name = 'PubMed'
#['Cora', 'CiteSeer', 'PubMed']

graph_model = 'epagcl_sage'
#['gcn', 'sage', 'epagcl_gcn', 'epagcl_sage' ]

main_seed = 3708

#____________________________________________________________________________________________________________________________________

if data_name == 'CiteSeer': 
    dataset = Planetoid(root='data/Planetoid', name='CiteSeer')
    data = dataset[0]
elif data_name == 'Cora': 
    dataset = Planetoid(root='data/Planetoid', name='Cora')
    data = dataset[0]
elif data_name == 'PubMed':
    dataset = torch.load(dataset_subgraph_path, weights_only=False) #torch.load(dataset_subgraph_path)
    data = remove_isolated_nodes(dataset)


calculate_edge_performance_v2(data, graph_model, data_name, trained_models_path, main_seed) 

#edge_performance[node].update({tuple(se): [top_k_indices, new_top_k_indices, [position_change, avg_promoted, avg_demoted, avg_changed, avg_unchanged, total_items]]})

#{0: 

# {(628, 0): 
# [tensor([[   0,  628, 2326,  ...,  925,  995, 1690]], device='cuda:0'), tensor([[   0,  628, 2326,  ...,  925,  995, 1690]], device='cuda:0'), 
# [0, 69.37957051654092, 75.65886075949368, 72.38328792007266, 0.007213706041478809, 3327]], 

#(0, 628): 
# [tensor([[   0,  628, 2326,  ...,  925,  995, 1690]], device='cuda:0'), tensor([[   0, 2326,  496,  ...,  995,  358, 1690]], device='cuda:0'), 
# [0, 84.51393188854489, 80.24103468547914, 82.3220747889023, 0.0033062819356777877, 3327]]}, 
# 
# 1: ...
            

#____________________________________________________________________________________________________________________________________
