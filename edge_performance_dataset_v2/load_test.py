
#with right split - classification

import sys
import os

#os.system("nvidia-smi")
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"


project_path = '/mnt/data/khosro/Graph-Pruning'
sys.path.append(project_path)

# Set a seed for reproducibility
#torch.manual_seed(3708) 

from utils.utils import *
from utils.utils_working import *

from model.retrieval import * 
from model.train_target_node import *  

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
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
import random
from torch_geometric.utils import dropout_adj
#________________________________________________________________________________________________________________________



def load_edge_performance(data, graph_model, data_name, top_k, n, trained_models_path):

    edge_performance_original, edge_performance = load_sort_and_create_edge_performance_dataset(data, graph_model, data_name ,top_k, n, 
                                                                                                trained_models_path) #format of plantoid                                                                                       trained_models_path, main_seed, promotion_mode) 
    simplified_performance = extract_edge_changes(edge_performance) #just edge: change_position
    #simplified_performance_link_prediction = mark_edges_above_threshold(simplified_performance, 0, promotion_mode) #change_position greater than thresh
    #data_with_label = build_data_with_label(data, simplified_performance_link_prediction)

    return simplified_performance
#________________________________________________________________________________________________________________________

data_name = 'Cora'
#['Cora', 'CiteSeer', 'PubMed']

graph_model = 'epagcl'
#['gcn', 'sage', 'graphpatcher', 'gat2', 'epagcl']
       
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#____________________________________________________________ Folders ___________________________________________________________


trained_models_path = f"/mnt/data/khosro/Graph-Pruning/edge_performance_dataset_v2"
dataset_subgraph_path = "/mnt/data/khosro/Graph-Pruning/data/pubmed_subgraph.pt"

#____________________________________________________________ Seeds ____________________________________________________________

if data_name == 'CiteSeer': 
    dataset = Planetoid(root='data/Planetoid', name='CiteSeer')
    data = dataset[0]
elif data_name == 'Cora': 
    dataset = Planetoid(root='data/Planetoid', name='Cora')
    data = dataset[0]
elif data_name == 'PubMed':
    dataset = torch.load(dataset_subgraph_path)
    data = dataset
                        

#________________________________________________________________________________________________________________________


top_k = data.x.size(0)
n = 2
simplified_performance = load_edge_performance(data, graph_model, data_name, top_k, n, trained_models_path)

# Print all neighbors of node 55 for inspection
print("Node 12 edges:", simplified_performance[12]) #12, 75

# Find and print the first node whose edge label > 0
found = False
for node, edges in simplified_performance.items():
    for edge, label in edges.items():
        if label > 2:
            print(f"Node {node} has a positive edge: {edge} → label = {label}")
            found = True
            break
    if found:
        break

if not found:
    print("No edge with label > 0 found.")


#print("_______________all____________")
#print(simplified_performance)
#data
#data_with_label


#Create 22 last nodes from 122 nodes and load the the edge performnce for them

#________________________________________________________________________________________________________________________


