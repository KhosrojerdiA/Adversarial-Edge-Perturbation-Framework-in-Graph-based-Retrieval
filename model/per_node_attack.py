import torch
from torch_geometric.utils import degree, remove_self_loops
from torch_geometric.nn import GCNConv
from torch.nn import Linear
import sys
import os
 
project_path = './'
sys.path.append(project_path)

from utils.utils import *
from utils.utils_scorer import *
from model.random_attack import *
from model.highest_degree_attack import *
from model.page_rank_attack import *
from model.viking_attack_v3 import * #train_node2vec_viking, fit_surrogate_to_node2vec, viking_attack_grad_all_edges
#from model.link_prediction_attack import *
from model.gold_attack import *

#from model.train_target_node import *
#from model.train_target_node import create_train_target_node
#from model.rl_attack import *
#from model.cluster_based_attack import *
#from model.train_reverse_link_prediction import *
#from model.train_rl import *

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
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.nn import VGAE, APPNP
from torch_geometric.utils import negative_sampling, remove_self_loops
from sklearn.metrics import roc_auc_score, average_precision_score
import torch_geometric.transforms as T
import math
from torch_geometric.data import Data
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from tqdm.notebook import tqdm
from torch_geometric.utils import degree
import random
import torch
from torch_geometric.utils import remove_self_loops
from networkx import pagerank
#____________________________________________________________________________________________________________________________________



def per_node_attack(model_name, graph_model, data_name, data, dataset_embeddings, model, selected_nodes, selected_node_embeddings, 
                    top_k_indice_at_20, top_k_indice_at_100, top_k_indice_at_500, top_k_indice_at_1000, top_k_indice_at_4000, 
                    trained_models_path, n, budget, scorer_path, result_path, promotion_mode):
    

    vgae_path = f"/mnt/data/khosro/Graph-Pruning/trained_scorer/{data_name}_{graph_model}_VGAE.pt"
    vgae_emb_path = f"/mnt/data/khosro/Graph-Pruning/trained_scorer/{data_name}_{graph_model}_VGAE_embeddings.pt"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #data = data.to(device)

#____________________________________________________________________________________________________________________________________

    #trained_models_path = f"/home/akhosrojerdi/Graph-Representation-Learning-for-Strategic-Edge-Removal-in-Keyword-Search-Demotion-Attacks/trained_models"

    training_rate = 0.1
    epochs = 2000
    channels = 128
    learning_rate = 0.01
    top_k = data.x.size(0)
    #n = 10
    #top_k: top retrieve nodes to check for demoting, n: number of most important edges to select for dataset
    data_num_nodes = data.num_nodes

    print("____________________________________________", flush=True)
    print("_")
    print(f'Model Name: {model_name:>3}')
    print("____________________________________________", flush=True)

    attacked_at_num_node_dict = {} 
     
# __________________________________________________________________________________________________________________________

    selected_node_data_structure = [{"node_id": selected_node.item(), "embedding": embedding} for selected_node, embedding in zip(selected_nodes, selected_node_embeddings)]
    step_count = 0

    if model_name == "per_node_link_prediction":
        vgae_model, vgae_embedding = load_vgae_model(data, vgae_path, vgae_emb_path)
    #elif model_name == "per_node_viking":
    #    n2v = train_node2vec_viking(data, epochs=50, device="cuda")
    #    surrogate = fit_surrogate_to_node2vec(data, n2v, hid_dim=128, epochs=100, device="cuda")
    #elif model_name == "per_node_rl":
    #    rl_model = create_train_rl_sv2_model(data, selected_nodes)
    elif model_name == "per_node_cluster":
        clusters = compute_clusters(data)
    elif model_name == "per_node_targeted_node":
        scorer_model = load_scorer_model(dataset_embeddings, scorer_path, data, feature_fn=edge_features_v5_targeted) 
    elif model_name == "per_node_gold_attack":
        simplified_performance = create_simplified_performance(data, graph_model, data_name, top_k, n, trained_models_path, promotion_mode)
 
 
# ___________________________________________________________ Loop _______________________________________________________________
    
    for node_data in selected_node_data_structure:

        node_id = node_data['node_id']
        one_node_selected_node_embedding = node_data['embedding'].to(device) #Query node embedding

        #print("____________________________________________")
        #print(f'Node # : {step_count:>3}')
        #print("________")

        if model_name == "per_node_random_attack":
            updated_data = build_random_attack(data, selected_nodes[step_count], budget, promotion_mode) 
            #print(selected_nodes[step_count]) -> tensor(55, device='cuda:0')
        elif model_name == "per_node_highest_degree":
            updated_data = build_highest_degree_attack(data, selected_nodes[step_count], budget, promotion_mode)

        elif model_name == "per_node_page_rank":
            updated_data = build_page_rank_attack(data, selected_nodes[step_count], budget, promotion_mode)

        elif model_name == "per_node_p_page_rank":
            updated_data = build_p_page_rank_attack(data, selected_nodes[step_count], budget, promotion_mode)

        elif model_name == "per_node_viking":
            #updated_data = build_viking_attack_v3(data, selected_nodes[step_count], dataset_embeddings, budget, promotion_mode)
            #updated_data = viking_attack_grad_all_edges(data, selected_nodes[step_count], surrogate, budget, promotion_mode)
            updated_data = viking_attack_per_node(data, selected_nodes[step_count], budget=budget, dim=32, window_size=5, supervised=True) 

        #elif model_name == "per_node_rl":
        #    updated_data = build_rl_attack(data, selected_nodes[step_count], rl_model, budget, promotion_mode) 

        #elif model_name == "per_node_cluster":
        #    updated_data = build_cluster_attack(data, selected_nodes[step_count], clusters, budget, promotion_mode) 

        #elif model_name == "per_node_link_prediction":
        #    updated_data = build_link_prediction_attack(data, selected_nodes[step_count], vgae_model, vgae_embedding, budget, promotion_mode) 

        elif model_name == "per_node_gold_attack":
            updated_data = build_gold_attack(data, simplified_performance, selected_nodes[step_count], budget, promotion_mode)

        elif model_name == "per_node_targeted_node":
            updated_data = build_new_target_node_attack(data, selected_nodes[step_count], budget, scorer_model, dataset_embeddings, promotion_mode)

 
        compare_original_vs_updated(data, updated_data, budget)

                                                                                                                         #tensor([12])
        attacked_dataset_embeddings, attacked_one_node_selected_node_embeddings = attacked_embedding_v2(updated_data, torch.tensor([selected_nodes[step_count]]), graph_model, model)
        attacked_at_num_node_dict = per_node_attacked_return(data_num_nodes, attacked_dataset_embeddings, attacked_one_node_selected_node_embeddings[0], node_id, 
                                                                                                                                                 attacked_at_num_node_dict)

        #print(attacked_at_num_node_dict[node_id])
        node_retrieval_rank = node_retrieval_position(node_id, top_k_indice_at_4000[step_count].tolist())
        show_query_position(data_name, model_name, graph_model, node_id, node_retrieval_rank, attacked_at_num_node_dict, result_path)
        step_count += 1

# ___________________________________________________________ Loop _______________________________________________________________
    
    #print("Data: ",data) 
    #log_attacked_positions_to_excel(
    #attacked_at_num_node_dict,
    #data_name,
    #graph_model,
    #model_name)

    
    
    return(
        per_node_dictionary_return(
                                    selected_nodes, top_k_indice_at_20, 
                                      top_k_indice_at_100, 
                                      top_k_indice_at_500, 
                                      top_k_indice_at_1000, 
                                      top_k_indice_at_4000, attacked_at_num_node_dict, data_num_nodes
                                      
                                    )
                                      
            )







