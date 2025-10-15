
import sys
import os

#os.system("nvidia-smi")
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"


project_path = '/mnt/data/khosro/Graph-Pruning'
sys.path.append(project_path)

# Set a seed for reproducibility
#torch.manual_seed(3708) 

from utils.utils import *
#from model.retrieval import * 
from model.retrieval_epaglc import * 
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
#torch.backends.cudnn.deterministic = True  # Force deterministic behavior
#torch.backends.cudnn.benchmark = False  # Disable auto-tuning for convolution algorithms



#____________________________________________________________ Inputs ____________________________________________________________



data_name_list = ['Cora', 'CiteSeer', 'PubMed']
#['Cora', 'CiteSeer', 'PubMed']



model_name_list = ['per_node_random_attack', 'per_node_highest_degree', 'per_node_p_page_rank', 'per_node_viking',
                   'per_node_gold_attack', 'per_node_targeted_node']
                   
#['per_node_highest_degree', 'per_node_p_page_rank', 'per_node_viking', 'per_node_gold_attack', 'per_node_targeted_node']


graph_model_list = ['epagcl_gcn', 'epagcl_sage']
#'epagcl_gcn', 'epagcl_sage'
       
text = "ex_1_v1_citeseer_gcn_model_false"


scorer_path = f'/mnt/data/khosro/Graph-Pruning/trained_scorer/{data_name_list[0]}_{graph_model_list[0]}_scorer_model_100_epochs_s4_ex8.pt'
#scorer_path = f'/mnt/data/khosro/Graph-Pruning/trained_scorer/{data_name_list[0]}_{model_name_list[0]}_scorer_model_50_epochs.pt'


budget_list = [1, 2, 3, 4, 5] 
#[1, 2, 3, 4, 5] 
#Number of budget we have for removing the edges (how many edges we allow to remove per node)

promotion_mode = False #True for Promotion, False for Demotion

runs = 1
#5
#Number of different random seeds 


min_number_edges_list = [10]
#Minimum number of edges connected to selected nodes (query) [5,10]

#________________________________________________________________________________________________________________________________

n = 10
#Number of most important edges to select for per node attack dataset generation

#top_k: top retrieve nodes to check for demoting

#____________________________________________________________ Folders ___________________________________________________________

result_path = f"/mnt/data/khosro/Graph-Pruning/outputs"

embedding_save_dir = "/mnt/data/khosro/Graph-Pruning/embeddings_v2"
trained_models_path = f"/mnt/data/khosro/Graph-Pruning/edge_performance_dataset_v2"
dataset_subgraph_path = "/mnt/data/khosro/Graph-Pruning/data/pubmed_subgraph.pt"

#____________________________________________________________ Seeds ____________________________________________________________


main_seed = 3708

#____________________________________________________________ Start of the loop ________________________________________________

for data_name in data_name_list:
    for graph_model in graph_model_list:
        for min_number_edges in min_number_edges_list:
            for budget in budget_list:
                for model_name in model_name_list:
            
                        print("NOTE____________________Starting Stats________________________", flush=True)
                        print(f'data_name: {data_name}', flush=True)
                        print(f'model_name: {model_name}', flush=True)
                        print(f'graph_model: {graph_model}', flush=True)
                        print(f'min_number_edges: {min_number_edges}', flush=True)
                        print("____________________Starting Stats________________________")



                        step_count = 1

                        retrieval_found_count = []
                        retrieval_recall = []
                        retrieval_avg_position = []
                        retrieval_avg_promoted = []
                        retrieval_avg_demoted = []
                        retrieval_avg_changed = []
                        retrieval_avg_unchanged = []

                        attacked_recall = []
                        attacked_retrieval_node_found_count = []
                        attacked_retrieval_node_avg_position = []
                        attacked_avg_promoted = []
                        attacked_avg_demoted = []
                        attacked_avg_changed = []
                        attacked_avg_unchanged = []
                        retrieval_position_after_attack = []

                        duration_per_run = []

                        #________________________________________________________________________________________________________________________



                        if data_name == 'CiteSeer': 
                            dataset = Planetoid(root='data/Planetoid', name='CiteSeer')
                            data = dataset[0]
                        elif data_name == 'Cora': 
                            dataset = Planetoid(root='data/Planetoid', name='Cora')
                            data = dataset[0]
                        elif data_name == 'PubMed':
                            dataset = torch.load(dataset_subgraph_path, weights_only=False) #torch.load(dataset_subgraph_path)
                            data = remove_isolated_nodes(dataset)

                        #data.is_undirected()

                        #____________________________________________________________ Loop ____________________________________________________________

                        for seed_idx in range(runs): 

                            print("***************************************************************************************************************************************", flush=True)
                            print("***************************************************************************************************************************************", flush=True)
                            print(f'Run Number {step_count:>3} for {data_name}_{model_name}_{graph_model}_{min_number_edges}')
                            print("***************************************************************************************************************************************", flush=True)
                            print("***************************************************************************************************************************************", flush=True)

                            #Retrieval
                            (
                
                            dataset_embeddings, model, selected_nodes, selected_node_embeddings, 
                            top_k_indice_at_20, top_k_indice_at_100, top_k_indice_at_500, top_k_indice_at_1000,top_k_indice_at_4000, 
                            found_count_20, found_count_100, found_count_500, found_count_1000, found_count_4000, 
                            recall_20, recall_100, recall_500, recall_1000, recall_4000, 
                            avg_position_20, avg_position_100, avg_position_500, avg_position_1000, avg_position_4000 

                            ) = retrieval_v3(data, data_name, graph_model, min_number_edges, embedding_save_dir, main_seed)
                            

                            print("____________________###________________________", flush=True)
                            print(f'Number of Selected Nodes: {len(selected_nodes)}', flush=True) 
                            print("____________________###________________________", flush=True)
                            viz_selected_nodes_edges(data, selected_nodes)
                            print("____________________###________________________", flush=True)

                            start_time = time.time()

                            if model_name in  ['per_node_highest_degree', 'per_node_p_page_rank']:
                                promotion_mode = False 
                            else: 
                                promotion_mode = True

                            #Attack
                            (
                                
                            attacked_recall_20, attacked_retrieval_node_found_count_20, attacked_retrieval_node_avg_position_20, attacked_avg_promoted_20, attacked_avg_demoted_20, attacked_avg_changed_20, attacked_avg_unchanged_20,
                            attacked_recall_100, attacked_retrieval_node_found_count_100, attacked_retrieval_node_avg_position_100, attacked_avg_promoted_100, attacked_avg_demoted_100, attacked_avg_changed_100, attacked_avg_unchanged_100,
                            attacked_recall_500, attacked_retrieval_node_found_count_500, attacked_retrieval_node_avg_position_500, attacked_avg_promoted_500, attacked_avg_demoted_500, attacked_avg_changed_500, attacked_avg_unchanged_500,
                            attacked_recall_1000, attacked_retrieval_node_found_count_1000, attacked_retrieval_node_avg_position_1000, attacked_avg_promoted_1000, attacked_avg_demoted_1000, attacked_avg_changed_1000, attacked_avg_unchanged_1000,
                            attacked_recall_4000, attacked_retrieval_node_found_count_4000, attacked_retrieval_node_avg_position_4000, attacked_avg_promoted_4000, attacked_avg_demoted_4000, attacked_avg_changed_4000, attacked_avg_unchanged_4000

                            )= per_node_attack(model_name, graph_model, data_name, data, dataset_embeddings, model, selected_nodes, selected_node_embeddings, 
                                            top_k_indice_at_20, top_k_indice_at_100, top_k_indice_at_500, top_k_indice_at_1000, top_k_indice_at_4000, 
                                            trained_models_path, n, budget, scorer_path, promotion_mode)

                            end_time = time.time()
                            duration = end_time - start_time
                            duration_per_run.append(duration)

                            #Results
                            retrieval_found_count.extend([found_count_20, found_count_100, found_count_500, found_count_1000, found_count_4000])
                            retrieval_recall.extend([recall_20, recall_100, recall_500, recall_1000, recall_4000])
                            retrieval_avg_position.extend([avg_position_20, avg_position_100, avg_position_500, avg_position_1000, avg_position_4000])

                            attacked_recall.extend([attacked_recall_20, attacked_recall_100, attacked_recall_500, attacked_recall_1000, attacked_recall_4000])
                            attacked_retrieval_node_found_count.extend([attacked_retrieval_node_found_count_20, attacked_retrieval_node_found_count_100, attacked_retrieval_node_found_count_500, attacked_retrieval_node_found_count_1000, attacked_retrieval_node_found_count_4000])
                            attacked_retrieval_node_avg_position.extend([attacked_retrieval_node_avg_position_20, attacked_retrieval_node_avg_position_100, attacked_retrieval_node_avg_position_500, attacked_retrieval_node_avg_position_1000, attacked_retrieval_node_avg_position_4000])
                            attacked_avg_promoted.extend([attacked_avg_promoted_20, attacked_avg_promoted_100, attacked_avg_promoted_500, attacked_avg_promoted_1000, attacked_avg_promoted_4000])
                            attacked_avg_demoted.extend([attacked_avg_demoted_20, attacked_avg_demoted_100, attacked_avg_demoted_500, attacked_avg_demoted_1000, attacked_avg_demoted_4000])
                            attacked_avg_changed.extend([attacked_avg_changed_20, attacked_avg_changed_100, attacked_avg_changed_500, attacked_avg_changed_1000, attacked_avg_changed_4000])
                            attacked_avg_unchanged.extend([attacked_avg_unchanged_20, attacked_avg_unchanged_100, attacked_avg_unchanged_500, attacked_avg_unchanged_1000, attacked_avg_unchanged_4000])
                            #retrieval_position_after_attack.extend([retrieval_position_after_attack_20, retrieval_position_after_attack_100, retrieval_position_after_attack_500, retrieval_position_after_attack_1000, retrieval_position_after_attack_4000])
                            
                            
                            step_count += 1
                            
                        #____________________________________________________________ Result ____________________________________________________________

                        print("____________________###________________________", flush=True)
                        print("All Runs are Done!", flush=True)

                        #retrieval_store_to_excel(data_name, graph_model, min_number_edges, model_name, retrieval_found_count, retrieval_recall, retrieval_avg_position, result_path)
                        print("____________________###________________________", flush=True)
                        store_to_excel(runs, data_name, graph_model, min_number_edges, model_name, budget, 
                                    retrieval_found_count, retrieval_recall, retrieval_avg_position, 
                                    attacked_recall, attacked_retrieval_node_found_count, attacked_retrieval_node_avg_position,
                                    attacked_avg_promoted, attacked_avg_demoted, 
                                    attacked_avg_changed, attacked_avg_unchanged, duration_per_run, result_path, promotion_mode, text)
                        
                        print(f'Results are ready for {data_name}_{graph_model}_{min_number_edges}_{budget}_{model_name} and appended to excel!', flush=True)
                    
process_excel_sheets(result_path, data_name, promotion_mode, text)

#____________________________________________________________ End of the loop ________________________________________________ min_number_edges

