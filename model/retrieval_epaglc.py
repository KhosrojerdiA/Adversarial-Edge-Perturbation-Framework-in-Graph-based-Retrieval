
import sys
import os

project_path = './'
sys.path.append(project_path)

from utils.utils import *
from EPAGCL.model_3 import *  

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn import Linear
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx
from torch.nn import Linear
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch
from torch_geometric.utils import degree
from torch_geometric.data import Data
from torch.nn import Linear
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.nn import GATv2Conv



def retrieval_v2(data, data_name, graph_model, min_number_edges, save_dir, main_seed):

    torch.manual_seed(main_seed)
    torch.cuda.manual_seed(main_seed)
    np.random.seed(main_seed)
    random.seed(main_seed)
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False  
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if graph_model == 'gat2':
        torch.use_deterministic_algorithms(True, warn_only=True)  # Enforce determinism

    #import os
    #os.environ['PYTHONHASHSEED'] = str(main_seed)
    #torch.set_num_threads(1)  # Reduce variability due to multi-threading


    data = data.to(device)  # Move data to the appropriate device 
    data_num_nodes = data.num_nodes


    if graph_model.lower() == "epagcl_gcn" or graph_model.lower() == "epagcl_sage":

        epochs=2000

        file_prefix = f"{data_name}_{graph_model}_epochs_{epochs}"
        emb_path = os.path.join(save_dir, f"{file_prefix}_embeddings.pt")
        model_path = os.path.join(save_dir, f"{file_prefix}_model.pt")

        # ---------------------------------------------------------
        # Check if cached embeddings & model exist
        # ---------------------------------------------------------
        if os.path.exists(emb_path) and os.path.exists(model_path):
            print(f">> Found cached EPAGCL {graph_model} model and embeddings for {data_name}! Loading from disk...", flush=True)
            print(f"Location: {emb_path}", flush=True)
            dataset_embeddings = torch.load(emb_path, map_location="cpu")
            model = torch.load(model_path, map_location=device)
            model.eval()
            print(f"- Loaded existing EPAGCL {graph_model} for {data_name} embeddings and model successfully!", flush=True)
            print("____________________________________________")
            
            # 🔧 Fix for numerical drift between old embeddings and new runtime
            #dataset_embeddings = dataset_embeddings.to(torch.float32)
            #dataset_embeddings = F.normalize(dataset_embeddings, p=2, dim=1)

        else:
            print(f">> No cached files found for {data_name}. Training EPAGCL from scratch...", flush=True)
            dataset_embeddings, model = get_epagcl_embeddings(
                data=data,
                device=str(device),
                epochs=epochs,
                batch_compute=False,
                add_single=False,
                not_add_edge=False,
                not_drop_edge=False,
                add_edge_random=False,
                graph_model=graph_model
            )
            model.eval()

            # Save both embeddings and model
            torch.save(dataset_embeddings, emb_path)
            torch.save(model, model_path)

            print(f"- EPAGCL training complete. Saved to:", flush=True)
            print(f"  • {emb_path}")
            print(f"  • {model_path}")
            print("____________________________________________")
#________________________________________________________________________

    else:

        class GraphModel(torch.nn.Module):
            def __init__(self, graph_model):
                num_classes = int(data.y.max().item()) + 1  
                super().__init__()

                if graph_model == 'gcn':
                    self.conv = GCNConv(data.num_features, 300).to(device)
                elif graph_model == 'gat':
                    self.conv = GATConv(data.num_features, 300).to(device)
                elif graph_model == 'sage':
                    self.conv = SAGEConv(data.num_features, 300).to(device)
                elif graph_model == 'graphpatcher':
                    self.conv = GraphPatcherLayer(data.num_features, 300).to(device)
                elif graph_model == 'gat2':
                    self.conv = GATv2Conv(data.num_features, 300, heads=1).to(device)

                else:
                    raise ValueError("Invalid graph_model. Choose from 'gcn', 'gat2', 'sage', or 'graphpatcher'.")

                self.out = Linear(300, num_classes).to(device)

            #    self.reset_parameters()  # Ensure deterministic initialization
            #def reset_parameters(self):
            #    for layer in self.children():
            #        if hasattr(layer, 'reset_parameters'):
            #            layer.reset_parameters()

            def forward(self, x, edge_index):
                x = x.to(device)
                edge_index = edge_index.to(device)
                h = self.conv(x, edge_index)
                h = F.relu(h)
                z = self.out(h)
                return h, z
        #_________________________________

        model = GraphModel(graph_model).to(device)

        criterion = torch.nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

        def accuracy(pred_y, y):
            return (pred_y == y).sum().float() / len(y)

        embeddings = []
        losses = []
        accuracies = []
        outputs = []

        for epoch in range(200):
            model.train()
            optimizer.zero_grad()
            h, z = model(data.x, data.edge_index)
            loss = criterion(z, data.y.to(device))
            acc = accuracy(z.argmax(dim=1), data.y.to(device))
            loss.backward()
            optimizer.step()

            embeddings.append(h)
            losses.append(loss.item())
            accuracies.append(acc.item())
            outputs.append(z.argmax(dim=1))

        model.eval()
        with torch.no_grad():
            h, z = model(data.x, data.edge_index)

        dataset_embeddings = h

    #____________________________________________________________________________________________________________________________________

    print("- Dataset Embedding is Generated!", flush=True)
    print("____________________________________________", flush=True)


    #____________________________________________________________________________________________________________________________________

    num_nodes = data.x.size(0)
    edge_index = data.edge_index
    node_degrees = degree(edge_index[0], num_nodes=num_nodes).to(device)


    nodes_with_min_edges = (node_degrees >= min_number_edges).nonzero(as_tuple=True)[0]

    if data_name in ['Cora', 'CiteSeer']:
        selected_nodes = nodes_with_min_edges
        all_nodes = selected_nodes
        print(f"Number of all edges with at least {min_number_edges}: {len(selected_nodes)}")
        selected_nodes = selected_nodes[:100]  #NEW     First 100 elements
    elif data_name == 'PubMed':
        max_nodes = int(0.05 * num_nodes)
        selected_nodes = nodes_with_min_edges[:max_nodes]
        print(f"Number of all edges with at least {min_number_edges}: {len(selected_nodes)}")
        all_nodes = selected_nodes
        selected_nodes = selected_nodes[:100]  #NEW     First 100 elements
 

    selected_nodes_set = set(selected_nodes.tolist())
    mask = [(edge[0].item() not in selected_nodes_set) and (edge[1].item() not in selected_nodes_set) for edge in data.edge_index.t()]
    mask = torch.tensor(mask, dtype=torch.bool, device=device)
    isolated_edge_index = data.edge_index[:, mask]


    if graph_model.lower() == "epagcl_gcn" or graph_model.lower() == "epagcl_sage":
        # Use pre-trained embeddings directly (no classifier layer)
        selected_node_embeddings = dataset_embeddings[selected_nodes.cpu()]
    else:
        model.eval()
        with torch.no_grad():
            h, z = model(data.x, isolated_edge_index)
        selected_node_embeddings = h[selected_nodes]
        
    # 🔧 Normalize selected node embeddings for stable cosine similarity
    #selected_node_embeddings = F.normalize(selected_node_embeddings, p=2, dim=1)


    print("- Query Embedding is Generated!", flush=True)
    print("____________________________________________", flush=True)

    #____________________________________________________________________________________________________________________________________

    top_k = 20
    top_k_indice_at_20 = similarity(top_k, selected_node_embeddings, dataset_embeddings)
    found_count_20, recall_20, avg_position_20 = summary(selected_nodes, top_k_indice_at_20)

    top_k = 100
    top_k_indice_at_100 = similarity(top_k, selected_node_embeddings, dataset_embeddings)
    found_count_100, recall_100, avg_position_100 = summary(selected_nodes, top_k_indice_at_100)

    top_k = 500
    top_k_indice_at_500 = similarity(top_k, selected_node_embeddings, dataset_embeddings)
    found_count_500, recall_500, avg_position_500 = summary(selected_nodes, top_k_indice_at_500)

    top_k = 1000
    top_k_indice_at_1000 = similarity(top_k, selected_node_embeddings, dataset_embeddings)
    found_count_1000, recall_1000, avg_position_1000 = summary(selected_nodes, top_k_indice_at_1000)

    top_k = data_num_nodes
    top_k_indice_at_4000 = similarity(top_k, selected_node_embeddings, dataset_embeddings)
    found_count_4000, recall_4000, avg_position_4000 = summary(selected_nodes, top_k_indice_at_4000)


    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.use_deterministic_algorithms(False)
    torch.seed()           # re-randomize global torch RNG
    np.random.seed(None)   # reset numpy RNG to random state
    random.seed()          # reset Python RNG to random state



    #____________________________________________________________________________________________________________________________________

    return (
        dataset_embeddings, model, selected_nodes, selected_node_embeddings,
        top_k_indice_at_20, top_k_indice_at_100, top_k_indice_at_500,
        top_k_indice_at_1000, top_k_indice_at_4000,
        found_count_20, found_count_100, found_count_500, found_count_1000, found_count_4000,
        recall_20, recall_100, recall_500, recall_1000, recall_4000,
        avg_position_20, avg_position_100, avg_position_500, avg_position_1000, avg_position_4000, 
        
    )


