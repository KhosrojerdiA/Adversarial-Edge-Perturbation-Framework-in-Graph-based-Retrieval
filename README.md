
# Adversarial Edge Perturbation Framework in Graph-based Retrieval

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/) [![Documentation Status](https://readthedocs.org/projects/ansicolortags/badge/?version=latest)](http://ansicolortags.readthedocs.io/?badge=latest)

GRAPH In this paper, we propose a novel approach to manipulate keyword search results by strategically removing edges in a graph using graph representation learning techniques. Our method targets a specific node, representing an entity or document, and performs an attack by identifying the most influential edges whose removal demotes the node in the retrieval task. By leveraging the learned embeddings from a graph neural network (GCN or Sage), we rank the edges based on their impact on the node’s ranking position. This approach provides an effective mechanism to di- minish the visibility of targeted nodes in keyword-based searches, offering insights into adversarial attacks on graph-based search systems with constrained of budget for multi-edge removal strategy. Our experiments on real-world dataset demonstrate the efficacy of our method, showing significant demotion of target nodes with minimal edge removals, while maintaining the overall structure and relevance of the graph. This work contributes to the growing body of research on adversarial attacks in graph-based systems, with implications for information retrieval and search engine optimization.


## Installation

You need to install the the libraries in requirements.txt file using pip or conda command 

```
pip install -r requirements.txt
```

## Dataset

### Cora, CiteSeer and PubMed
We conduct experiments on a widely used benchmark, namely Cora, CiteSeer which are most used datasets for this task and PubMed. The Cora and CiteSeer datasets consists of 2,708, 3327 scientific publications as nodes, which are categorized into seven and six distinct classes representing different topics. The edges between nodes represent citation relationships, forming a directed graph. Each node is described by a 1,433 and 3703-dimensional binary feature vector. PubMed also has 19,717 nodes, 88,648 edges with 3 classes.

## Running the Model

After preparing the input data, models can be run individualy. You can then simply run the **"main.py"**  by using this command in terminal:
	
     `python <main.py>`

	* Take note that model_name parameter is initilized at the beggining of the code and can be changed for experiments for running different attack model.

For training the model, please check the model folder and run the model you want to train. 

## Evaluating the results

To evaluate the effectiveness of our adversarial attacks, we measured the average rank demotion (ARD) in retrieval scenarios. We used dataset features as keywords for search. For baseline results before the attack, we employed two graph representation methods, GCN and Graph-SAGE.

Here is the result table:

![](images/results.png)
<p align="center"><em>Results.</em></p>

## Findings
We evaluate the performance of our proposed method and baselines using both single-edge and multiple-edge removal. The motivation for multiple-edge removal is to assess model performance when given more opportunities to remove edges. We define a budget to indicate the number of allowed edge removals. The average rank demotion (ARD) results are presented in Table 2. 


## Contributing
This github is provided as a complementary materials for a ECIR 2026 short paper.

