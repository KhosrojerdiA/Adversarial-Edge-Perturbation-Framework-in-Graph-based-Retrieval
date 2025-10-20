
# Adversarial Edge Perturbation Framework in Graph-based Retrieval

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/) [![Documentation Status](https://readthedocs.org/projects/ansicolortags/badge/?version=latest)](http://ansicolortags.readthedocs.io/?badge=latest)

We propose a data-driven approach for adversarial edge removal that learns to estimate the effect of each edge on retrieval rankings. Rather than relying on gradient access or retraining, the framework trains a neural estimator to approximate the mapping between local edge representations and their induced ranking degradation. Once trained, the model efficiently identifies high-impact edges within a constrained perturbation budget, enabling targeted rank demotion. The contributions of this work are threefold: (1) We provide a formal analysis showing that traditional structural heuristics are non-deterministic predictors of adversarial sensitivity in graph-based retrieval. (2) We introduce a learning-based framework that estimates the influence of individual edges on ranking outcomes through data-driven approximation rather than structural assumptions. (3) We demonstrate, through experiments on real-world graph retrieval benchmarks, that the proposed method achieves stronger and more efficient rank demotion than both heuristic and gradient-based baselines


## Installation

You need to install the the libraries in requirements.txt file using pip or conda command 

```
pip install -r requirements.txt
```

## Dataset

### Cora, CiteSeer and PubMed
We conduct experiments on a widely used benchmark, namely Cora, CiteSeer which are most used datasets for this task and PubMed. The Cora and CiteSeer datasets consists of 2,708, 3327 scientific publications as nodes, which are categorized into seven and six distinct classes representing different topics. The edges between nodes represent citation relationships, forming a directed graph. Each node is described by a 1,433 and 3703-dimensional binary feature vector. PubMed also has 19,717 nodes, 88,648 edges with 3 classes.

## Running the Model

For running Edge Performance dataset, simply run **/edge_performance_dataset_v2/edge_performance_v2.py** by changing the method and dataset name. 

For trained the estimator using Edge Performance dataset, run **"train_aepf.py"** by changing the method and dataset name. The trained model will be saved in trained_scorer folder. 

After preparing the input data, models can be run individualy. You can then simply run the **"attack_setup.py"**  by using this command in terminal:
	
     `python <attack_setup.py>`

	* Take note that model_name and dataset_name parameters are initilized at the beggining of the code and can be changed for experiments for running different attack model.

For training the model, please check the model folder and run the model you want to train. 

## Evaluating the results

To evaluate the effectiveness of our adversarial attacks, we measured the average rank demotion (ARD) in retrieval scenarios. We used dataset features as keywords for search. For baseline results before the attack, we employed EPAGLC for our represenation method with two setting using GCN and Graph-SAGE.


## Findings
We evaluate the performance of our proposed method and baselines using both single-edge and multiple-edge removal. The motivation for multiple-edge removal is to assess model performance when given more opportunities to remove edges. We define a budget to indicate the number of allowed edge removals. The average rank demotion (ARD) results are presented in the paper.


## Contributing
This github is provided as a complementary materials for a ECIR 2026 short paper.

