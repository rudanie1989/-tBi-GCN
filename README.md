# tBi-GCN
This is a repository for our paper "[Early Rumour Detection with Temporal Bidirectional Graph Convolutional Networks](https://aisel.aisnet.org/pacis2021/74/)"

# Paper Abstract

Automatic rumour detection has drawn significant research attention and deep learning models are proposed. It is shown that misinformation propagates further and wider on social networks. Existing research has focused on using the information propagation pattern for rumour detection. But the temporal propagation pattern for rumours has been largely ignored. This paper addresses this gap. We propose a temporal Bi-directional Graph Convolutional Network (tBi-GCN) model to learn representations for rumour propagation and rumour dispersion by encoding the temporal information for local graph structures and nodes. Specifically, we constructed a time-weighted adjacency matrix to represent the effect of time delay between nodes on information dissemination. Experimental results across five events of the PHEME dataset show that tBi-GCN can achieve a comparable performance in comparison with several state-of-the-art models for early rumour detection.

# Dataset

The paper used the PHEME dataset ([Link](https://figshare.com/articles/dataset/PHEME_dataset_of_rumours_and_non-rumours/4010619)) to evaluate our proposed tBi-GCN model and baselines. PHEME dataset includes five different events: Charlie Hebdo, Ferguson, Germanwings Crash, Ottawa Shooting, and Sydney Siege, with annotations of rumours and non-rumour tweets.

# Requirements
python
numpy
torch
torch_scatter
torch_sparse
torch_cluster
torch_geometric
tqdm
# Usage
Run the following command:
```bash
python tBi-GCN.py
```
# References
[1] Arkaitz Zubiaga, Maria Liakata, and Rob Procter. Exploiting context for rumour
 detection in social media. In International Conference on Social Informatics, pages
 109–123. Springer, 2017.

 [2] Thomas N Kipf and Max Welling. Semi-supervised classification with graph con
volutional networks. arXiv preprint arXiv:1609.02907, 2016.
[3] Tian Bian, Xi Xiao, Tingyang Xu, Peilin Zhao, Wenbing Huang, Yu Rong, and
 Junzhou Huang. Rumor detection on social media with bi-directional graph convo
lutional networks. In Proceedings of the AAAI Conference on Artificial Intelligence,
 volume 34, pages 549–556, 2020.
 
# Citation
if you find this code and paper are useful, please cite our paper.

@inproceedings{nie2021early,
  title={Early Rumour Detection with Temporal Bidirectional Graph Convolutional Networks.},
  author={Nie, H Ruda and Zhang, Xiuzhen and Li, Minyi and Baglin, James and Dolgun, Anil},
  booktitle={PACIS},
  pages={74},
  year={2021}
}
