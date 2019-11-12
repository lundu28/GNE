## Galaxy Network Embedding: A Hierarchical Community Structure Preserving Approach
#### Authors: Lun Du, Zhicong Lu, Yun Wang, Guojie Song, Yiming Wang, Wei Chen

## How to use
- We upload a sample dataset Hamilton and you can run our algorithm on this dataset by the following command:
```shell
python main.py --conf hamilton_real2
```

## Introduction of Data Structure
- "edges_hamilton.txt" saves the structure of network.
- "flag_hamilton.txt" saves the labels of nodes that are utilized as ground truth in the node classification task.
- "tree2_hamilton" saves all the edges of the tree, and each pair number is a directed edge including the predecessor node ID in first and the successor node ID in back. And the leaf of the tree represents the original node in the network and their IDs are consistent.
