# Dock2D
Predicting the physical interaction of proteins is a cornerstone problem in computational biology. New learning-based algorithms are typically trained end-to-end on protein structures extracted from the Protein Data Bank. However, these training datasets tend to be large and difficult to use for prototyping and, unlike image or natural language datasets, they are not easily interpretable by non-experts.

In this paper we propose Dock2D-IP and Dock2D-FI, two toy datasets that can be used to select algorithms predicting protein-protein interactions (or any other type of molecular interactions). Using two-dimensional shapes as input, each example from Dock2D-FI describes the fact of interaction (FI) between two shapes and each example from Dock2D-IP describes the interaction pose (IP) of two shapes known to interact.

We propose baselines that represent different approaches to the problem and demonstrate the potential for transfer learning across the IP prediction and FI prediction tasks.

# Datasets 
The generated dataset is available here:

The main dataset generation script is *generate_dataset.py*. Example usage:

Then to visualize different metrics presented in the paper you can use:
