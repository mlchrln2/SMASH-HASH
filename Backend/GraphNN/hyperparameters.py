from grid_graph import grid_graph
from coarsening import coarsen

Options = {
    'batch_size': 100,
    'training_split': 0.7,
    'n_batches': 20,
    'grid_side': 28,
    'number_edges': 8,
    'metric': 'euclidean',
    'coarsening': 4,
    'output_folder': './data/processed/',
    'input_data': './data',
    'CL1_F': 32,
    'CL1_K': 25,
    'CL2_F': 64,
    'CL2_K': 25,
    'FC1_F': 512,
    'FC2_F': 10,
    'learning_rate': 0.05,
    'dropout_value': 0.5,
    'l2_regularization': 5e-4,
    'batch_size': 100,
    'num_epochs': 20
}

grid_side = Options['grid_side']
number_edges = Options['number_edges']
metric = Options['metric']
# create graph of Euclidean grid
Grid = grid_graph(grid_side, number_edges, metric)

# Compute coarsened graphs
coarsening_levels = Options['coarsening']
_, perm = coarsen(Grid, coarsening_levels)
Options['D'] = max(perm) + 1
del(perm)
Options['FC1Fin'] = Options['CL2_F'] * (Options['D'] // 16)

if __name__ == '__main__':
    print(Options)