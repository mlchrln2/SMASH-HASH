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
    'decay' : 0.95
}

if __name__ == '__main__':
    print(Options)