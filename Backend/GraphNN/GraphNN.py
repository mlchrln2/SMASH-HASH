import torch
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import argparse
import pathlib


# user defined
from Graph_Convnet import Graph_ConvNet_LeNet5
from grid_graph import grid_graph
from coarsening import coarsen, lmax_L, perm_data, rescale_L
from hyperparameters import Options

# GPU Compatibility

parser = argparse.ArgumentParser(description='Trainer')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
args = parser.parse_args()
args.device = None
if not args.disable_cuda and torch.cuda.is_available():
    # args.device = torch.device('cuda')
    # torch.set_default_tensor_type(torch.cuda.FloatTensor)
    args.device = torch.device('cpu')
else:
    args.device = torch.device('cpu')


class SpectralGraph_Trainer:

    def __init__(self, options, dataset, output_folder):
        self.options = options
        self.n_samples = len(dataset)
        self.n_training_events = int(
            self.options['training_split'] * self.n_events)
        self.dataset = dataset
        self.history_logger = SummaryWriter(output_folder)

    def prepare(self):
        # split train and test
        self.training_events = self.dataset[:self.n_training_events]
        self.test_events = self.dataset[self.n_training_events:]
        # set up model
        self.model = Graph_ConvNet_LeNet5(self.options)

    def make_batch(self):
        training_loader = DataLoader(
            self.training_events, batch_size=self.options["batch_size"],
            shuffle=True, drop_last=True)

        testing_loader = DataLoader(
            self.testing_events, batch_size=self.options["batch_size"],
            shuffle=True, drop_last=True)

        return training_loader, testing_loader

    def train(self, Print=True):
        train_loss = 0
        train_acc = 0
        for batch_n in range(self.options["n_batches"]):
            training_batch, testing_batch = self.make_batch()
            train_loss, train_acc, _, _ = self.model.do_train(training_batch)
            test_loss, test_acc, _, _ = self.model.do_eval(testing_batch)
            self.history_logger.add_scalar(
                'Accuracy/Train Accuracy', train_acc, batch_n)
            self.history_logger.add_scalar(
                'Accuracy/Test Accuracy', test_acc, batch_n)
            self.history_logger.add_scalar(
                'Loss/Train Loss', train_loss, batch_n)
            self.history_logger.add_scalar(
                'Loss/Test Loss', test_loss, batch_n)
            if Print:
                print("Batch: %d, Train Loss: %0.4f, Train Acc: %0.4f, "
                      "Test Loss: %0.4f, Test Acc: %0.4f" % (
                          batch_n, train_loss, train_acc, test_loss, test_acc))
        return train_loss

    def test(self):
        testing_loader = DataLoader(
            self.testing_events, batch_size=self.options['batch_size'],
            collate_fn=collate, shuffle=True, drop_last=True)

        _, _, self.test_raw_results, self.test_truth = self.model.do_eval(
            testing_loader)

    def train_and_test(self, do_print=True, save=True):
        '''Function to run and the execute the network'''
        self.prepare()
        loss = self.train(do_print)
        self.test()
        return loss

    def save_model(self, save_path):
        net, optimizer = self.model.get_model()
        torch.save(net, save_path + "/saved_net.pt")
        torch.save(optimizer, save_path + "/saved_optimizer.pt")

    def log_history(self, save_path):
        # self.history_logger.export_scalars_to_json(save_path + "/scalar_history.json")
        self.history_logger.close()


def train(options):
    # load data
    dataset = MNIST('./data/', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(
                            (0.1307,), (0.3081,))
                    ]))

    # prepare outputs
    output_folder = options['output_folder']
    if not pathlib.Path(output_folder).exists():
        pathlib.Path(output_folder).mkdir(parents=True)

    # perform training
    SpectralGraph_trainer = SpectralGraph_Trainer(
        options, dataset, output_folder)
    SpectralGraph_trainer.train_and_test()

    # save results
    SpectralGraph_trainer.log_history(output_folder)
    SpectralGraph_trainer.save_model(output_folder)


if __name__ == '__main__':
    options = Options
    batch_size = options['batch_size']
    dataset = MNIST('./data/', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(
                            (0.1307,), (0.3081,))
                    ]))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    img, truth = 0, 0
    # Construct graph
    grid_side = options['grid_side']
    number_edges = options['number_edges']
    metric = options['metric']
    # create graph of Euclidean grid
    A = grid_graph(grid_side, number_edges, metric)

    # Compute coarsened graphs
    coarsening_levels = options['coarsening']
    L, perm = coarsen(A, coarsening_levels)

    # Compute max eigenvalue of graph Laplacians
    lmax = []
    for i in range(coarsening_levels):
        lmax.append(lmax_L(L[i]))
    print('lmax: ' + str([lmax[i] for i in range(coarsening_levels)]))

    train_data = []
    train_labels = []
    for i in dataloader:
        train_data, train_labels = i
        break

    print(train_data.reshape(100, -1).shape)
    # pdb.set_trace()
    # Reindex nodes to satisfy a binary tree structure
    train_data = perm_data(train_data.reshape(100, -1), perm)

    print(train_data)
    print(train_data.shape)
