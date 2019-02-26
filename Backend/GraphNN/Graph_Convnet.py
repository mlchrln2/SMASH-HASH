from coarsening import lmax_L
from coarsening import perm_data
from coarsening import rescale_L
import torch
import torch.nn as nn
import argparse
# class definitions

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


class sparse_mm(torch.autograd.Function):
    """
    Implementation of a new autograd function for sparse variables, 
    called "sparse_mm", by subclassing torch.autograd.Function 
    and implementing the forward and backward passes.
    """
    @staticmethod
    def forward(self, W, x):  # W is SPARSE
        self.save_for_backward(W, x)
        y = torch.mm(W, x)
        return y

    @staticmethod
    def backward(self, grad_output):
        W, x = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input_dL_dW = torch.mm(grad_input, x.t())
        grad_input_dL_dx = torch.mm(W.t(), grad_input)
        return grad_input_dL_dW, grad_input_dL_dx



class Graph_ConvNet(nn.Module):

    def __init__(self, options):

        print('Graph ConvNet: LeNet5')

        super(Graph_ConvNet_LeNet5, self).__init__()

        # parameters
        self.options = options
        self.global_step = 0  # for adjusting lr
        CL1_F = options['CL1_F']
        CL1_K = options['CL1_K']
        CL2_F = options['CL2_F']
        CL2_K = options['CL2_K']
        FC1_F = options['FC1_F']
        FC2_F = options['FC2_F']
        D = options['D']
        FC1Fin = options['FC1Fin']

        # graph CL1
        self.cl1 = nn.Linear(CL1_K, CL1_F)
        Fin = CL1_K
        Fout = CL1_F
        scale = torch.sqrt(2.0 / (Fin + Fout))
        self.cl1.weight.data.uniform_(-scale, scale)
        self.cl1.bias.data.fill_(0.0)
        self.CL1_K = CL1_K
        self.CL1_F = CL1_F

        # graph CL2
        self.cl2 = nn.Linear(CL2_K * CL1_F, CL2_F)
        Fin = CL2_K * CL1_F
        Fout = CL2_F
        scale = torch.sqrt(2.0 / (Fin + Fout))
        self.cl2.weight.data.uniform_(-scale, scale)
        self.cl2.bias.data.fill_(0.0)
        self.CL2_K = CL2_K
        self.CL2_F = CL2_F

        # FC1
        self.fc1 = nn.Linear(FC1Fin, FC1_F)
        Fin = FC1Fin
        Fout = FC1_F
        scale = torch.sqrt(2.0 / (Fin + Fout))
        self.fc1.weight.data.uniform_(-scale, scale)
        self.fc1.bias.data.fill_(0.0)
        self.FC1Fin = FC1Fin

        # FC2
        self.fc2 = nn.Linear(FC1_F, FC2_F)
        Fin = FC1_F
        Fout = FC2_F
        scale = torch.sqrt(2.0 / (Fin + Fout))
        self.fc2.weight.data.uniform_(-scale, scale)
        self.fc2.bias.data.fill_(0.0)

        # no of parameters
        no_param = CL1_K * CL1_F + CL1_F          # CL1
        no_param += CL2_K * CL1_F * CL2_F + CL2_F  # CL2
        no_param += FC1Fin * FC1_F + FC1_F        # FC1
        no_param += FC1_F * FC2_F + FC2_F         # FC2
        print('no of parameters=', no_param, '\n')

    def init_weights(self, W, Fin, Fout):

        scale = torch.sqrt(2.0 / (Fin + Fout))
        W.uniform_(-scale, scale)

        return W

    def graph_conv_cheby(self, x, cl, L, lmax, Fout, K):

        # parameters
        # B = batch size
        # V = number of vertices
        # Fin = number of input features
        # Fout = number of output features
        # K = Chebyshev order & support size
        B, V, Fin = x.size()
        B, V, Fin = int(B), int(V), int(Fin)

        # rescale Laplacian
        lmax = lmax_L(L)
        L = rescale_L(L, lmax)

        # convert scipy sparse matrix L to pytorch
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col)).T
        indices = indices.astype(np.int64)
        indices = torch.from_numpy(indices)
        indices = indices.type(torch.LongTensor)
        L_data = L.data.astype(np.float32)
        L_data = torch.from_numpy(L_data)
        L_data = L_data.type(torch.FloatTensor)
        L = torch.sparse.FloatTensor(indices, L_data, torch.Size(L.shape))
        L = L.requires_grad_(False).to(args.device)

        # transform to Chebyshev basis
        x0 = x.permute(1, 2, 0).contiguous()  # V x Fin x B
        x0 = x0.view([V, Fin * B])            # V x Fin*B
        x = x0.unsqueeze(0)                 # 1 x V x Fin*B

        if K > 1:
            x1 = my_sparse_mm()(L, x0)              # V x Fin*B
            x = torch.cat((x, x1.unsqueeze(0)), 0)  # 2 x V x Fin*B
        for k in range(2, K):
            x2 = 2 * my_sparse_mm()(L, x1) - x0
            x = torch.cat((x, x2.unsqueeze(0)), 0)  # M x Fin*B
            x0, x1 = x1, x2

        x = x.view([K, V, Fin, B])           # K x V x Fin x B
        x = x.permute(3, 1, 2, 0).contiguous()  # B x V x Fin x K
        x = x.view([B * V, Fin * K])             # B*V x Fin*K

        # Compose linearly Fin features to get Fout features
        x = cl(x)                            # B*V x Fout
        x = x.view([B, V, Fout])             # B x V x Fout

        return x

    # Max pooling of size p. Must be a power of 2.
    def graph_max_pool(self, x, p):
        if p > 1:
            x = x.permute(0, 2, 1).contiguous()  # x = B x F x V
            x = nn.MaxPool1d(p)(x)             # B x F x V/p
            x = x.permute(0, 2, 1).contiguous()  # x = B x V/p x F
            return x
        else:
            return x

    def forward(self, x, d, L, lmax):

        # graph CL1
        x = x.unsqueeze(2)  # B x V x Fin=1
        x = self.graph_conv_cheby(
            x, self.cl1, L[0], lmax[0], self.CL1_F, self.CL1_K)
        x = F.relu(x)
        x = self.graph_max_pool(x, 4)

        # graph CL2
        x = self.graph_conv_cheby(
            x, self.cl2, L[2], lmax[2], self.CL2_F, self.CL2_K)
        x = F.relu(x)
        x = self.graph_max_pool(x, 4)

        # FC1
        x = x.view(-1, self.FC1Fin)
        x = self.fc1(x)
        x = F.relu(x)
        x = nn.Dropout(d)(x)

        # FC2
        x = self.fc2(x)

        return x

    def loss(self, y, y_target, l2_regularization):

        loss = nn.CrossEntropyLoss()(y, y_target)

        l2_loss = 0.0
        for param in self.parameters():
            data = param * param
            l2_loss += data.sum()
        loss += 0.5 * l2_regularization * l2_loss

        return loss

    def update(self, lr):

        update = torch.optim.SGD(self.parameters(), lr=lr, momentum=0.9)

        return update

    def update_learning_rate(self, optimizer, lr):

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        return optimizer

    def accuracy(self, y_predicted, test_l):

        _, class_predicted = torch.max(y_predicted.data, 1)
        return 100.0 * (class_predicted == test_l).sum() / y_predicted.size(0)

    def do_train(self, samples, perm, L, L_max, do_training=True):
        if do_training:
            self.train()
        else:
            self.eval()
        total_loss = 0
        total_acc = 0
        embedding_size = 0
        optimizer = self.update(selfoptions['learning_rate'])
        dropout_value = self.options['dropout_value']
        l2_regularization = self.options['l2_regularization']
        for i, data in enumerate(samples, 1):
            optimizer.zero_grad()

            train_data, train_label = data
            train_data = perm_data(train_data, perm)
            embedding_size = len(train_data)
            # moving tensors to adequate device
            train_data = train_data.to(args.device)
            train_labels = train_labels._requires_grad(False).to(args.device)

            output = self.forward(
                train_data, dropout_value, L, lmax).to(args.device)
            loss = self.loss(output, train_labels, l2_regularization)
            loss_train = loss.data.item()
            if do_training is True:
                loss.backward()
                optimizer.step()

            total_loss += loss
            acc = self.accuracy(output.cpu().detach(),
                                train_labels.data.cpu().detach())
            total_acc += acc

        if do_training is True:
            global_step += len(samples)
            options['learning_rate'] = options['learning_rate'] * \
                pow(options['decay'], float(global_step // embedding_size))
            optimizer = net.update_learning_rate(
                optimizer, options['learning_rate'])

        total_loss = total_loss / \
            len(samples.dataset) * self.options['batch_size']
        total_acc = total_acc / len(samples.dataset) * \
            self.options['batch_size']
        total_loss = torch.tensor(total_loss)
        return total_loss, total_acc

    def do_eval(self, samples, do_training=False):
        return self.do_train(samples, do_training=False)

    def get_model(self):
        return self.state_dict()
