'''Main method for validating that the 2-d local attention model works properly '''

# dependencies
import torch
import torch.nn as nn

# user defined modules
from DataLoader import TRAIN_LOADER as dataloader
from Attention import LocalAttention2d


# predetermined variable values to test
q_i = torch.tensor([[[[0.6200, 0.1500, 0.5400, 0.5100, 0.7800],
                      [0.6500, 0.7500, 0.5600, 0.9300, 0.7800],
                      [0.9400, 0.6300, 0.1100, 0.9600, 0.9700],
                      [0.0000, 0.3600, 0.8700, 0.3100, 0.7900]]],


                    [[[0.0200, 0.7500, 0.3000, 0.7400, 0.3100],
                      [0.0600, 0.3500, 0.0300, 0.2400, 0.5200],
                      [0.1700, 0.4500, 0.3200, 0.6000, 0.6500],
                      [0.7100, 0.7000, 0.3000, 0.4800, 0.3300]]]])


p_t = torch.tensor([[0.1682, 3.6395],
                    [3.3280, 1.1567]])

c_t = torch.tensor([[0.7000, 0.6100, 0.9500, 0.0100, 0.0700, 0.4700],
                    [0.2100, 0.8300, 0.6900, 0.7700, 0.2000, 0.5300]])


W_a = torch.tensor([[0.3000],
                    [0.0700],
                    [0.7700],
                    [0.3300],
                    [0.1000],
                    [0.3800]])

# initialize the model
model = LocalAttention2d(query_size=1,
                         context_size=6,
                         window=(3, 3))

# use a standard optimizer and loss to test just to see if weights update
OPTIMIZER = torch.optim.Adam(params=model.parameters(), lr=1e-4)
LOSS = nn.BCELoss()

# print what the weights are before anything is done
model.W_a.weight.data = W_a
print(model.W_a.weight)

# update the weights once and see if the correct results appear
for i, (img, labels, lengths) in enumerate(dataloader):
    OPTIMIZER.zero_grad()
    output = model(q_i[0].unsqueeze(0), c_t[0].unsqueeze(0))
    print(output)
    error = LOSS(output, torch.randn_like(output))
    error.backward()
    OPTIMIZER.step()
    print(model.W_a.weight)
    break
