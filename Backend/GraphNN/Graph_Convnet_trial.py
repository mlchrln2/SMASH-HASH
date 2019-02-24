import GraphNN

# network parameters
D = train_data.shape[1]
CL1_F = 32
CL1_K = 25
CL2_F = 64
CL2_K = 25
FC1_F = 512
FC2_F = 10
net_parameters = [D, CL1_F, CL1_K, CL2_F, CL2_K, FC1_F, FC2_F]


# instantiate the object net of the class
net = Graph_ConvNet_LeNet5(net_parameters)
if torch.cuda.is_available():
    net.cuda()
print(net)


# # Weights
# L_net = list(net.parameters())


# # learning parameters
# learning_rate = 0.05
# dropout_value = 0.5
# l2_regularization = 5e-4
# batch_size = 100
# num_epochs = 20
# train_size = train_data.shape[0]
# nb_iter = int(num_epochs * train_size) // batch_size
# print('num_epochs=', num_epochs, ', train_size=',
#       train_size, ', nb_iter=', nb_iter)


# # Optimizer
# global_lr = learning_rate
# global_step = 0
# decay = 0.95
# decay_steps = train_size
# lr = learning_rate
# optimizer = net.update(lr)


# # loop over epochs
# indices = collections.deque()
# for epoch in range(num_epochs):  # loop over the dataset multiple times

#     # reshuffle
#     indices.extend(np.random.permutation(train_size))  # rand permutation

#     # reset time
#     t_start = time.time()

#     # extract batches
#     running_loss = 0.0
#     running_accuray = 0
#     running_total = 0
#     while len(indices) >= batch_size:

#         # extract batches
#         batch_idx = [indices.popleft() for i in range(batch_size)]
#         train_x, train_y = train_data[batch_idx, :], train_labels[batch_idx]
#         train_x = Variable(torch.FloatTensor(
#             train_x).type(dtypeFloat), requires_grad=False)
#         train_y = train_y.astype(np.int64)
#         train_y = torch.LongTensor(train_y).type(dtypeLong)
#         train_y = Variable(train_y, requires_grad=False)

#         # Forward
#         y = net.forward(train_x, dropout_value, L, lmax)
#         loss = net.loss(y, train_y, l2_regularization)
#         loss_train = loss.data[0]

#         # Accuracy
#         acc_train = net.evaluation(y, train_y.data)

#         # backward
#         loss.backward()

#         # Update
#         global_step += batch_size  # to update learning rate
#         optimizer.step()
#         optimizer.zero_grad()

#         # loss, accuracy
#         running_loss += loss_train
#         running_accuray += acc_train
#         running_total += 1

#         # print
#         if not running_total % 100:  # print every x mini-batches
#             print('epoch= %d, i= %4d, loss(batch)= %.4f, accuray(batch)= %.2f' % (
#                 epoch + 1, running_total, loss_train, acc_train))

#     # print
#     t_stop = time.time() - t_start
#     print('epoch= %d, loss(train)= %.3f, accuracy(train)= %.3f, time= %.3f, lr= %.5f' %
#           (epoch + 1, running_loss / running_total, running_accuray / running_total, t_stop, lr))

#     # update learning rate
#     lr = global_lr * pow(decay, float(global_step // decay_steps))
#     optimizer = net.update_learning_rate(optimizer, lr)

#     # Test set
#     running_accuray_test = 0
#     running_total_test = 0
#     indices_test = collections.deque()
#     indices_test.extend(range(test_data.shape[0]))
#     t_start_test = time.time()
#     while len(indices_test) >= batch_size:
#         batch_idx_test = [indices_test.popleft() for i in range(batch_size)]
#         test_x, test_y = test_data[batch_idx_test,
#                                    :], test_labels[batch_idx_test]
#         test_x = Variable(torch.FloatTensor(test_x).type(
#             dtypeFloat), requires_grad=False)
#         y = net.forward(test_x, 0.0, L, lmax)
#         test_y = test_y.astype(np.int64)
#         test_y = torch.LongTensor(test_y).type(dtypeLong)
#         test_y = Variable(test_y, requires_grad=False)
#         acc_test = net.evaluation(y, test_y.data)
#         running_accuray_test += acc_test
#         running_total_test += 1
#     t_stop_test = time.time() - t_start_test
#     print('  accuracy(test) = %.3f %%, time= %.3f' %
#           (running_accuray_test / running_total_test, t_stop_test))
