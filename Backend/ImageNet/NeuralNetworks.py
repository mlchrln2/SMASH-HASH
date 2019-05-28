'''This module defines the image captioning neural networks for the COCO Dataset
and the MNIST Dataset'''

# dependencies
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from Attention import LocalAttention2d

# user defined modules
from HyperParameters import OPTIONS


class Image2Caption(nn.Module):
    """Our Image captioning model starts by creating feature maps using the pretrained vgg16 layers
    and word embeddings using the default embedding layer. The vgg16 model takes as input rgb
    images while the embedding layer takes as input word captions pre-mapped to unique indices.
    The feature maps serve as the input queries to the 2d local attention model while the
    captions concatenated to the attention layer outputs serve as the inputs to the rnn layer.
    The context vectors for the 2d local attention model are the hidden states of the rnn
    layer where the first context vector is the zero vector. The significance of using the zero
    vector as the first context is that the attention mechanism treats every query within the window
    equally - effectively taking the mean. The rnn hidden states are also fed into a linear layer
    that decodes the hidden states into word captions in the vocabulary. Lastly, in order to prevent
    overfitting drop out is implemented after the embedding and rnn stages.

    Args:
        learning_rate (float): the rate used to update the gradient.
        channel_size (int): the size of the feature maps.
        momentum (float): the momentum for batch normalization
        vocab_size (int): the size of the vocabulary.
        embed_size (int): the embeding size to use for a word caption.
        drop (float): the drop out rate.
        hidden_size (int): the output size of the rnn layer.
        window (int): the window used for the 2d local attention model.
        max_len (int): the maximum length of an inferred caption.
        beam_size (int): the number of paths to check during the beam search.

    Inputs: images, captions, lengths
        - **images** of shape (batch,img_channels,x_dim,y_dim): tensor containing all
        of the images in the current batch.
        - **captions** of shape (batch,cap_size): long tensor containing the captions for
        the current batch. The captions are padded with 1's (end word) if they are smaller than
        the longest caption in the current batch (cap_size).
        - **lengths** of shape (batch): long tensor containing the number of words in each caption
        for the current batch.

    Outputs: words
        - **words** of shape (batch,cap_size,vocab_size): tensor containing the one-hot encoded
        captions predicted by the model.

    Attributes: image_encoder, embedding, dropout, attention, rnn, decoder, decoder_dropout
        - **image_encoder** with parameters (channel_size, momentum): the feature maps extracted
        from the image data using the pretrained vgg16 model.
        - **embedding** with parameters (vocab_size,embed_size): the embedding layer mapping indices
        to embeddings.
        - **dropout** with parameter (drop): the dropout layer for the captions to prevent
        overfitting.
        - **attention** with parameters (channel_size, hidden_size, window): the 2d local attention
        used to attend on a portion of an image and weigh the importance of each subregion.
        - **rnn** with parameters (channel_size+embed_size,hidden_size): the recurrent layer used to
        learn sequential information obtained from the network.
        - **decoder** with parameters (hidden_size,vocab_size): the linear layer used to decode the
        captions from the rnn output.
        - **decoder_dropout** with parameter (drop): the dropout layer for the rnn outputs used to
        prevent overfitting.
    """

    def __init__(self):
        super(Image2Caption, self).__init__()
        self.learning_rate = OPTIONS['learning_rate']
        self.channel_size = OPTIONS['channel_size']
        self.momentum = OPTIONS['momentum']
        self.vocab_size = OPTIONS['vocab_size']
        self.embed_size = OPTIONS['embed_size']
        self.drop = OPTIONS['drop']
        self.hidden_size = OPTIONS['hidden_size']
        self.window = OPTIONS['window']
        self.max_len = OPTIONS['max_len']
        self.beam_size = OPTIONS['beam_size']
        self.device = OPTIONS['device']
        self.image_encoder = ImageEncoder(channel_size=self.channel_size,
                                          momentum=self.momentum,
                                          device=self.device)
        self.embedding = nn.Embedding(num_embeddings=self.vocab_size,
                                      embedding_dim=self.embed_size).to(self.device)
        self.dropout = nn.Dropout(self.drop)
        self.attention = LocalAttention2d(query_size=self.channel_size,
                                          context_size=self.hidden_size,
                                          window=self.window,
                                          device=self.device)
        self.rnn = nn.GRUCell(input_size=self.channel_size + self.embed_size,
                              hidden_size=self.hidden_size).to(self.device)
        self.decoder = nn.Linear(in_features=self.hidden_size,
                                 out_features=self.vocab_size).to(self.device)
        self.decoder_dropout = nn.Dropout(self.drop)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.softmax = nn.Softmax(1)
        self.optimizer = torch.optim.Adam(params=self.parameters(),
                                          lr=self.learning_rate)

    def forward(self, images, captions, lengths):
        #features = images
        features = self.image_encoder(images)
        captions = self.embedding(captions)
        captions = self.dropout(captions)
        captions = captions.transpose(0, 1)
        h_n = torch.zeros(features.size(0), self.hidden_size, device=self.device)
        words = torch.zeros(features.size(0),
                            captions.size(0),
                            self.vocab_size,
                            device=self.device)
        batch = lengths.size(0)
        batch_item = lengths[batch - 1].detach().item()
        for i, cap in enumerate(captions):
            while i + 1 > batch_item and batch > 1:
                batch -= 1
                batch_item = lengths[batch - 1].item()
            cap = cap[:batch]
            h_n = h_n[:batch]
            feats = self.attention(features[:batch], h_n)
            z_inputs = torch.cat((feats, cap), 1)
            h_n = self.rnn(z_inputs, h_n)
            out = self.decoder_dropout(h_n)
            words[:batch, i] = self.decoder(out)
        return words

    def model_helper(self, features, h_n, idx, k):
        """A helper function for the beam search inference method.

        Args:
            k (int): the number of top values to select from the max

        Inputs: features, h_n, idx, k
            - **features** of shape (1,channel_size,num_x_queries,num_y_queries): tensor containing
            all of the feature maps.
            - **h_n** of shape(1,hidden_size): tensor containing the current hidden state in
            consideration.
            - **idx** of shape(1,1): tensor containing the current word in consideration.

        Outputs: captions, probs, h_n, alpha
            - **captions** of shape (beam_size): tensor containing the top-k captions predicted by
            the model for the current path.
            - **probs** of shape (beam_size): tensor containing the top-k log probabilities for
            the current path.
            - **h_n** of shape (1,hidden_size): tensor containing the next hidden state in
            consideration.
            - **alpha** of shape (1,num_x_queries,num_y_queries): tensor containing the weights
            obtained from the attention layer for the current path.
        """
        cap = self.embedding(idx)
        cap = self.dropout(cap)
        feats, alpha = self.attention.infer(features, h_n)
        z_inputs = torch.cat((feats, cap), 1)
        h_n = self.rnn(z_inputs, h_n)
        output = self.decoder_dropout(h_n)
        output = self.decoder(output)
        probs, captions = torch.topk(input=output, k=k, dim=1)
        captions = captions.squeeze(0).detach()
        probs = F.log_softmax(probs.squeeze(0).detach(), dim=0)
        return captions, probs, h_n, alpha

    def infer_greedy(self, image):
        """Inference method for validating the model predictions using a greedy algorithm. This
        method requires the first word to be 0 (start word) and all subsequent words to be the most
        likely words at that time step. The process terminates either when the model outputs a 1
        (end word) or the time step has reached the max_len. This method does not gurantee finding
        the most probable caption and thus requires a very well trained model to achieve good
        results.

        Inputs: image
            - **image** of shape (1,img_channels,x_dim,y_dim): tensor containing the input image.

        Outputs: words, summaries, alphas
            - **words** of shape (1,max_len): tensor containing the caption predicted by the model.
            - **summaries** of shape (1,hidden_size): tensor containing the last rnn output from
            the network. This is used as an embedding for the graph network later on.
            - **alphas** of shape (1,max_len,x_dim,y_dim): tensor containing the weights obtained
            from the attention layer upsampled to the dimension of the input image.
        """
        #features = image
        features = self.image_encoder(image)
        idxs = torch.zeros(features.size(0), dtype=torch.long)
        num_words = 0
        words = torch.zeros(self.max_len, dtype=torch.long)
        alphas = torch.zeros(features.size(0), self.max_len,
                             features.size(2), features.size(3))
        h_n = torch.zeros(features.size(0), self.hidden_size)
        for num_words in range(self.max_len):
            if idxs.item() == 1:
                break
            idxs, _, h_n, alpha = self.model_helper(features, h_n, idxs, 1)
            words[num_words] = idxs
            alphas[:, num_words] = alpha
        alphas = alphas[:, 0].unsqueeze(
            1) if num_words == 0 else alphas[:, :num_words]
        words = words[0].unsqueeze(0) if num_words == 0 else words[
            :num_words].unsqueeze(0)
        summaries = h_n
        return words, summaries, alphas

    def infer_beam_search(self, image):
        """Inference method for validating the model predictions using a beam search algorithm.
        This method requires the first word to be 0 (start word) and all subsequent words to be the
        top-k most probable captions at that time step where k is the beam_size. The process
        terminates either when the model outputs a 1 for every path or if the time step has reached
        the max_len. In order to improve the space efficiency of the algorithm, the method is
        implemented using 2 stacks labeled stack and garbage. This method does not gurantee finding
        the most probable caption but can achieve better results than the standard greedy method.

        Inputs: image
            - **image** of shape (1,img_channels,x_dim,y_dim): tensor containing the input image.

        Outputs: words, summaries, alphas
            - **words** of shape (beam_size,max_len): tensors containing the captions from each path
            predicted by the model.
            - **summaries** of shape (beam_size,hidden_size): tensors containing the last rnn outputs
            from each path predicted by the model. These are used as embeddings for the graph
            network later on.
            - **alphas** of shape (beam_size,max_len,x_dim,y_dim): tensors containing the weights
            obtained from the attention layer for each path upsampled to the dimension of the input
            image.
        """
        # encode image
        #features = image
        features = self.image_encoder(image)
        # the first idx/word is 0 (start_word)
        idxs = torch.zeros(features.size(0), dtype=torch.long)
        # there are currently no words stored
        num_words = 0
        # initialize blank containers for the output paths and weights
        paths = torch.ones(self.max_len, self.beam_size, dtype=torch.long)
        alphas = torch.zeros(self.max_len, self.beam_size,
                             features.size(2), features.size(3))
        # the first hidden state is None
        h_n = torch.zeros(features.size(0), self.hidden_size)

        # starting caption, probability, output hidden state, and weights
        captions, probs, h_n, alpha = self.model_helper(
            features, h_n, idxs, self.beam_size)
        h_n = h_n.repeat(1, self.beam_size, 1)
        alpha = alpha.repeat(self.beam_size, 1, 1)
        # store the starting weights and captions
        alphas[num_words] = alpha
        paths[num_words] = captions
        # start with a blank stack and garbage
        stack = []
        garbage = []
        # continue on to the next word
        num_words += 1
        words = []
        out_hn = []
        out_alphas = []
        done_flag = False
        # repeat this procedure until you reach the max length or every path
        # has hit the end_word
        while num_words < self.max_len:
            # pair the caption with its respective confidence for the current
            # iteration
            for i, (cap, prob) in enumerate(zip(captions, probs)):
                # a path is terminated if it ended with the end_word
                if cap == 1 and len(words) != self.beam_size:
                    # append the terminated path characteristics to the word
                    # list
                    words.append(paths[:num_words, i])
                    out_alphas.append(alphas[:num_words, i])
                    out_hn.append(h_n[:, i])
                    if len(words) == self.beam_size:
                        done_flag = True
                        # if beam_size words have been stored then finish the
                        # program
                        break
                    # skip this caption when determining future paths
                    continue
                # obtain the next beam_size branches from the current
                # path/hidden state
                curr_captions, curr_probs, curr_hn, curr_alpha = self.model_helper(
                    features, h_n[:, i], cap.unsqueeze(0), self.beam_size)
                # add the log probabilities of the current path and branches to
                # obtain the most likely next path
                curr_probs = prob + curr_probs
                # for every branch get the caption and the probability of
                # taking that branch
                for curr_c, curr_p in zip(curr_captions, curr_probs):
                    # As long as the current branch beats the least likely item on the stack
                    # (the top) move the item to the top of the garbage
                    while stack and curr_p > float(stack[-1][2]):
                        garbage.append(stack.pop())
                    # As long as stack is not full and the current branch loses to the most
                    # likely item on the garbage the top) move the item back to the top of the
                    # stack
                    while len(stack) != self.beam_size \
                            and garbage and curr_p <= float(garbage[-1][2]):
                        stack.append(garbage.pop())
                    if len(stack) != self.beam_size:
                        # as long as an object was removed from the stack push the new object in
                        # question to the top of the stack
                        stack.append([i, curr_c, curr_p, curr_hn, curr_alpha])
                        if len(stack) != self.beam_size and garbage:
                            # as long as the stack is not full and the garbage is not empty
                            # move the top item from the garbage to the top of
                            # the stack
                            stack.append(garbage.pop())
                        else:
                            # otherwise clear the garbage
                            garbage = []
                    else:
                        # if nothing was removed from the stack then the next less likely branches
                        # will not remove anything from the stack either so
                        # ignore them
                        garbage = []
                        break
            if done_flag:
                break
            # unpack the next branch indices, captions, probabilities, hidden
            # states, and weights
            indices = torch.tensor([word[0] for word in stack]).long()
            captions = torch.tensor([word[1] for word in stack]).long()
            probs = torch.tensor([word[2] for word in stack])
            h_n = torch.stack([word[3] for word in stack], dim=1)
            alpha = torch.stack([word[4] for word in stack], dim=1)
            # store the weights for this iteration
            alphas = alphas[:, indices]
            alphas[num_words] = alpha.squeeze(0)
            # update the maximal paths based on the previous paths
            paths = paths[:, indices]
            # add the next word in the sentences
            paths[num_words] = captions
            # increment the number of words
            num_words += 1
            # clear the stack for the next iteration
            stack = []
        summaries = [hni.squeeze(0) for hni in out_hn]
        alphas = [F.interpolate(alp.unsqueeze(0), size=(image.size(
            2), image.size(3)), mode='nearest') for alp in out_alphas]
        return words, summaries, alphas


class ImageEncoder(nn.Module):
    """The image encoder creates feature maps from the images using the pretrained vgg16 model
    up until the second to last module.

    Args:
        channel_size (int): the size of the output channel.

    Inputs: image
        - **image** of shape (batch,img_channels,x_dim,y_dim): tensor containing all of the images
        of the current batch.

    Outputs: image_embedding
        - **image_embedding** of shape (batch,channel_size,num_x_queries,num_y_queries): tensor
        containing all of the feature maps of the current batch.

    Attributes: pretrained_net, lin_map, batch_norm
        - **pretrained_net** with parameters (img_channels,in_channels): the feature maps extracted
        from the image data using the pretrained vgg16 model.
        - **lin_map** with parameters (in_channels, channel_size): the linear layer that connects
        the encoder model to the decoder model if the in_channels is not equal to the channel_size.
        - **batch_norm** with parameters (channel_size,momentum): the 2d batch normalization layer.
    """

    def __init__(self, channel_size, momentum, device):
        super(ImageEncoder, self).__init__()
        self.channel_size = channel_size
        self.momentum = momentum
        self.device = device
        pretrained_net = models.vgg16(pretrained=True).features
        modules = list(pretrained_net.children())[:29]
        self.in_channels = modules[-1].in_channels
        self.pretrained_net = nn.Sequential(*modules)
        if self.in_channels != self.channel_size:
            self.lin_map = nn.Linear(in_features=self.in_channels,
                                     out_features=self.channel_size).to(self.device)
        self.batch_norm = nn.BatchNorm2d(num_features=modules[-1].in_channels,
                                         momentum=self.momentum).to(self.device)

    def forward(self, image):
        with torch.no_grad():
            image_embedding = self.pretrained_net(image).to(self.device)
        if self.in_channels != self.channel_size:
            image_embedding = self.lin_map(image_embedding)
        image_embedding = self.batch_norm(image_embedding)
        return image_embedding


class MNISTEncoder(nn.Module):
    '''Sample encoder network testing on MNIST'''

    def __init__(self):
        super(MNISTEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(16, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        return x


class MNISTDecoder(nn.Module):
    '''Sample decoder network testing on MNIST'''

    def __init__(self):
        super(MNISTDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.decoder(x)
        return x


class MNISTAutoencoder(nn.Module):
    '''Sample autoencoder network testing on MNIST'''

    def __init__(self):
        super(MNISTAutoencoder, self).__init__()
        self.learning_rate = OPTIONS['learning_rate']
        self.encoder = MNISTEncoder()
        self.decoder = MNISTDecoder()
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate,
                                          weight_decay=1e-5)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
