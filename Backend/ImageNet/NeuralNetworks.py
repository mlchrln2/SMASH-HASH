'''This module defines the neural networks used to caption images from the COCO Dataset
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
    """Our Image caption model starts by creating feature maps from the images using the pretrained
    vgg16 model. Next unique words are mapped to unique indices that are used to create embeddings.
    In order to avoid overfitting on the captions drop out is implemented. The feature maps are
    treated as the queries for our 2d local attention mechanism with an initial context of zeros.
    This equates to taking the mean of all the queries within the window determined by the model.
    The output from the Attention layer and the embedding are concatenated together fed into an rnn
    with an initial hidden state of zeros. All future contexts for the Attention layer come from the
    next rnn hidden states. An additional drop out layer is used on the rnn outputs and then decoded
    to words in the vocabulary using a linear layer.

    Args: None

    Inputs: images, captions, lengths
        - **images** of shape (batch,3,x_dim,y_dim): tensor containing all
        of the rgb images in the current batch.
        - **captions** of shape (batch,longest_word): long tensor containing the captions for
        the current batch. The captions are padded with 1's (end word) if they are smaller than
        the largest caption in the batch.
        - **lengths** of shape (batch): long tensor containing the number of words in each caption
        for the current batch.

    Outputs: outputs
        - **outputs** of shape (batch,longest_word,vocab_size): tensor containing the captions
        predicted by the model.

    Attributes:
        - image_encoder of shape (channel_size): the feature maps extracted from the image data
        using the pretrained vgg16 model.
        - embedding of shape (vocab_size,embed_size): the embedding layer mapping indices to
        embeddings.
        - dropout of shape (drop): the dropout layer for the captions to prevent overfitting.
        - attention of shape (channel_size, hidden_size, window): the 2d local attention used to
        focus on a portion of an image and weigh the importance of each subregion.
        - rnn of shape (channel_size+embed_size,hidden_size): the recurrent layer used to learn
        sequential information obtained from the network.
        - decoder of shape (hidden_size,vocab_size): the linear layer used to decode the captions
        from the sequential information.
        - decoder_dropout of shape (drop): the dropout layer for the sequential information used to
        prevent overfitting.
    """

    def __init__(self):
        super(Image2Caption, self).__init__()
        self.channel_size = OPTIONS['channel_size']
        self.embed_size = OPTIONS['embed_size']
        self.vocab_size = OPTIONS['vocab_size']
        self.max_len = OPTIONS['max_len']
        self.learning_rate = OPTIONS['learning_rate']
        self.hidden_size = OPTIONS['hidden_size']
        self.window = OPTIONS['window']
        self.drop = OPTIONS['drop']
        self.beam_size = OPTIONS['beam_size']
        self.image_encoder = ImageEncoder(channel_size=self.channel_size)
        self.embedding = nn.Embedding(num_embeddings=self.vocab_size,
                                      embedding_dim=self.embed_size)
        self.dropout = nn.Dropout(self.drop)
        self.attention = LocalAttention2d(query_size=self.channel_size,
                                          context_size=self.hidden_size,
                                          window=self.window)
        self.rnn = nn.GRUCell(input_size=self.channel_size + self.embed_size,
                              hidden_size=self.hidden_size)
        self.decoder = nn.Linear(in_features=self.hidden_size,
                                 out_features=self.vocab_size)
        self.decoder_dropout = nn.Dropout(self.drop)
        self.optimizer = torch.optim.Adam(params=self.parameters(),
                                          lr=self.learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(1)

    def forward(self, images, captions, lengths):
        features = self.image_encoder(images)
        captions = self.embedding(captions)
        captions = self.dropout(captions)
        captions = captions.transpose(0, 1)
        h_n = torch.zeros(features.size(0), self.hidden_size)
        outputs = torch.zeros(features.size(0),
                              captions.size(0), self.vocab_size)
        batch = lengths.size(0)
        batch_item = lengths[batch - 1].item()
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
            outputs[:batch, i] = self.decoder(out)
        return outputs

    def infer(self, image):
        """The inference method behaves in a similar fashion to the forward method except that
        the weights are returned from the attention layer and the next caption to use is the most
        probable index from the previous pass. This requires that the initial hidden state for the
        rnn to be zeros and for the first caption to be 0 (start word).

        Inputs: image
            - **image** of shape (batch,3,x_dim,y_dim): tensor containing all
            of the rgb images in the current batch.

        Outputs: outputs
            - **outputs** of shape (batch,max_len): tensor containing the captions
            predicted by the model.
            - **summaries** of shape (batch,hidden_size): the last rnn output from the network
            used as an embedding for the graph network later on.
            - **alphas** of shape (batch,max_len,x_dim,y_dim): the weights obtained from the
            attention layer upsampled to the dimension of the input image.
        """
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
            feats, alpha = self.attention.infer(features, h_n)
            cap = self.embedding(idxs)
            z_inputs = torch.cat((feats, cap), 1)
            h_n = self.rnn(z_inputs, h_n)
            output = self.decoder(h_n)
            idxs = torch.argmax(output, 1)
            words[num_words] = idxs
            alphas[:, num_words] = alpha
        alphas = alphas[:, 0].unsqueeze(
            1) if num_words == 0 else alphas[:, :num_words]
        alphas = F.interpolate(alphas, size=(
            image.size(2), image.size(3)), mode='nearest')
        words = words[0].unsqueeze(0) if num_words == 0 else words[:num_words]
        summaries = h_n
        return words, summaries, alphas

    def infer_beam(self, image):
        """The beam search inference method behaves in a similar fashion to the previous
        inference method except that the model explores captions that maximize the log probability.
        Multiple (beam_size) paths are explored in this inference method instead of settling
        with the greedy result. This method does not gurantee finding the most probable
        caption but with a sufficiently well trained model it tends towards the accepted caption.
        In order to improve the space efficiency of the algorithm, the method is implemented using
        2 stacks. The results are collected in words, out_hn, and out_alphas.

        Inputs: image
            - **image** of shape (1,3,x_dim,y_dim): tensor containing all
            of the rgb images in the current batch.

        Outputs: outputs
            - **outputs** of shape (1,max_len,beam_size): tensor containing the captions
            predicted by the model for each beam_size path.
            - **summaries** of shape (1,hidden_size,beam_size): the last rnn output from the network
            used as an embedding for the graph network later on for each beam_size path.
            - **alphas** of shape (1,max_len,beam_size,x_dim,y_dim): the weights obtained from the
            attention layer upsampled to the dimension of the input image for each beam_size path.
        """
        # encode image
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
        captions, probs, h_n, alpha = self.model_function(features, idxs, h_n)
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
                curr_captions, curr_probs, curr_hn, curr_alpha = self.model_function(
                    features, cap.unsqueeze(0), h_n[:, i])
                # add the log probabilities of the current path and branches to
                # obtain the most likely next path
                curr_probs = prob + curr_probs
                # for every branch get the caption and the probability of
                # taking that branch
                for curr_c, curr_p in zip(curr_captions, curr_probs):
                    # As long as the current branch beats the least likely item on the stack
                    # (the top) move the item to the top of the garbage
                    while not stack and curr_p > float(stack[-1][2]):
                        garbage.append(stack.pop())
                    # As long as stack is not full and the current branch loses to the most
                    # likely item on the garbage the top) move the item back to the top of the
                    # stack
                    while len(stack) != self.beam_size \
                            and not garbage and curr_p <= float(garbage[-1][2]):
                        stack.append(garbage.pop())
                    if len(stack) != self.beam_size:
                        # as long as an object was removed from the stack push the new object in
                        # question to the top of the stack
                        stack.append([i, curr_c, curr_p, curr_hn, curr_alpha])
                        if len(stack) != self.beam_size and not garbage:
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

    def model_function(self, features, idxs, h_n):
        """A helper function for the beam search inference method.

        Inputs: features, idxs, h_n
            - **features** of shape (1,512,num_x_queries,num_y_queries): tensor containing all of
            the feature maps.
            - **idxs** of shape(1): tensor containing the current word in consideration.
            - **h_n** of shape(1,hidden_size): tensor containing the current hidden state in
            consideration.

        Outputs: captions, probs, h_n, alpha
            - **captions** of shape (1): tensor containing the captions
            predicted by the model for the current path.
            - **probs** of shape (1): the log probability of taking the
            current path.
            - **alpha** of shape (1,max_len,x_dim,y_dim): the weights obtained from the
            attention layer for the current path.
        """
        feats, alpha = self.attention.infer(features, h_n)
        cap = self.embedding(idxs)
        z_inputs = torch.cat((feats, cap), 1)
        h_n = self.rnn(z_inputs, h_n)
        output = self.decoder(h_n)
        probs, captions = torch.topk(input=output, k=self.beam_size, dim=1)
        captions = captions.squeeze(0).detach()
        probs = F.log_softmax(probs.squeeze(0).detach(), dim=0)
        return captions, probs, h_n, alpha


class ImageEncoder(nn.Module):
    """The image encoder creates feature maps from the images using the pretrained vgg16 model
    without the last layer.

    Args:
        channel_size (int): the size of the image channel. For black and white images it should be 1
        for rgb the size should be 3, and for rgba images it should be 4.

    Inputs: x
        - **x** of shape (batch,3,x_dim,y_dim): tensor containing all of the images of the
        current batch.

    Outputs: out
        - **out** of shape (batch,512,num_x_queries,num_y_queries): tensor containing all of the
        feature maps of the current batch.

    Attributes:
        - pretrained_net of shape (3,channel_size): the feature maps extracted from the image data
        using the pretrained vgg16 model.
        - batch_norm of shape (channel_size): the 2d batch normalization layer that helps the model
        converge with larger learning rates.
    """
    def __init__(self, channel_size):
        super(ImageEncoder, self).__init__()
        self.channel_size = channel_size
        pretrained_net = models.vgg16(pretrained=True).features
        modules = list(pretrained_net.children())[:29]
        self.pretrained_net = nn.Sequential(*modules)

        self.batch_norm = nn.BatchNorm2d(num_features=modules[-1].in_channels,
                                         momentum=0.01)

    def forward(self, x):
        with torch.no_grad():
            x = self.pretrained_net(x)
        out = self.batch_norm(x)
        return out

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
