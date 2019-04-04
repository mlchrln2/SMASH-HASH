'''This module outlines all of the attention mechanisms explored in this project'''

# dependencies
import torch
import torch.nn as nn


class Attention(nn.Module):
    """General self-attention model that scores the context (c_t) at time t for every query i then
    weighs the queries using the alignments generated from that score.

    Args:
            query_size (int): the size of a single query.
            context_size (int): the size of the context vectors.

    Inputs: q_i, c_t
            - **q_i** of shape (batch,query_size,num_queries): tensor containing all of the queries
            with variable size num_queries.
            - **c_t** of shape (batch,context_size): tensor containing the context vectors.
            Defaults to zero if not provided.

    Outputs: s_t
            - **s_t** of shape (batch,query_size): tensor containing the weighted sum of queries.

    Attributes:
            - w_a of shape (query_size,context_size): the learnable weight from the attention layer.
    """

    def __init__(self, query_size, context_size):
        super(Attention, self).__init__()
        self.q_size = query_size
        self.c_size = context_size
        self.w_a = nn.Linear(in_features=self.q_size,
                             out_features=self.c_size,
                             bias=False)
        self.softmax = nn.Softmax(1)

    def forward(self, q_i, c_t):
        if c_t is None:
            c_t = q_i.new_zeros(q_i.size(0), self.c_size)
        q_i = q_i.transpose(1, 2)
        w_attn = self.align(q_i, c_t)
        s_t = torch.bmm(w_attn.transpose(1, 2), q_i).squeeze(1)
        return s_t

    def infer(self, q_i, c_t):
        '''Method used for inference when testing. This method unlike the forward method
        returns the weights in addition to the summary vector.

        Inputs: q_i, c_t
            - **q_i** of shape (batch,query_size,num_queries): tensor containing all of the queries
            with variable size num_queries.
            - **c_t** of shape (batch,context_size): tensor containing the context vectors.
            Defaults to zero if not provided.

        Outputs: s_t, w_attn
            - **s_t** of shape (batch,query_size): tensor containing the weighted sum of queries.
            - **w_attn** of shape (batch,num_queries): tensor containing the attention weights.
        '''

        if c_t is None:
            c_t = q_i.new_zeros(q_i.size(0), self.c_size)
        q_i = q_i.transpose(1, 2)
        w_attn = self.align(q_i, c_t)
        s_t = torch.bmm(w_attn.transpose(1, 2), q_i).squeeze(1)
        return s_t, w_attn.squeeze(2)

    def align(self, q_i, c_t):
        '''The alignment method converts scores to probabilities and yields
        the attention weights.'''
        w_attn = self.softmax(self.score(q_i, c_t))
        return w_attn

    def score(self, q_i, c_t):
        '''General attention scoring as outlined in Luong et al. Scoring is a metric for
        deterimining how relevant certain queries are to the given context vector.'''
        return torch.bmm(self.w_a(q_i), c_t.unsqueeze(2))


class LocalAttention1d(nn.Module):
    """The Luong (Local) Attention model applies an attention mechanism to a subset of the
    queries by learning the predictive alignment position p_t for a given context vector c_t.
    The predictive alignment position p_t provides the centroid of the queries on which to attend
    over. The queries surrounding the alignment position p_t within the window s_i in (-s_win.s_win)
    are weighed as in general self-attention. In order to promote the idea of localized attention,
    a gaussian window is applied around the alignment position with standard deviation s_win/2.

    Args:
        query_size (int): the size of a single query.
        context_size (int): the size of the context vectors.
        align_size (int): the size of the predictive alignment dimension.
        window (int): the query window around the centroid. Defaults to 2 if not provided.

    Inputs: q_i, c_t
        - **q_i** of shape (batch,query_size,num_queries): tensor containing all of the queries
        with variable size num_queries.
        - **c_t** of shape (batch,context_size): tensor containing the context vectors.
        Defaults to zero if not provided.

    Outputs: s_t
        - **s_t** of shape (batch,query_size): tensor containing the weighted sum of queries.

    Attributes:
        - w_a of shape (query_size,context_size): the learnable weight from the attention layer.
        - w_p of shape (context_size,align_size): the learnable weight for projecting the
        context vector into the prediction space.
        - v_p of shape (align_size, 1): the learnable weight for predicting the alignment location.

    Notes:
        Nan padding is used to prevent out of bounds indexing. The basic idea is that if an index
        in the window lies outside of the boundaries then nans are returned. Nans are used to
        catalog undesired behaviour, changed to zeros when updating weights and changed to -inf
        when computing softmax probabilities.
    """

    def __init__(self, query_size, context_size, align_size, window=2):
        super(LocalAttention1d, self).__init__()
        self.q_size = query_size
        self.c_size = context_size
        self.p_size = align_size
        self.s_win = window
        self.w_a = nn.Linear(in_features=self.q_size,
                             out_features=self.c_size,
                             bias=False)
        self.w_p = nn.Linear(in_features=self.c_size,
                             out_features=self.p_size,
                             bias=False)
        self.v_p = nn.Linear(in_features=self.p_size,
                             out_features=1,
                             bias=False)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(2)

    def forward(self, q_i, c_t):
        if c_t is None:
            c_t = q_i.new_zeros(q_i.size(0), self.c_size)
        q_i = nn.ConstantPad1d((1, 0), float('nan'))(q_i)
        size = q_i.size(2)
        p_t = self.predictive_alignment(q_i.size(2) - 2, c_t)
        s_i = torch.fmod(torch.clamp(torch.stack([torch.arange(p_batch - (self.s_win // 2) + 1,
                                                               p_batch + (self.s_win + 1) // 2 + 1)
                                                  for p_batch in torch.round(p_t).long()]),
                                     min=0, max=size), size)
        s_exp = 2 * torch.pow((torch.clamp(s_i - 1, min=0).float() -
                               p_t.unsqueeze(-1)) / (self.s_win // 2), 2)
        q_i = q_i.transpose(1, 2)
        q_i = torch.stack([batch[idx] for batch, idx in zip(q_i, s_i)])
        w_attn = self.align(q_i, c_t) * torch.exp(-s_exp.unsqueeze(2))
        s_t = torch.bmm(w_attn.transpose(1, 2), q_i).squeeze(1)
        return s_t

    def infer(self, q_i, c_t):
        '''Method used for inference when testing. This method unlike the forward method
        returns the weights in addition to the summary vector.

        Inputs: q_i, c_t
            - **q_i** of shape (batch,query_size,num_queries): tensor containing all of the queries
            with variable size num_queries.
            - **c_t** of shape (batch,context_size): tensor containing the context vectors.
            Defaults to zero if not provided.

        Outputs: s_t
            - **s_t** of shape (batch,query_size): tensor containing the weighted sum of queries.
            - **w_attn** of shape (batch,num_queries): tensor containing the attention weights.
        '''
        if c_t is None:
            c_t = q_i.new_zeros(q_i.size(0), self.c_size)
        q_i = nn.ConstantPad1d((1, 0), float('nan'))(q_i)
        size = q_i.size(2)
        p_t = self.predictive_alignment(q_i.size(2) - 2, c_t)
        s_i = torch.fmod(torch.clamp(torch.stack([torch.arange(p_batch - (self.s_win // 2) + 1,
                                                               p_batch + (self.s_win + 1) // 2 + 1)
                                                  for p_batch in torch.round(p_t).long()]),
                                     min=0, max=size), size)
        s_exp = 2 * torch.pow((torch.clamp(s_i - 1, min=0).float() -
                               p_t.unsqueeze(-1)) / (self.s_win // 2), 2)
        q_i = q_i.transpose(1, 2)
        q_i = torch.stack([batch[idx] for batch, idx in zip(q_i, s_i)])
        w_attn = self.align(q_i, c_t) * torch.exp(-s_exp.unsqueeze(2))
        s_t = torch.bmm(w_attn.transpose(1, 2), q_i).squeeze(1)

        w_attn = w_attn.view(w_attn.size(0), self.s_win)
        s_i = s_i.view(s_i.size(0), self.s_win)
        w_attn = self.alpha_frames(size, w_attn, s_i)
        return s_t, w_attn

    def predictive_alignment(self, s_size, c_t):
        '''Predict the window centroid given the total number of queries and the context vector'''
        loc = self.sigmoid(self.v_p(self.tanh(self.w_p(c_t))))
        p_t = loc[:, 0] * s_size
        return p_t

    def align(self, q_i, c_t):
        '''The alignment method converts scores to probabilities and yields the attention weights.
        A trick is used to ensure that if the area around the center goes out of bounds,
        it does not contribute to the learnable weights and the attention weights themselves.'''
        nan_loc = torch.isnan(q_i[..., 0])
        q_i[nan_loc] = 0
        a_t = self.score(q_i, c_t)
        a_t[nan_loc] = -float('inf')
        w_attn = self.softmax(a_t)
        return w_attn

    def score(self, q_i, c_t):
        '''General attention scoring as outlined in Luong et al. Scoring is a metric for
        deterimining how relevant certain queries are to the given context vector.'''
        a_t = torch.bmm(self.w_a(q_i), c_t.unsqueeze(2))
        return a_t

    def alpha_frames(self, size, w_attn, loc):
        '''Method for returning the weights in the shape of the original input.
        Args:
            size (int): the number of queries.

        Inputs: w_attn, loc
            - **w_attn** of shape (batch,s_win): tensor containing the attention weights.
            - **loc** of shape(batch,s_win): tensor containing the locations in the nan-padded
            queries for every weight.

        Outputs: alpha_matrix
            - **alpha_matrix** of shape (batch,num_queries): tensor containing the weights in the
            shape of the original input.
        '''
        alpha_matrix = torch.zeros(w_attn.size(0), size)
        alpha_matrix[:, loc] = w_attn
        return alpha_matrix[:, 1:]


class LocalAttention2d(nn.Module):
    """Our 2-dimensional Local Attention model follows from Luong (Local) Attention except that now
    two predictive alignment positions p_t = (p_x_t, p_y_t) are learned for a given context c_t.
    The predictive alignment positions give the centroid of the image queries on which to attend
    over. The queries surrounding the alignment pair p_t within the window r_i in (-r_win,r_win)
    and c_i in (-c_win,c_win)] are weighed as in general self-attention. In order to promote
    the idea of local attention, a 2-d gaussian window is applied around the alignment position
    with standard deviations (r_win/2,c_win/2). The difference in the structure of this model
    when compared to Luong Attention is that the gaussian window appears as a shift in the score.
    Also, only two learnable parameters are trained instead of three removing the need for an
    arbitrary and non-intuitive alignment size.

    Args:
        query_size (int): the size of a single query.
        context_size (int): the size of the context vectors.
        window (int,int): the query window around the centroid. Defaults to (2,2)
        if not provided.

    Inputs: q_i, c_t
        - **q_i** of shape (batch,query_size,num_x_queries,num_y_queries): tensor containing all
        of the queries with variable size num_x_queries and num_y_queries.
        - **c_t** of shape (batch,context_size): tensor containing the context vector.
        Defaults to zero if not provided.

    Outputs: s_t
        - **s_t** of shape (batch,query_size): tensor containing the weighted sum of queries.

    Attributes:
        - w_a of shape (query_size,context_size): the learnable weight from the attention layer.
        - w_p of shape (context_size,2): the learnable weight for the alignment position prediction.

    Notes:
        Nan padding is used to prevent out of bounds indexing. The basic idea is that if an index
        in the window lies outside of the boundaries then nans are returned. Nans are used to
        catalog undesired behaviour, changed to zeros when updating weights and changed to -inf
        when computing softmax probabilities.
    """

    def __init__(self, query_size, context_size, window=(2, 2)):
        super(LocalAttention2d, self).__init__()
        self.q_size = query_size
        self.c_size = context_size
        self.r_win = window[0]
        self.c_win = window[1]
        self.w_a = nn.Linear(in_features=self.q_size,
                             out_features=self.c_size,
                             bias=False)
        self.w_p = nn.Linear(in_features=self.c_size,
                             out_features=2,
                             bias=False)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(1)

    def forward(self, q_i, c_t):
        if c_t is None:
            c_t = q_i.new_zeros(q_i.size(0), self.c_size)
        q_i = nn.ConstantPad2d((1, 0, 1, 0), float('nan'))(q_i)
        rows, cols = q_i.size(2), q_i.size(3)
        p_x_t, p_y_t = self.predictive_alignment(rows - 2, cols - 2, c_t)
        r_i = torch.fmod(torch.clamp(torch.stack([torch.arange(p_batch - (self.r_win // 2) + 1,
                                                               p_batch + (self.r_win + 1) // 2 + 1)
                                                  for p_batch in torch.round(p_x_t).long()]),
                                     min=0, max=rows), rows)
        c_i = torch.fmod(torch.clamp(torch.stack([torch.arange(p_batch - (self.c_win // 2) + 1,
                                                               p_batch + (self.c_win + 1) // 2 + 1)
                                                  for p_batch in torch.round(p_y_t).long()]),
                                     min=0, max=cols), cols)
        s_i = torch.stack([r_i[:, i] * cols + c_i[:, j]
                           for i in range(r_i.size(-1))
                           for j in range(c_i.size(-1))], dim=1)
        r_exp = 2 * torch.pow((torch.clamp(r_i - 1, min=0).float() -
                               p_x_t.unsqueeze(-1)) / (self.r_win // 2), 2)
        c_exp = 2 * torch.pow((torch.clamp(c_i - 1, min=0).float() -
                               p_y_t.unsqueeze(-1)) / (self.c_win // 2), 2)
        s_exp = torch.stack([r_exp[..., i] + c_exp[..., j]
                             for i in range(r_exp.size(-1))
                             for j in range(c_exp.size(-1))], dim=1)
        q_i = q_i.view(q_i.size(0), q_i.size(1), -1)
        q_i = q_i.transpose(1, 2)
        q_i = torch.stack([batch[idx] for batch, idx in zip(q_i, s_i)])
        w_attn = self.align(q_i, c_t, s_exp)
        s_t = torch.bmm(w_attn.transpose(1, 2), q_i).squeeze(1)
        return s_t

    def infer(self, q_i, c_t):
        '''Method used for inference when testing. This method unlike the forward method
        returns the weights in addition to the summary vector.

        Inputs: q_i, c_t
            - **q_i** of shape (batch,query_size,num_x_queries,num_y_queries): tensor containing all
            of the queries with variable size num_x_queries and num_y_queries.
            - **c_t** of shape (batch,context_size): tensor containing the context vectors.
            Defaults to zero if not provided.

        Outputs: s_t, w_attn
            - **s_t** of shape (batch,query_size): tensor containing the weighted sum of queries.
            - **w_attn** of shape (batch,num_x_queries,num_y_queries): tensor containing the
            attention weights.
        '''
        if c_t is None:
            c_t = q_i.new_zeros(q_i.size(0), self.c_size)
        q_i = nn.ConstantPad2d((1, 0, 1, 0), float('nan'))(q_i)
        rows, cols = q_i.size(2), q_i.size(3)
        p_x_t, p_y_t = self.predictive_alignment(rows - 2, cols - 2, c_t)
        r_i = torch.fmod(torch.clamp(torch.stack([torch.arange(p_batch - (self.r_win // 2) + 1,
                                                               p_batch + (self.r_win + 1) // 2 + 1)
                                                  for p_batch in torch.round(p_x_t).long()]),
                                     min=0, max=rows), rows)
        c_i = torch.fmod(torch.clamp(torch.stack([torch.arange(p_batch - (self.c_win // 2) + 1,
                                                               p_batch + (self.c_win + 1) // 2 + 1)
                                                  for p_batch in torch.round(p_y_t).long()]), min=0,
                                     max=cols), cols)
        s_i = torch.stack([r_i[:, i] * cols + c_i[:, j]
                           for i in range(r_i.size(-1))
                           for j in range(c_i.size(-1))], dim=1)
        r_exp = 2 * torch.pow((torch.clamp(r_i - 1, min=0).float() -
                               p_x_t.unsqueeze(-1)) / (self.r_win // 2), 2)
        c_exp = 2 * torch.pow((torch.clamp(c_i - 1, min=0).float() -
                               p_y_t.unsqueeze(-1)) / (self.c_win // 2), 2)
        s_exp = torch.stack([r_exp[..., i] + c_exp[..., j]
                             for i in range(r_exp.size(-1))
                             for j in range(c_exp.size(-1))], dim=1)
        q_i = q_i.view(q_i.size(0), q_i.size(1), -1)
        q_i = q_i.transpose(1, 2)
        q_i = torch.stack([batch[idx] for batch, idx in zip(q_i, s_i)])
        w_attn = self.align(q_i, c_t, s_exp)
        s_t = torch.bmm(w_attn.transpose(1, 2), q_i).squeeze(1)
        w_attn = w_attn.view(w_attn.size(0), self.r_win, self.c_win)
        s_i = torch.stack([torch.div(s_i, cols).long(),
                           torch.fmod(s_i, cols).long()], dim=2)
        s_i = s_i.view(s_i.size(0), self.r_win, self.c_win, 2)
        w_attn = self.alpha_frames(rows, cols, w_attn, s_i)
        return s_t, w_attn

    def predictive_alignment(self, s_x, s_y, c_t):
        '''Predict the window centroid given total number of queries in both dimensions and the
        context vector.'''
        loc = self.sigmoid(self.w_p(c_t))
        p_x_t = loc[:, 0] * s_x
        p_y_t = loc[:, 1] * s_y
        return p_x_t, p_y_t

    def align(self, q_i, c_t, s_exp):
        '''The alignment method converts scores to probabilities and yields the attention weights.
        A trick is used to ensure that if the area around the center goes out of bounds,
        it does not contribute to the learnable weights and the attention weights themselves.'''
        nan_loc = torch.isnan(q_i[..., 0])
        q_i[nan_loc] = 0
        a_t = self.score(q_i, c_t)
        a_t[nan_loc] = -float('inf')
        w_attn = self.softmax(a_t + (-s_exp.unsqueeze(2)))
        return w_attn

    def score(self, q_i, c_t):
        '''General attention scoring as outlined in Luong et al. Scoring is a metric for
        deterimining how relevant certain queries are to the given context vector.'''
        a_t = torch.bmm(self.w_a(q_i), c_t.unsqueeze(2))
        return a_t

    def alpha_frames(self, rows, cols, w_attn, loc):
        '''Method for returning the weights in the shape of the original input.
        Args:
            rows (int): the number of rows in the queries.
            cols (int): the number of columns in the queries.

        Inputs: w_attn, loc
            - **w_attn** of shape (batch,r_win,c_win): tensor containing the attention weights.
            - **loc** of shape(batch,r_win,c_win, 2): tensor containing the (x,y) locations in the
            nan-padded query grid for every weight.

        Outputs: alpha_matrix
            - **alpha_matrix** of shape (batch,num_x_queries,num_y_queries): tensor containing the
            weights in the shape of the original input.
        '''
        alpha_matrix = torch.zeros(w_attn.size(0), rows, cols)
        alpha_matrix[:, loc[..., 0], loc[..., 1]] = w_attn
        return alpha_matrix[:, 1:, 1:]
