# created by Yuanbiao Wang
# implements a simple contrastive learning pretrain learner


import jittor as jt
import jittor.nn as nn
from advance.ssl_utils import *
from tqdm import tqdm


class Co(nn.Module):
    '''
    modified from https://github.com/facebookresearch/moco/blob/master/moco/builder.py
    a simple contrastive learning loss computer
    '''
    def __init__(self, encoder, embedding_channel, projection_dim, K=1024, T=0.07, dim=128):
        super(Co, self).__init__()
        self.K = K
        self.T = T
        self.encoder = encoder
        self.project = Projection(embedding_channel, projection_dim)
        self.queue = jt.randn(dim, K)
        self.queue = jt.misc.normalize(self.queue, dim=0)
        self.ptr = 0
        
    def _dequeue_and_enqueue(self, keys):
        # implements a queue-like ring buffer
        # for the cause of simplicity, if the length of the buffer to be appended
        # exceeds the remaining room of the buffer, we curtail it and reset the 
        # start pointer to the beginning of the buffer array
        with jt.no_grad():
            batch_size = keys.shape[0]
            left_space = self.K - self.ptr
            key_size = min(batch_size, left_space)
            keys = keys[:key_size]
            self.queue[:, self.ptr : self.ptr + key_size] = keys.transpose()
            self.ptr = (self.ptr + key_size) % self.K 
        
    def execute(self, im_q, im_k):
        # computes contrastive loss by cosine similarity
        # and returns the predicted logits and groundtruth labels
        q = self.encoder(im_q)
        q = self.project(q)
        q = jt.misc.normalize(q, dim=1)
        k = self.encoder(im_k)
        k = self.project(k)
        k = jt.misc.normalize(k, dim=1)
        l_pos = (q * k).sum(dim=1).unsqueeze(-1)
        l_neg = jt.matmul(q, self.queue.clone().detach())
        logits = jt.contrib.concat([l_pos, l_neg], dim=1)
        logits /= self.T
        labels = jt.zeros(logits.shape[0], dtype=jt.int)
        self._dequeue_and_enqueue(k)
        return logits, labels


class OutputHiddenLayer(nn.Module):
    '''
    implements a single wrapper to outputs hidden layer feature of a neural network
    modified from https://github.com/lucidrains/contrastive-learner/blob/master/contrastive_learner/contrastive_learner.py
    '''
    def __init__(self, net, layer=(-2)):
        '''
        :param net: the neural network
        :param layer: the index of the hidden layer or the name of the hidden layer
        '''
        super().__init__()
        self.net = net
        self.layer = layer
        self.hidden = None
        self._register_hook()

    def _find_layer(self):
        if (type(self.layer) == str):
            modules = dict([*self.net.named_modules()])
            return modules.get(self.layer, None)
        elif (type(self.layer) == int):
            children = [*self.net.children()]
            return children[self.layer]
        return None

    def _register_hook(self):
        def hook(_, __, output):
            self.hidden = output
        layer = self._find_layer()
        assert (layer is not None)
        handle = layer.register_forward_hook(hook)

    def execute(self, x):
        if (self.layer == (- 1)):
            return self.net(x)
        _ = self.net(x)
        hidden = self.hidden
        self.hidden = None
        assert (hidden is not None)
        return hidden


class Projection(nn.Module):
    '''
    a simple project head for the contrastive learning
    inspired by the SimCLR paper
    '''
    def __init__(self, input_channel, project_dim):
        super(Projection, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(output_size=2)
        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(input_channel * 4, project_dim)
        )
    
    def execute(self, x):
        y = self.pool(x)
        y = y.view(x.size(0), -1)
        y = self.fc(y)
        return y
    
        
class CoLearner():
    '''
    a contrastive learning wrapper
    need to designate the model, the hidden layer and the data loader for the SSL training
    '''
    def __init__(self, model, layer, loader, embedding_channel=1024, project_dim=128):
        super(CoLearner, self).__init__()
        encoder = OutputHiddenLayer(model, layer)
        self.co = Co(encoder, embedding_channel, project_dim)
        self.loader = loader
        self.criterion = nn.CrossEntropyLoss()
        self.optim = jt.optim.Adam(model.parameters(), lr=1e-5)
    
    def update(self, query, key):
        output, target = self.co(query, key)
        loss = self.criterion(output, target)
        self.optim.step(loss)
        return loss.item()
    
    def train(self):
        loss_mean = 0.0
        total = 0
        bar = tqdm(self.loader, desc='loss')
        for i, (query, key, _) in enumerate(bar):
            loss = self.update(query, key)
            bar.set_description('loss: [%.6f]' % loss)
            bar.update()
            loss_mean += loss * query.shape[0]
            total += query.shape[0]
        loss_mean /= total
        return loss_mean
    