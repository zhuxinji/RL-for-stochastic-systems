import numpy.linalg as alg

import numpy as np
import math
import torch

# import debugpy
# debugpy.debug_this_thread()


class NetF(torch.nn.Module):
    def __init__(self):
        super(NetF, self).__init__()

        self.net = torch.nn.Sequential(
            # #---------------------------
            torch.nn.Linear(2, 512, bias=False),
            torch.nn.Tanh(),
            # torch.nn.ReLU(),
            torch.nn.Linear(512, 512, bias=False),
            # torch.nn.ReLU(),
            torch.nn.Tanh(),
            torch.nn.Linear(512, 2, bias=False),
            # torch.nn.Linear(2, 2, bias=False),
            # torch.nn.Tanh(),

        )

    def forward(self, x, dx):
        a = [x, dx]
        # a = Normalize(a)
        data_input = torch.cat(a, dim=1)
        # self.net[3].weight = torch.nn.Parameter(torch.Tensor(
        #     np.array([[1/20, 0], [0, 1/20]])))
        y = self.net(data_input)
        # y = mean_std(y)  # y*20 #
        y = y
        return y


def mean_std(m, mean_ub=[35., 35.], mean_lb=[15., 15.]):

    mean = (torch.tensor(mean_ub) - torch.tensor(mean_lb)) * \
        (m-1)/2 + torch.tensor(mean_ub)  # tanh
    return mean


def Normalize(data):
    mu = [0, 0]
    sigma = [1, 15]
    data = [(data[i] - mu[i])/sigma[i] for i in range(len(data))]
    return data


def Initialize_weight(NNmodel):

    for i in range(len(NNmodel.net)):
        if type(NNmodel.net[i]) == torch.nn.Linear:
            a = NNmodel.net[i].weight.shape[0]
            b = NNmodel.net[i].weight.shape[1]
            # weight = torch.nn.init.uniform_(torch.Tensor(a, b), 0, 0.3)
            weight = torch.nn.init.uniform_(torch.Tensor(a, b),0, 0.1)
            NNmodel.net[i].weight = torch.nn.Parameter(torch.tensor(weight))
    print(NNmodel.net[0].weight)
    return NNmodel


class Counter:
    def __init__(self):
        self.count, self.sum, self.avg = 0, 0, 0
        return

    def update(self, value, num_updata=1):
        self.count += num_updata
        self.sum += value * num_updata
        self.avg = self.sum / self.count
        return

    def clear(self):
        self.count, self.sum, self.avg = 0, 0, 0
        return


# class Net(torch.nn.Module):
#   def __init__(self, **kwargs):
#     super(Net, self).__init__()

#     # Unpack the dictionary
#     self.args = kwargs
#     self.dtype = torch.float
#     self.use_cuda = torch.cuda.is_available()
#     self.device = torch.device("cpu")

#     # defining ANN topology
#     self.input_size = self.args['input_size']
#     self.hs1 = self.args['hs1']
#     self.hs2 = self.args['hs2']
#     self.output_sz = self.args['output_size']

#     # defining activation functions
#     self.tanh = torch.nn.Tanh()

#     # connect the layers
#     self.hidden1 = torch.nn.RNN(self.input_size, self.hs1)
#     self.hidden2 = torch.nn.Linear(self.hs1, self.hs2)

#     # defining output layer
#     self.output = torch.nn.Linear(self.hs2, self.output_sz)
#     self.reset_hn()

# def forward(self, x):
#     x  = torch.tensor(x.view(1,1,-1)).float()
#     a1, self.hn = self.hidden1(x, self.hn)
#     a2  = self.tanh(self.hidden2(a1))
#     y   = self.tanh(self.output(a2))

#     y = mean_std(y)
#     return y.detach().numpy()


class NetBF(torch.nn.Module):
    def __init__(self):
        super(NetBF, self).__init__()

        self.net = torch.nn.Sequential(
            # #---------------------------
            torch.nn.Linear(2, 64, bias=True),
            torch.nn.Tanh(),
            # torch.nn.ReLU(),
            torch.nn.Linear(64, 32, bias=True),
            # torch.nn.ReLU(),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 1, bias=True),
        )

    def forward(self, x, dx):
        a = [x, dx]
        data_input = torch.cat(a, dim=1)
        y = self.net(data_input)
        y = y
        return y