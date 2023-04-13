"""Structural Agnostic Model.

Author: Diviyan Kalainathan, Olivier Goudet
Date: 09/3/2018
"""
import math
import numpy as np
import torch as th
from time import time
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from joblib import Parallel, delayed
from sklearn.preprocessing import scale
from ..utils.linear3d import Linear3D
from ..utils.graph import GraphSampler, notears_constr


class SAM_generators(th.nn.Module):
    """Ensemble of all the generators."""

    def __init__(self, data_shape, nh):
        """Init the model."""
        super(SAM_generators, self).__init__()
        # Commented: Non linear function
        # self.weights_in = Linear3D(data_shape[1], data_shape[1], nh)
        # self.act = th.nn.Tanh()
        # self.weights_out = Linear3D(data_shape[1], nh, 1)
        self.weights = Linear3D(data_shape[1], data_shape[1], 1)

    def forward(self, data, adj_matrix):
        """Forward through all the generators."""
        # return self.weights_out(self.act(self.weights_in(data, adj_matrix)))[:, :, 0]
        return self.weights(data, adj_matrix)[:, :, 0]


class SAM_discriminator(th.nn.Module):
    """SAM discriminator."""

    def __init__(self):
        """We use MSE for now."""
        raise NotImplementedError


def run_SAM(in_data, skeleton=None, device="cpu", train=1000, test=1000,
            batch_size=-1, temperature=False, KLpenalization=False, lr_gen=.01, regul_param=.1,
            nh=None, drawhard=True, tau=1, verbose=True):
    """run SAM on data."""
    # lr_disc = kwargs.get('lr_disc', lr_gen)

    d_str = "Epoch: {} -- DAG: {:.4} -- Graph: {:.4} -- Gen: {:.4}"

    list_nodes = list(in_data.columns)
    data = scale(in_data[list_nodes].as_matrix())
    data = data.astype('float32')
    data = th.from_numpy(data)
    data.to(device)

    if batch_size == -1:
        batch_size = data.shape[0]
    rows, cols = data.size()
    # Get the list of indexes to ignore
    if skeleton is not None:
        skeleton = th.from_numpy(skeleton.astype('float32')).to(device)

    sam = SAM_generators(data.shape, nh).to(device)
    graph_sampler = GraphSampler(data.shape[1], mask=skeleton, device=device).to(device)

    criterion = th.nn.MSELoss(reduce=False)
    g_optimizer = th.optim.Adam(sam.parameters(), lr=lr_gen)
    graph_optimizer = th.optim.Adam(graph_sampler.parameters(), lr=lr_gen)

    # _true = th.ones(1).to(device)
    # _false = th.zeros(1).to(device)
    output = th.zeros(data.shape[1], data.shape[1]).to(device)

    data_iterator = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=2)

    # TRAIN
    for epoch in range(train + test):
        for i_batch, batch in enumerate(data_iterator):

            g_optimizer.zero_grad()
            graph_optimizer.zero_grad()
            # Train the discriminator
            drawn_graph = graph_sampler(tau=tau, drawhard=drawhard)
            generated_variables = sam(batch, drawn_graph).t()
            gen_loss = criterion(generated_variables, batch).sum(dim=0).log().sum()
            # 3. Compute filter regularization
            # print(drawn_graph)
            graph_penal = (drawn_graph*2 + 1).sum()
            # print(graph_penal)
            filters = th.nn.functional.sigmoid(2 * graph_sampler.weights) * graph_sampler.mask
            # if KLpenalization:
            #     penal = -th.log(1 - filters)
            # else:
            #     penal = filters
            #
            # # print(filters)
            # l1_reg = regul_param * (penal.sum() - penal.diag().sum())
            # loss = gen_loss + l1_reg
            # print(batch.size()[0] * gen_loss)
            # print(regul_param * float(np.log(batch.size()[0])) * graph_penal.sum())

            # dag_constraint = (drawn_graph @ drawn_graph @ drawn_graph).abs().sum()
            if epoch >= train:
                dag_constraint = notears_constr(drawn_graph)
                loss = (batch.size()[0] * gen_loss + regul_param * float(np.log(batch.size()[0])) * graph_penal)/20000 + .1*dag_constraint

                if verbose and epoch % 20 == 0 and i_batch == 0:

                    print(str(i_batch) + " " + d_str.format(epoch,
                                                            dag_constraint.item(),
                                                            graph_penal.item(),
                                                            gen_loss.item(),
                                                            ))
                    print(filters)

            else:
                loss = (batch.size()[0] * gen_loss + regul_param * float(np.log(batch.size()[0])) * graph_penal.sum())/20000
                # print(regul_param)
                # print(filters)
                if verbose and epoch % 20 == 0 and i_batch == 0:

                    print(str(i_batch) + " " + d_str.format(epoch,
                                                            0,
                                                            graph_penal.item(),
                                                            gen_loss.item(),
                                                            ))
                    print(filters)

            loss.backward()

            # STORE ASSYMETRY values for output
            if epoch >= train:
                output.add_(filters.data)
            g_optimizer.step()
            graph_optimizer.step()
    return th.nn.functional.sigmoid(2 * graph_sampler.weights) * graph_sampler.mask  # output.div_(test).cpu().numpy()


class gSAML3d(object):
    """Structural Agnostic Model."""

    def __init__(self, lr=0.1, dlr=0.1, l1=0.1, nh=200, dnh=200,
                 train_epochs=1000, test_epochs=1000, batchsize=-1,
                 gumble_by_sample=True, temperature=False, KLpenalization=False, drawhard=False):
        """Init and parametrize the SAM model.

        :param lr: Learning rate of the generators
        :param dlr: Learning rate of the discriminator
        :param l1: L1 penalization on the causal filters
        :param nh: Number of hidden units in the generators' hidden layers
        :param dnh: Number of hidden units in the discriminator's hidden layer$
        :param train_epochs: Number of training epochs
        :param test_epochs: Number of test epochs (saving and averaging the causal filters)
        :param batchsize: Size of the batches to be fed to the SAM model.
        """
        super(gSAML3d, self).__init__()
        self.lr = lr
        self.dlr = dlr
        self.l1 = l1
        self.nh = nh
        self.dnh = dnh
        self.train = train_epochs
        self.test = test_epochs
        self.batchsize = batchsize
        self.gumble_by_sample = gumble_by_sample
        self.temperature = temperature
        self.KLpenalization = KLpenalization
        self.drawhard = drawhard

    def exec_sam_instance(self, data, skeleton=None, gpus=0, gpuno=0, verbose=True):
            device = "cuda:{}".format(gpuno) if bool(gpus) else "cpu"
            return run_SAM(data, skeleton=skeleton, lr_gen=self.lr,  # lr_disc=self.dlr,
                           regul_param=self.l1, nh=self.nh,  # dnh=self.dnh,
                           device=device, train=self.train,
                           test=self.test, batch_size=self.batchsize,
                           KLpenalization=self.KLpenalization,
                           temperature=self.temperature, drawhard=self.drawhard,
                           verbose=verbose)

    def predict(self, data, skeleton=None, nruns=6, njobs=1, gpus=0, verbose=True):
        """Execute SAM on a dataset given a skeleton or not.

        :param data: Observational data for estimation of causal relationships by SAM
        :param skeleton: A priori knowledge about the causal relationships as an adjacency matrix.
                         Can be fed either directed or undirected links.
        :param nruns: Number of runs to be made for causal estimation.
                      Recommended: >5 for optimal performance.
        :param njobs: Numbers of jobs to be run in Parallel.
                      Recommended: 1 if no GPU available, 2*number of GPUs else.
        :param gpus: Number of available GPUs for the algorithm.
        :param verbose: verbose mode
        :param plot: Plot losses interactively. Not recommended if nruns>1
        :param plot_generated_pair: plots a generated pair interactively.  Not recommended if nruns>1
        :return: Adjacency matrix (A) of the graph estimated by SAM,
                A[i,j] is the term of the ith variable for the jth generator.
        """
        assert nruns > 0
        if nruns == 1:
            return self.exec_sam_instance(data, skeleton=skeleton,
                                          verbose=verbose)
        else:
            list_out = Parallel(n_jobs=njobs)(delayed(self.exec_sam_instance)(
                                              data, gpus=gpus, skeleton=skeleton,
                                              verbose=verbose,
                                              gpuno=idx % gpus) for idx in range(nruns))

            W = list_out[0]
            for w in list_out[1:]:
                W += w
            W /= nruns
            return W
