"""Structural Agnostic Model.

Author: Diviyan Kalainathan, Olivier Goudet
Date: 09/3/2018
"""
import math
import numpy as np
import torch as th
from torch.autograd import Variable
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from joblib import Parallel, delayed
from sklearn.preprocessing import scale
from time import time
from scipy.misc import factorial
from ..utils.gumble_utils import gumbel_softmax


class SAM_block(th.nn.Module):
    """SAM-Block: conditional generator.

    Generates one variable while selecting the parents. Uses filters to do so.
    One fixed filter and one with parameters on order to keep a fixed skeleton.
    """

    def __init__(self, sizes, zero_components=[], **kwargs):
        """Initialize a generator."""
        super(SAM_block, self).__init__()
        gpu = kwargs.get('gpu', False)
        gpu_no = kwargs.get('gpu_no', 0)
        self.gumble_by_sample = kwargs.get('gumble_by_sample', True)
        self.temperature = kwargs.get('temperature', False)
        self.drawhard = kwargs.get('drawhard', True)

        layers = []
        self.zero_components = zero_components

        for i, j in zip(sizes[:-2], sizes[1:-1]):
            layers.append(th.nn.Linear(i, j))

        layers.append(th.nn.Linear(sizes[-2], sizes[-1]))
        self.layers = th.nn.Sequential(*layers)

        # Filtering the unconnected nodes.
        self._filter = th.ones(sizes[0]-1, 1)
        for i in zero_components:
            self._filter[i, :].zero_()

        self._filter = Variable(self._filter, requires_grad=False)
        self.fs_filter = th.nn.Parameter(self._filter.data)
        self.fs_filter.data.zero_()

        self.mask = Variable(th.zeros(1))
        self.n_mask = Variable(th.ones(1))
        if gpu:
            self._filter = self._filter.cuda(gpu_no)
            self.mask = self.mask.cuda(gpu_no)
            self.n_mask = self.n_mask.cuda(gpu_no)

        self.tau = 1
        self.epoch = 0
        self.proba = None

    def forward(self, x):
        """Feed-forward the model."""
        self.proba = th.stack([c for v in [[gumbel_softmax(th.stack([i, -i], 1), tau=self.tau,
                                            hard=self.drawhard)[:, 0]
                               if idx not in self.zero_components else self.mask.expand(1)
                               for idx, i in enumerate(self.fs_filter)], [self.n_mask]] for c in v], 1)
        return self.layers(x*self.proba.expand_as(x))


class SAM_generators(th.nn.Module):
    """Ensemble of all the generators."""

    def __init__(self, data_shape, zero_components, nh=200, batch_size=-1, **kwargs):
        """Init the model."""
        super(SAM_generators, self).__init__()
        if batch_size == -1:
            batch_size = data_shape[0]
        gpu = kwargs.get('gpu', False)
        gpu_no = kwargs.get('gpu_no', 0)
        rows, self.cols = data_shape

        # building the computation graph
        self.noise = [Variable(th.FloatTensor(batch_size, 1))
                      for i in range(self.cols)]
        if gpu:
            self.noise = [i.cuda(gpu_no) for i in self.noise]
        self.blocks = th.nn.ModuleList()

        # Init all the blocks
        for i in range(self.cols):
            self.blocks.append(SAM_block(
                [self.cols + 1, 1], zero_components[i], **kwargs))

    def forward(self, x):
        """Feed-forward the model."""
        for i in self.noise:
            i.data.normal_()

        self.generated_variables = [self.blocks[i](
            th.cat([x, self.noise[i]], 1)) for i in range(self.cols)]
        return self.generated_variables


def run_SAM(df_data, skeleton=None, **kwargs):
    """Execute the SAM model.

    :param df_data:
    """
    gpu = kwargs.get('gpu', False)
    gpu_no = kwargs.get('gpu_no', 0)

    train_epochs = kwargs.get('train_epochs', 1000)
    test_epochs = kwargs.get('test_epochs', 1000)
    batch_size = kwargs.get('batch_size', -1)
    KLpenalization = kwargs.get('KLpenalization', False)
    lr_gen = kwargs.get('lr_gen', 0.1)
    lr_disc = kwargs.get('lr_disc', lr_gen)
    verbose = kwargs.get('verbose', True)
    regul_param = kwargs.get('regul_param', 0.1)
    dnh = kwargs.get('dnh', None)

    plot = kwargs.get("plot", False)
    plot_generated_pair = kwargs.get("plot_generated_pair", False)

    d_str = "Epoch: {} -- Disc: {} -- Gen: {} -- L1: {}"

    list_nodes = list(df_data.columns)
    df_data = scale(df_data[list_nodes].as_matrix())
    data = df_data.astype('float32')
    data = th.from_numpy(data)
    if batch_size == -1:
        batch_size = data.shape[0]
    rows, cols = data.size()
    softmax = th.nn.Softmax(dim=2)
    # Get the list of indexes to ignore
    if skeleton is not None:
        connections = []
        for idx, i in enumerate(list_nodes):
            connections.append([list_nodes.index(j)
                                for j in skeleton.dict_nw()[i]])

        zero_components = [
            [i for i in range(cols) if i not in j] for j in connections]

    else:
        zero_components = [[i] for i in range(cols)]

    sam = SAM_generators((rows, cols), zero_components, batch_norm=True, **kwargs)

    # Begin UGLY
    activation_function = kwargs.get('activation_function', th.nn.Tanh)
    try:
        del kwargs["activation_function"]
    except KeyError:
        pass

    kwargs["activation_function"] = activation_function
    # End of UGLY

    if gpu:
        sam = sam.cuda(gpu_no)

        data = data.cuda(gpu_no)

    # Select parameters to optimize : ignore the non connected nodes
    criterion = th.nn.MSELoss(reduce=False)
    g_optimizer = th.optim.Adam(sam.parameters(), lr=lr_gen)

    true_variable = Variable(
        th.ones(batch_size, 1), requires_grad=False)
    false_variable = Variable(
        th.zeros(batch_size, 1), requires_grad=False)
    causal_filters = th.zeros(data.shape[1], data.shape[1])

    def fill_nan(mat):
        mat[mat.ne(mat)] = 0
        return mat

    if gpu:
        true_variable = true_variable.cuda(gpu_no)
        false_variable = false_variable.cuda(gpu_no)
        causal_filters = causal_filters.cuda(gpu_no)

    data_iterator = DataLoader(data, batch_size=batch_size, shuffle=True)

    # TRAIN
    for epoch in range(train_epochs + test_epochs):
        for i_batch, batch in enumerate(data_iterator):
            batch = Variable(batch)

            g_optimizer.zero_grad()
            # Train the discriminator
            generated_variables = th.cat(sam(batch), 1)
            gen_loss = criterion(generated_variables, batch).sum(dim=0).log().sum()
            # 3. Compute filter regularization

            drawn_graph = th.stack([i.proba[0, :-1] for i in sam.blocks], 1)
            # print(drawn_graph)
            graph_penal = drawn_graph*2 + 1
            # print(graph_penal)
            filters = th.nn.functional.sigmoid(2 * th.stack([i.fs_filter[:, 0] for i in sam.blocks], 1))
            #
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
            if epoch > 20000:
                sum_terms = [drawn_graph, drawn_graph @ drawn_graph, drawn_graph @ drawn_graph @ drawn_graph]
                dag_constraint = sum([i/float(factorial(idx)) for idx, i in enumerate(sum_terms)]).diag().sum()
                loss = (batch.size()[0] * gen_loss + regul_param * float(np.log(batch.size()[0])) * graph_penal.sum())/20000 + .1*dag_constraint

                if verbose and epoch % 20 == 0 and i_batch == 0:

                    print(str(i_batch) + " " + d_str.format(epoch,
                                                            dag_constraint,
                                                            gen_loss.cpu(
                                                            ).data[0] / cols,
                                                            loss.cpu().data[0]))

            else:
                loss = (batch.size()[0] * gen_loss + regul_param * float(np.log(batch.size()[0])) * graph_penal.sum())/20000
                # print(regul_param)
                # print(filters)
                if verbose and epoch % 20 == 0 and i_batch == 0:

                    print(str(i_batch) + " " + d_str.format(epoch,
                                                            graph_penal.sum(),
                                                            gen_loss.cpu(
                                                            ).data[0] / cols,
                                                            loss.cpu().data[0]))
            loss.backward()

            # STORE ASSYMETRY values for output
            if epoch >= train_epochs:
                causal_filters.add_(filters.data)
            g_optimizer.step()

            if plot and i_batch == 0:
                try:
                    ax.clear()
                    ax.plot(range(len(adv_plt)), adv_plt, "r-",
                            linewidth=1.5, markersize=4,
                            label="Discriminator")
                    ax.plot(range(len(adv_plt)), gen_plt, "g-", linewidth=1.5,
                            markersize=4, label="Generators")
                    ax.plot(range(len(adv_plt)), l1_plt, "b-",
                            linewidth=1.5, markersize=4,
                            label="L1-Regularization")
                    ax.plot(range(len(adv_plt)), asym_plt, "c-",
                            linewidth=1.5, markersize=4,
                            label="Assym penalization")

                    plt.legend()

                    adv_plt.append(adv_loss.cpu().data[0])
                    gen_plt.append(gen_loss.cpu().data[0] / cols)
                    l1_plt.append(l1_reg.cpu().data[0])
                    asym_plt.append(asymmetry_reg.cpu().data[0])
                    plt.pause(0.0001)

                except NameError:
                    plt.ion()
                    plt.figure()
                    plt.xlabel("Epoch")
                    plt.ylabel("Losses")

                    plt.pause(0.0001)

                    # adv_plt = [adv_loss.cpu().data[0]]
                    gen_plt = [gen_loss.cpu().data[0] / cols]
                    l1_plt = [loss.cpu().data[0]]

            elif plot:
                # adv_plt.append(adv_loss.cpu().data[0])
                gen_plt.append(gen_loss.cpu().data[0] / cols)
                l1_plt.append(loss.cpu().data[0])

            if plot_generated_pair and epoch % 200 == 0:
                if epoch == 0:
                    plt.ion()
                to_print = [[0, 1]]  # , [1, 0]]  # [2, 3]]  # , [11, 17]]
                plt.clf()
                for (i, j) in to_print:

                    plt.scatter(generated_variables[i].data.cpu().numpy(
                    ), batch.data.cpu().numpy()[:, j], label="Y -> X")
                    plt.scatter(batch.data.cpu().numpy()[
                        :, i], generated_variables[j].data.cpu().numpy(), label="X -> Y")

                    plt.scatter(batch.data.cpu().numpy()[:, i], batch.data.cpu().numpy()[
                        :, j], label="original data")
                    plt.legend()

                plt.pause(0.01)
    return causal_filters.div_(test_epochs).cpu().numpy()


class gSAML(object):
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
        super(gSAML, self).__init__()
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

    def exec_sam_instance(self, data, skeleton=None, gpus=0, gpuno=0, plot=False, verbose=True):
            return run_SAM(data, skeleton=skeleton, lr_gen=self.lr, lr_disc=self.dlr,
                           regul_param=self.l1, nh=self.nh, dnh=self.dnh,
                           gpu=bool(gpus), gpu_no=gpuno, train_epochs=self.train,
                           test_epochs=self.test, batch_size=self.batchsize,
                           gumble_by_sample=self.gumble_by_sample, KLpenalization=self.KLpenalization,
                           temperature=self.temperature, drawhard=self.drawhard,
                           plot=plot, verbose=verbose)

    def predict(self, data, skeleton=None, nruns=6, njobs=1, gpus=0, verbose=True,
                plot=False, plot_generated_pair=False):
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
                                          plot=plot, verbose=verbose)
        else:
            list_out = Parallel(n_jobs=njobs)(delayed(self.exec_sam_instance)(
                                              data, gpus=gpus, skeleton=skeleton,
                                              plot=plot, verbose=verbose,
                                              gpuno=idx % gpus) for idx in range(nruns))

            W = list_out[0]
            for w in list_out[1:]:
                W += w
            W /= nruns
            return W
