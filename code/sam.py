"""Structural Agnostic Model.

Author: Diviyan Kalainathan, Olivier Goudet
Date: 09/3/2018
"""
import os
import numpy as np
import torch as th
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.preprocessing import scale
from .utils.linear3d import Linear3D
from .utils.graph import SimpleMatrixConnection, MatrixSampler, MatrixSampler3, notears_constr
from .utils.batchnorm import ChannelBatchNorm1d
from .utils.batchnorm import ParallelBatchNorm1d

from .utils.treillis import compute_total_effect
from .utils.parlib import parallel_identical


class SAM_generators(th.nn.Module):
    """Ensemble of all the generators."""

    def permutation_matrix(self, skeleton, data_shape, max_dim):
        reshape_skeleton = th.zeros(self.nb_vars, int(data_shape[1]), max_dim)

        for channel in range(self.nb_vars):
            perm_matrix = skeleton[:, channel] * th.eye(data_shape[1],data_shape[1])
            skeleton_list = [i for i in th.unbind(perm_matrix, 1) if len(th.nonzero(i)) > 0]
            perm_matrix = th.stack(skeleton_list, 1) if len(skeleton_list)>0 else th.zeros(data_shape[1], 1)
            reshape_skeleton[channel, :, :perm_matrix.shape[1]] = perm_matrix

        return reshape_skeleton

    def __init__(self, data_shape, nh, skeleton=None, cat_sizes=None, linear=False, numberHiddenLayersG=1):
        """Init the model."""
        super(SAM_generators, self).__init__()
        layers = []
        # Building skeleton
        self.sizes = cat_sizes
        self.linear = linear

        if cat_sizes is not None:
            nb_vars = len(cat_sizes)
            output_dim = max(cat_sizes)
            cat_reshape = th.zeros(nb_vars, sum(cat_sizes))
            for var, (cat, cumul) in enumerate(zip(cat_sizes, np.cumsum(cat_sizes))):
                cat_reshape[var, cumul-cat:cumul].fill_(1)
        else:
            nb_vars = data_shape[1]
            output_dim = 1
            cat_reshape = th.eye(nb_vars, nb_vars)

        self.nb_vars = nb_vars
        if skeleton is None:
            skeleton = 1 - th.eye(nb_vars, nb_vars)

        # Redimensioning the skeleton according to the categorical vars
        skeleton = cat_reshape.t() @ skeleton @ cat_reshape
        # torch 0.4.1
        max_dim = th.as_tensor(skeleton.sum(dim=0).max(), dtype=th.int)
        # torch 0.4.0
        # max_dim = skeleton.sum(dim=0).max()

        # Building the custom matrix for reshaping.
        reshape_skeleton = self.permutation_matrix(skeleton, data_shape, max_dim)       

        if linear:
            self.input_layer = Linear3D(nb_vars, max_dim, output_dim,
                                        noise=True, batch_size=data_shape[0])
        else:
            self.input_layer = Linear3D(nb_vars, max_dim, nh, noise=True, batch_size=data_shape[0])
            layers.append(ChannelBatchNorm1d(nb_vars, nh))
            layers.append(th.nn.Tanh())


            for i in range(numberHiddenLayersG-1):
                layers.append(Linear3D(nb_vars, nh, nh))
                layers.append(ChannelBatchNorm1d(nb_vars, nh))
                layers.append(th.nn.Tanh())

            self.output_layer = Linear3D(nb_vars, nh, output_dim)
            # self.weights = Linear3D(data_shape[1], data_shape[1], 1)
            self.layers = th.nn.Sequential(*layers)

        self.register_buffer('skeleton', reshape_skeleton)
        self.register_buffer("categorical_matrix", cat_reshape)

    def forward(self, data, adj_matrix, drawn_neurons=None):
        """Forward through all the generators."""

        if self.linear:
            output = self.input_layer(data, self.categorical_matrix.t() @ adj_matrix, self.skeleton)
        else:
            output = self.output_layer(self.layers(self.input_layer(data,
                                                                self.categorical_matrix.t() @ adj_matrix,
                                                                self.skeleton)),
                                                                drawn_neurons)

        if self.sizes is None:
            return output.squeeze(2)
        else:
            return th.cat([th.nn.functional.softmax(output[:, idx, :i], dim=1)
                           if i>1 else output[:, idx, :i] for idx, i in enumerate(self.sizes)], 1)

    def reset_parameters(self):
        if not self.linear:
            self.output_layer.reset_parameters()
            for layer in self.layers:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
        self.input_layer.reset_parameters()

    def apply_filter(self, skeleton, data_shape, device):

        skeleton = self.categorical_matrix.cpu().t() @ skeleton @ self.categorical_matrix.cpu()
        max_dim = skeleton.sum(dim=0).max()
        reshape_skeleton = self.permutation_matrix(skeleton,
                                                   data_shape,
                                                   max_dim).to(device)

        self.register_buffer('skeleton', reshape_skeleton)
        self.input_layer.apply_filter(th.cat([self.skeleton,
                                              th.ones(self.skeleton.shape[0],
                                                      self.skeleton.shape[1],
                                                      1).to(device)],2) )


class SAM_discriminator(th.nn.Module):
    """SAM discriminator."""

    def __init__(self, nfeatures, dnh, numberHiddenLayersD = 2, mask=None):
        super(SAM_discriminator, self).__init__()
        self.nfeatures = nfeatures
        layers = []
        layers.append(th.nn.Linear(nfeatures, dnh))
        layers.append(ParallelBatchNorm1d(dnh))
        layers.append(th.nn.LeakyReLU(.2))
        for i in range(numberHiddenLayersD-1):
            layers.append(th.nn.Linear(dnh, dnh))
            layers.append(ParallelBatchNorm1d(dnh))
            layers.append(th.nn.LeakyReLU(.2))

        layers.append(th.nn.Linear(dnh, 1))
        self.layers = th.nn.Sequential(*layers)

        if mask is None:
            mask = th.eye(nfeatures, nfeatures)
        self.register_buffer("mask", mask.unsqueeze(0))

    def forward(self, input, obs_data=None):
        if obs_data is not None:
            return self.layers(obs_data.unsqueeze(1) * (1 - self.mask) + input.unsqueeze(1) * self.mask)
        else:
            return self.layers(input)
    def reset_parameters(self):
        for layer in self.layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()



def run_SAM(in_data, skeleton=None, is_mixed=False, device="cpu",
            train=10000, test=1,
            batch_size=-1, lr_gen=.001,
            lr_disc=.01, lambda1=0.001, lambda2=0.0000001, nh=None, dnh=None,
            verbose=True, losstype="fgan", functionalComplexity="numberHiddenUnits", sampletype="sigmoidproba", initweight=0,
            kfactor=100, dagstart=0, dagloss=False,
            dagpenalization=0.05, dagpenalization_increase=0.0,
            categorical_threshold=50, use_filter=False,
            filter_threshold=0.5, dag_threshold=0.5, eval_total_effect=False,
            linear=False, numberHiddenLayersG=1, numberHiddenLayersD=2, cleanup=True, idx=0):

    d_str = "Epoch: {} -- Disc: {:.4f} --  Total: {:.4f} -- Gen: {:.4f} -- L1: {:.4f}"
    # print("KLPenal:{}, fganLoss:{}".format(KLpenalization, fganLoss))
    list_nodes = list(in_data.columns)
    if is_mixed:
        onehotdata = []
        for i in range(len(list_nodes)):
            # print(pd.get_dummies(in_data.iloc[:, i]).values.shape[1])
            if pd.get_dummies(in_data.iloc[:, i]).values.shape[1] < categorical_threshold:
                onehotdata.append(pd.get_dummies(in_data.iloc[:, i]).values)
            else:
                onehotdata.append(scale(in_data.iloc[:, [i]].values))
        cat_sizes = [i.shape[1] for i in onehotdata]

        data = np.concatenate(onehotdata, 1)
    else:
        data = scale(in_data[list_nodes].values)
        cat_sizes = None

    nb_var = len(list_nodes)
    data = data.astype('float32')
    data = th.from_numpy(data).to(device)
    if batch_size == -1:
        batch_size = data.shape[0]

    lambda2_sauv = lambda2

    lambda1 = lambda1/data.shape[0]
    lambda2 = lambda2/data.shape[0]


    rows, cols = data.size()
    # Get the list of indexes to ignore
    if skeleton is not None:
        skeleton = th.from_numpy(skeleton.astype('float32'))

    sam = SAM_generators((batch_size, cols), nh, skeleton=skeleton,
                         cat_sizes=cat_sizes, linear=linear, numberHiddenLayersG=numberHiddenLayersG).to(device)

    sam.reset_parameters()
    g_optimizer = th.optim.Adam(list(sam.parameters()), lr=lr_gen)

    if losstype != "mse":
        discriminator = SAM_discriminator(cols, dnh, numberHiddenLayersD,
                                          mask=sam.categorical_matrix,).to(device)
        discriminator.reset_parameters()
        d_optimizer = th.optim.Adam(discriminator.parameters(), lr=lr_disc)
        criterion = th.nn.BCEWithLogitsLoss()
    else:
        criterion = th.nn.MSELoss()
        disc_loss = th.zeros(1)


    if sampletype == "sigmoid":

        graph_sampler = SimpleMatrixConnection(len(list_nodes), mask=skeleton).to(device)

    elif sampletype == "sigmoidproba":

        graph_sampler = MatrixSampler(len(list_nodes), mask=skeleton, gumble=False).to(device)

    elif sampletype == "gumbleproba":

        graph_sampler = MatrixSampler(len(list_nodes), mask=skeleton, gumble=True).to(device)

    elif sampletype == "correlproba":

        graph_sampler = MatrixSampler3(len(list_nodes), mask=skeleton).to(device)
        graph_sampler.k = kfactor


    graph_sampler.weights.data.fill_(2)

    graph_optimizer = th.optim.Adam(graph_sampler.parameters(), lr=lr_gen)

    if not linear and functionalComplexity=="numberHiddenUnits":
        neuron_sampler = MatrixSampler((nh, len(list_nodes)), mask=False, gumble=True).to(device)
        neuron_optimizer = th.optim.Adam(list(neuron_sampler.parameters()),lr=lr_gen)

    _true = th.ones(1).to(device)
    _false = th.zeros(1).to(device)
    output = th.zeros(len(list_nodes), len(list_nodes)).to(device)

    data_iterator = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)




    # RUN
    if verbose:
        pbar = tqdm(range(train + test))
    else:
        pbar = range(train+test)
    for epoch in pbar:
        for i_batch, batch in enumerate(data_iterator):

            if eval_total_effect:
                batch.requires_grad_()
            g_optimizer.zero_grad()
            graph_optimizer.zero_grad()

            if losstype != "mse":
                d_optimizer.zero_grad()

            if not linear and functionalComplexity=="numberHiddenUnits":
                neuron_optimizer.zero_grad()

            # Train the discriminator

            if not(epoch > train and eval_total_effect):
                drawn_graph = graph_sampler()

                if not linear and functionalComplexity=="numberHiddenUnits":
                    drawn_neurons = neuron_sampler()


            if linear or functionalComplexity!="numberHiddenUnits":
                generated_variables = sam(batch, drawn_graph)
            else:
                generated_variables = sam(batch, drawn_graph, drawn_neurons)

            if losstype == "mse":
                gen_loss = criterion(generated_variables, batch)
            else:
                disc_vars_d = discriminator(generated_variables.detach(), batch)
                disc_vars_g = discriminator(generated_variables, batch)
                true_vars_disc = discriminator(batch)

                if losstype == "gan":
                    disc_loss = sum([criterion(gen, _false.expand_as(gen)) for gen in disc_vars_d]) / nb_var \
                                     + criterion(true_vars_disc, _true.expand_as(true_vars_disc))
                    # Gen Losses per generator: multiply py the number of channels
                    gen_loss = sum([criterion(gen,
                                              _true.expand_as(gen))
                                    for gen in disc_vars_g])
                elif losstype == "fgan":

                    disc_loss = th.mean(th.exp(disc_vars_d - 1), [0, 2]).sum() / nb_var - th.mean(true_vars_disc)
                    gen_loss = -th.mean(th.exp(disc_vars_g - 1), [0, 2]).sum()


                disc_loss.backward()
                d_optimizer.step()

            filters = graph_sampler.get_proba()

            struc_loss = lambda1*drawn_graph.sum()

            if linear :
                func_loss = 0 
            else :
                if functionalComplexity=="numberHiddenUnits":
                    func_loss = lambda2*drawn_neurons.sum()


                elif functionalComplexity=="l2_norm":
                    l2_reg = th.tensor(0.).to(device)
                    for param in sam.parameters():
                        l2_reg += th.norm(param)

                    func_loss = lambda2*l2_reg



            regul_loss = struc_loss + func_loss


            # Optional: prune edges and sam parameters before dag search
            if epoch == int(train*dagstart) and use_filter:
                ones_tensor = th.ones(len(list_nodes),len(list_nodes))
                zeros_tensor = th.zeros(len(list_nodes),len(list_nodes))
                if not linear:
                    skeleton = th.where(filters.cpu() > filter_threshold, ones_tensor, zeros_tensor)
                sam.apply_filter(skeleton, (batch_size, cols), device)
                graph_sampler.set_skeleton(skeleton.to(device))
                g_optimizer = th.optim.Adam(list(sam.parameters()), lr=lr_gen)

            if dagloss and epoch > train * dagstart:
                dag_constraint = notears_constr(filters*filters)
                #dag_constraint = notears_constr(drawn_graph)

                loss = gen_loss + regul_loss + (dagpenalization + (epoch - train * dagstart) * dagpenalization_increase) * dag_constraint  
            else:
                loss = gen_loss + regul_loss
            if verbose and epoch % 20 == 0 and i_batch == 0:
                pbar.set_postfix(gen=gen_loss.item()/cols,
                                 disc=disc_loss.item(),
                                 regul_loss=regul_loss.item(),
                                 tot=loss.item())

            if epoch < train + test - 1:
                loss.backward(retain_graph=True)
            
            if epoch >= train:
                output.add_(filters.data)

            # Retrieve final DAG
            if epoch == train and eval_total_effect:

                drawn_graph_cpu = th.where(drawn_graph.cpu() > dag_threshold,
                                           th.ones(cols, cols),
                                           th.zeros(cols, cols))

                if not linear and functionalComplexity=="numberHiddenUnits":
                    drawn_neurons_cpu = th.where(drawn_neurons.cpu() > 0.5,
                                                th.ones((nh, cols)),
                                                th.zeros((nh, cols)))
                    drawn_neurons = drawn_neurons_cpu.to(device)

                drawn_graph = drawn_graph_cpu.to(device)
                sam.eval()

            g_optimizer.step()
            graph_optimizer.step()
            if not linear and functionalComplexity=="numberHiddenUnits":
                neuron_optimizer.step()




    if not eval_total_effect:
        return output.div_(test).cpu().numpy()
    else:
        return compute_gradients(batch, generated_variables,
                                 drawn_graph.detach().cpu().numpy(), in_data)
    # Evaluate total effect with final DAG

def compute_gradients(input_data, output, graph, raw_data):
    cols = output.shape[1]
    gradients = th.stack([th.autograd.grad(output[:, i].sum(),
                                           input_data, retain_graph=True)[0]
                          for i in range(cols)], 2)

    gradients = gradients.cpu().numpy()
    tot_grad = compute_total_effect(graph,gradients, 5)

    direct_grad_abs = np.mean(np.abs(gradients), axis=0)
    total_grad_abs = np.mean(np.abs(tot_grad), axis=0)

    direct_gradient_matrix = np.mean(gradients, axis=0)
    direct_gradient_matrix_std = np.std(gradients, axis=0)

    total_gradient_matrix = np.mean(tot_grad, axis=0)
    total_gradient_matrix_std = np.std(tot_grad, axis=0)

    mean_vect = np.mean(raw_data.values, axis=0)
    std_vect = np.std(raw_data.values, axis=0)

    direct_grad_rs = direct_gradient_matrix * std_vect[np.newaxis, :] / std_vect[:, np.newaxis]
    total_grad_rs = total_gradient_matrix * std_vect[np.newaxis, :] / std_vect[:, np.newaxis]
    direct_elasticity_matrix = direct_grad_rs * mean_vect[:, np.newaxis] / mean_vect[np.newaxis, :]
    total_elasticity_matrix = total_grad_rs *mean_vect[:, np.newaxis] / mean_vect[np.newaxis, :] 

    return (graph, direct_grad_abs, total_grad_abs, direct_gradient_matrix,
            direct_gradient_matrix_std, total_gradient_matrix,
            total_gradient_matrix_std, direct_grad_rs, total_grad_rs,
            direct_elasticity_matrix, total_elasticity_matrix)


def exec_sam_instance(data, skeleton=None, mixed_data=False, gpus=0,
                      device='cpu', verbose=True, log=None,
                      lr=0.01, dlr=0.01, lambda1=0.001, lambda2=0.0000001, nh=200, dnh=500,
                      train=10000, test=1000, batchsize=-1,
                      losstype="fgan", functionalComplexity="numberHiddenUnits", sampletype="sigmoidproba", initweight=0,
                      kfactor=100, dagstart=0, dagloss=False,
                      dagpenalization=0.001, dagpenalization_increase=0.0,
                      use_filter=False, filter_threshold=0.5,
                      dag_threshold=0.5, eval_total_effect=False,
                      linear=False, numberHiddenLayersD=2, numberHiddenLayersG=1):

        out = run_SAM(data, skeleton=skeleton,
                      is_mixed=mixed_data, device=device,lr_gen=lr, lr_disc=dlr,
                      lambda1=lambda1, lambda2=lambda2,
                      nh=nh, dnh=dnh,
                      train=train,
                      test=test, batch_size=batchsize,
                      dagstart=dagstart,
                      dagloss=dagloss,
                      dagpenalization=dagpenalization,
                      dagpenalization_increase=dagpenalization_increase,
                      use_filter=use_filter,
                      filter_threshold=filter_threshold,
                      losstype=losstype,
                      functionalComplexity=functionalComplexity,
                      sampletype=sampletype,
                      initweight=initweight,
                      kfactor=kfactor,
                      dag_threshold=dag_threshold,
                      eval_total_effect=eval_total_effect,
                      linear=linear,
                      numberHiddenLayersD=numberHiddenLayersD,
                      numberHiddenLayersG=numberHiddenLayersG)

        #if log is not None:
        #    np.savetxt(log, out, delimiter=",")
        return out


class gSAM3d(object):
    """Structural Agnostic Model."""

    def __init__(self, lr=0.001, dlr=0.01, lambda1=0.001, lambda2=0.0000001, nh=200, dnh=500,
                 train_epochs=10000, test_epochs=1000, batchsize=-1,
                 losstype="fgan", functionalComplexity="numberHiddenUnits", sampletype="sigmoidproba", initweight=0, kfactor=100, dagstart=0, dagloss=False, dagpenalization=0.001,
                 dagpenalization_increase=0.0, use_filter=False, filter_threshold = 0.5, dag_threshold = 0.5, eval_total_effect = False, linear = False, numberHiddenLayersG = 1, numberHiddenLayersD = 2):

        """Init and parametrize the SAM model.

        :param lr: Learning rate of the generators
        :param dlr: Learning rate of the discriminator
        :param l1: L1 penalization on the causal filters
        :param nh: Number of hidden units in the generators' hidden layers
           a
           ((cols,cols)
        :param dnh: Number of hidden units in the discriminator's hidden layer$
        :param train_epochs: Number of training epochs
        :param test_epochs: Number of test epochs (saving and averaging the causal filters)
        :param batchsize: Size of the batches to be fed to the SAM model.
        """
        super(gSAM3d, self).__init__()
        self.lr = lr
        self.dlr = dlr
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.nh = nh
        self.dnh = dnh
        self.train = train_epochs
        self.test = test_epochs
        self.batchsize = batchsize
        self.losstype = losstype
        self.sampletype = sampletype
        self.dagstart = dagstart
        self.dagloss = dagloss
        self.dagpenalization = dagpenalization
        self.dagpenalization_increase = dagpenalization_increase
        self.use_filter = use_filter
        self.filter_threshold = filter_threshold
        self.dag_threshold = dag_threshold
        self.eval_total_effect = eval_total_effect
        self.linear = linear
        self.initweight = initweight
        self.kfactor = kfactor
        self.numberHiddenLayersD = numberHiddenLayersD
        self.numberHiddenLayersG = numberHiddenLayersG 
        self.functionalComplexity = functionalComplexity


    def predict(self, data, skeleton=None, mixed_data=False, nruns=6, njobs=1, 
                gpus=0, verbose=True, log=None):
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
            return exec_sam_instance(data, skeleton=skeleton,
                                     mixed_data=mixed_data,
                                     verbose=verbose, gpus=gpus)
        else:
            list_out = []
            if log is not None:
                idx = 0
                while os.path.isfile(log + str(idx)):
                    list_out.append(np.loadtxt(log + str(idx), delimiter=","))
                    idx += 1
            results = parallel_identical(run_SAM, data, skeleton=skeleton,
                                         nruns=nruns-len(list_out),
                                         njobs=njobs, gpus=gpus, lr_gen=self.lr,
                                         lr_disc=self.dlr,
                                         verbose=verbose,
                                         lambda1=self.lambda1, lambda2=self.lambda2,
                                         nh=self.nh, dnh=self.dnh,
                                         train=self.train,
                                         test=self.test, batch_size=self.batchsize,
                                         dagstart=self.dagstart,
                                         dagloss=self.dagloss,
                                         dagpenalization=self.dagpenalization,
                                         dagpenalization_increase=self.dagpenalization_increase,
                                         use_filter=self.use_filter,
                                         filter_threshold=self.filter_threshold,
                                         losstype=self.losstype,
                                         functionalComplexity=self.functionalComplexity,
                                         sampletype=self.sampletype,
                                         initweight=self.initweight,
                                         kfactor=self.kfactor,
                                         dag_threshold=self.dag_threshold,
                                         eval_total_effect=self.eval_total_effect,
                                         linear=self.linear,
                                         numberHiddenLayersD=self.numberHiddenLayersD,
                                         numberHiddenLayersG=self.numberHiddenLayersG)
            list_out.extend(results)

            

            if self.eval_total_effect:
                return [sum([run[i] for run in list_out])/nruns
                        for i in range(len(list_out[0]))]
            else:
                list_out = [i for i in list_out if not np.isnan(i).any()]
                
                #for idx, run in enumerate(list_out):
                #    np.savetxt("variance_analysis/score run " + str(idx) + "dag_" + str(self.dagloss) + ".csv", run)
                    

                try:
                    assert len(list_out) > 0
                except AssertionError as e:
                    print("All solutions contain NaNs")
                    raise(e)
                return sum(list_out)/len(list_out)



