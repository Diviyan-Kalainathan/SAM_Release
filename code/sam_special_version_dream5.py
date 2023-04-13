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
from joblib import Parallel, delayed
from sklearn.preprocessing import scale
from .utils.linear3d import Linear3D
from .utils.graph import SimpleMatrixConnection, MatrixSampler, MatrixSampler3, notears_constr
from .utils.batchnorm import ChannelBatchNorm1d
from .utils.batchnorm import ParallelBatchNorm1d
from .utils.treillis import compute_total_effect
import time


import os
import re
from collections import namedtuple
from subprocess import run, PIPE
from typing import Sequence, List, Optional
from parallel import parallel_run

import networkx as nx
import random
from random import sample


class SAM_generators(th.nn.Module):
    """Ensemble of all the generators."""

    def __init__(self, batch_size, nb_features, nb_targets, nh,  linear=False, numberHiddenLayersG=1):
        """Init the model."""
        super(SAM_generators, self).__init__()
        layers = []
        self.linear = linear
        if linear:
            self.input_layer = Linear3D(nb_targets, nb_features, 1, noise=True, batch_size=batch_size)
        else:
            self.input_layer = Linear3D(nb_targets, nb_features, nh, noise=True, batch_size=batch_size)
            layers.append(ChannelBatchNorm1d(nb_targets, nh))
            layers.append(th.nn.Tanh())

            for i in range(numberHiddenLayersG-1):
                layers.append(Linear3D(nb_targets, nh, nh))
                layers.append(ChannelBatchNorm1d(nb_targets, nh))
                layers.append(th.nn.Tanh())

            self.output_layer = Linear3D(nb_targets, nh, 1)
            self.layers = th.nn.Sequential(*layers)

    def forward(self, data, adj_matrix, drawn_neurons=None):
        """Forward through all the generators."""
        if self.linear:
            output = self.input_layer(data, adj_matrix)
        else:
            output = self.output_layer(self.layers(self.input_layer(data, adj_matrix)), drawn_neurons)

        return output.squeeze(2)

    def reset_parameters(self):
        if not self.linear:
            self.output_layer.reset_parameters()
            for layer in self.layers:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
        self.input_layer.reset_parameters()


class SAM_discriminator(th.nn.Module):
    """SAM discriminator."""

    def __init__(self, nb_variables, nb_targets, dnh, numberHiddenLayersD = 2, mask=None):
        super(SAM_discriminator, self).__init__()

        layers = []
        layers.append(th.nn.Linear(nb_variables, dnh))
        layers.append(ParallelBatchNorm1d(dnh))
        layers.append(th.nn.LeakyReLU(.2))

        for i in range(numberHiddenLayersD-1):
            layers.append(th.nn.Linear(dnh, dnh))
       	    layers.append(ParallelBatchNorm1d(dnh))
            layers.append(th.nn.LeakyReLU(.2))
 
        layers.append(th.nn.Linear(dnh, 1))
        self.layers = th.nn.Sequential(*layers)

        if mask is None:
            mask = th.eye(nb_targets, nb_targets)
            
        self.register_buffer("mask", mask.unsqueeze(0))

    def forward(self, input, obs_targets_data=None, obs_features_data=None, ):
    
        if (obs_targets_data is not None and obs_features_data is not None):
            return self.layers(th.cat((obs_features_data, obs_targets_data.unsqueeze(1) * (1 - self.mask) + input.unsqueeze(1) * self.mask),2))
        elif (obs_targets_data is not None):
            return self.layers(obs_targets_data.unsqueeze(1) * (1 - self.mask) + input.unsqueeze(1) * self.mask)
        else:
            return self.layers(input)

    def reset_parameters(self):
        for layer in self.layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
                

def run_SAM(in_data, list_features, list_targets,  nb_targets, skeleton=None, is_mixed=False, device="cpu",
            train=10000, test=1,
            batch_size=-1, lr_gen=.001,
            lr_disc=.01, lambda1=0.001, lambda1_increase=0.000001,  lambda2=0.0000001, lambda2_increase =0.0000001, startRegul = 1000, nh=None, dnh=None,
            verbose=True, losstype="fgan", linear=False, numberHiddenLayersD = 2,complexPen=False, bootstrap_ratio=0.8, datasetName=None, numberHiddenLayersG=2, idx=0):



    
    print(idx)

    nb_points = int(in_data.shape[0]*bootstrap_ratio)
    p = np.random.permutation(in_data.shape[0])
    data =  scale(in_data.values[p[:int(nb_points)], :])
    in_data = pd.DataFrame(data = data, columns = in_data.columns)


    lambda1 = lambda1/nb_points
    lambda2 = lambda2/nb_points
    
    
    sample_targets = random.sample(list_targets,k=nb_targets)
    if(list_features == None):
        list_features = sample_targets.copy()

    sample_features_only = list_features.copy()

    for node in sample_targets:
        if(node in sample_features_only):
            sample_features_only.remove(node)


    data_features = in_data[list_features].values
    data_targets = in_data[sample_targets].values


    
    if (len(sample_features_only) > 0):
        data_features_only = in_data[sample_features_only].values



    nb_features = len(list_features)
    nb_targets = len(sample_targets)
    nb_variables = len(sample_features_only) + len(sample_targets)

    data_targets = data_targets.astype('float32')
    data_targets = th.from_numpy(data_targets).to(device)

    if (len(sample_features_only) > 0):
        data_features_only = data_features_only.astype('float32')
        data_features_only = th.from_numpy(data_features_only).to(device)
        data_features_only_duplicated = data_features_only.unsqueeze(1).repeat(1, nb_targets,1)
        
    data_features = data_features.astype('float32')
    data_features = th.from_numpy(data_features).to(device)

    

    if (len(sample_features_only) > 0):
        data = th.cat((data_features_only,data_targets),1)
    else:
        data = data_targets

    if batch_size == -1:
        batch_size = data.shape[0]

    sam = SAM_generators(batch_size, nb_features, nb_targets, nh, linear=linear, numberHiddenLayersG=numberHiddenLayersG).to(device)
    sam.reset_parameters()
    g_optimizer = th.optim.Adam(list(sam.parameters()), lr=lr_gen)

    if losstype != "mse":
        discriminator = SAM_discriminator(nb_variables, nb_targets, dnh, numberHiddenLayersD,).to(device)
        discriminator.reset_parameters()
        d_optimizer = th.optim.Adam(discriminator.parameters(), lr=lr_disc)
        criterion = th.nn.BCEWithLogitsLoss()
    else:
        criterion = th.nn.MSELoss()
        disc_loss = th.zeros(1)


    mask = np.ones((nb_features,nb_targets))
    for i, node_i in enumerate(list_features):
        for j, node_j in enumerate(sample_targets):
            if(node_i==node_j):
                mask[i,j] = 0;

    mask = th.from_numpy(mask.astype('float32'))

    graph_sampler = MatrixSampler((nb_features, nb_targets), mask=mask, gumble=False).to(device)

    graph_sampler.weights.data.fill_(2)
    graph_optimizer = th.optim.Adam(graph_sampler.parameters(), lr=lr_gen)

    _true = th.ones(1).to(device)
    _false = th.zeros(1).to(device)

    output = th.zeros(nb_features, nb_targets).to(device)

    # RUN
    if verbose:
        pbar = tqdm(range(train + test))
    else:
        pbar = range(train+test)

    for epoch in pbar:

        g_optimizer.zero_grad()
        graph_optimizer.zero_grad()

        if losstype != "mse":
            d_optimizer.zero_grad()



        # Train the discriminator

        drawn_graph = graph_sampler()


        generated_variables = sam(data_features, drawn_graph)

        if losstype == "mse":
            gen_loss = criterion(generated_variables, data_targets)
        else:
            if (len(sample_features_only) > 0):
                disc_vars_d = discriminator(generated_variables.detach(), data_targets, data_features_only_duplicated )
                disc_vars_g = discriminator(generated_variables, data_targets, data_features_only_duplicated )
            else:
                disc_vars_d = discriminator(generated_variables.detach(), data_targets)
                disc_vars_g = discriminator(generated_variables, data_targets)
            
            true_vars_disc = discriminator(data)


            if losstype == "gan":

                disc_loss = criterion(disc_vars_d,_false.expand_as(disc_vars_d)) + criterion(true_vars_disc, _true.expand_as(true_vars_disc))
                gen_loss  = criterion(disc_vars_g, _true.expand_as(disc_vars_g))


            elif losstype == "fgan":

                disc_loss = th.mean(th.exp(disc_vars_d - 1), [0, 2]).sum() / nb_targets - th.mean(true_vars_disc)
                gen_loss = -th.mean(th.exp(disc_vars_g - 1), [0, 2]).sum()


            disc_loss.backward()
            d_optimizer.step()

        filters = graph_sampler.get_proba()

        

        struc_loss = (lambda1 + (epoch - startRegul) * lambda1_increase)*drawn_graph.sum()

        l2_reg = th.tensor(0.).to(device)
        for param in sam.parameters():
            l2_reg += th.norm(param)

        func_loss = (lambda2 + (epoch - startRegul) * lambda2_increase)*l2_reg

        regul_loss = struc_loss + func_loss


        loss = gen_loss + regul_loss

        if verbose and epoch % 20 == 0 :
        
            if(regul_loss!=0):
                regul_loss = regul_loss.item()
                
            pbar.set_postfix(gen=gen_loss.item()/nb_targets,
                             disc=disc_loss.item(),
                             regul_loss=regul_loss,
                             tot=loss.item())

        if epoch < train + test - 1:
            loss.backward(retain_graph=True)

        if epoch >= train:
            output.add_(filters.data)

        g_optimizer.step()
        graph_optimizer.step()


    
    df_result = pd.DataFrame(data=output.div_(test).cpu().numpy(), index=list_features, columns=sample_targets)
           
    return df_result



class SAMPlus(object):
    """Structural Agnostic Model."""

    def __init__(self, lr=0.001, dlr=0.01, lambda1=0.001, lambda1_increase=0.000001, lambda2=0.0000001, lambda2_increase =0.0000001, startRegul = 1000, nh=200, dnh=500,
                 train_epochs=10000, test_epochs=1000, losstype="fgan",  kfactor=100, linear = False, numberHiddenLayersD = 2, numberHiddenLayersG = 2, complexPen = False, njobs=1, gpus=0, verbose=True, nruns=8,bootstrap_ratio=0.8):




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
        """
        super(SAMPlus, self).__init__()
        self.lr = lr
        self.dlr = dlr
        self.lambda1 = lambda1
        self.lambda1_increase = lambda1_increase
        self.lambda2 = lambda2
        self.lambda2_increase = lambda2_increase
        self.startRegul = startRegul
        self.nh = nh
        self.dnh = dnh
        self.train = train_epochs
        self.test = test_epochs
        self.losstype = losstype
        self.linear = linear
        self.numberHiddenLayersD = numberHiddenLayersD
        self.complexPen = complexPen
        self.njobs = njobs
        self.gpus = gpus
        self.verbose = verbose
        self.nruns = nruns
        self.bootstrap_ratio = bootstrap_ratio
        self.numberHiddenLayersG = numberHiddenLayersG

        
        
    def predict(self, data, list_features, list_targets, nb_targets, datasetName=None, return_list_results=False):
    
        """Execute SAM on a dataset given a skeleton or not.

        Args:
            data (pandas.DataFrame): Observational data for estimation of causal relationships by SAM
            skeleton (numpy.ndarray): A priori knowledge about the causal relationships as an adjacency matrix.
                      Can be fed either directed or undirected links.
        Returns:
            networkx.DiGraph: Graph estimated by SAM, where A[i,j] is the term
            of the ith variable for the jth generator.
        """

        assert self.nruns > 0
        if self.gpus == 0:
            results = [run_SAM(data, list_features, list_targets, 
                               nb_targets=nb_targets, 
                               lr_gen=self.lr,
                               lr_disc=self.dlr,
                               verbose=self.verbose,
                               lambda1=self.lambda1,
                               lambda1_increase=self.lambda1_increase,
                               lambda2=self.lambda2,
                               lambda2_increase=self.lambda2_increase,
                               startRegul=self.startRegul,
                               nh=self.nh, dnh=self.dnh,
                               train=self.train,
                               test=self.test,
                               losstype=self.losstype,
                               linear=self.linear,
                               numberHiddenLayersD=self.numberHiddenLayersD, 
                               numberHiddenLayersG=self.numberHiddenLayersG,
                               complexPen=self.complexPen, 
                               bootstrap_ratio=self.bootstrap_ratio, datasetName=datasetName,
                               device='cpu') for i in range(self.nruns)]
        else:
            results = parallel_run(run_SAM, data, list_features, list_targets,
                                   nb_targets=nb_targets,            
                                   nruns=self.nruns,
                                   njobs=self.njobs, gpus=self.gpus, lr_gen=self.lr,
                                   lr_disc=self.dlr,
                                   verbose=self.verbose,
                                   lambda1=self.lambda1,
                                   lambda1_increase=self.lambda1_increase,
                                   lambda2=self.lambda2,
                                   lambda2_increase=self.lambda2_increase,
                                   startRegul=self.startRegul,
                                   nh=self.nh, dnh=self.dnh,
                                   train=self.train,
                                   test=self.test, 
                                   losstype=self.losstype,
                                   linear=self.linear,
                                   numberHiddenLayersD=self.numberHiddenLayersD,
                                   numberHiddenLayersG=self.numberHiddenLayersG,
                                   complexPen=self.complexPen,
                                   bootstrap_ratio=self.bootstrap_ratio, datasetName=datasetName)
        
        if(list_features==None):
            list_features = list_targets
            
        df_score = pd.DataFrame(0.0, index = list_features, columns = list_targets)
        df_cpt = pd.DataFrame(0.0, index = list_features, columns = list_targets)
        
        for df in results:
            if not np.isnan(df.values).any():
                df_score = df_score.add(df, fill_value=0)
                df2 = pd.DataFrame(1.0,  index =df.index, columns = df.columns)
                df_cpt = df_cpt.add(df2, fill_value=0)
        
        df_results = df_score.div(df_cpt,fill_value=1)  
        df_results = df_results.fillna(value=0.0)
        
        return df_results, df_cpt, df_score
        




