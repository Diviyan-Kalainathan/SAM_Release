#!/usr/bin/env python

import pandas as pd
import random
from sklearn.metrics import average_precision_score
from code.sam_special_version_dream5 import SAMPlus
import numpy as np
from itertools import product
from copy import deepcopy
import argparse
from datetime import datetime
from random import sample 
   
def parameter_set(params, ex_rules=None, custom_ex_rules=None):
    full_set = [dict(zip(params, x)) for x in product(*params.values())]
    if ex_rules is not None:
        for param in full_set:
            for rule in ex_rules:
                if not param[rule]:
                    for excluded in ex_rules[rule]:
                      param.pop(excluded)
    if custom_ex_rules is not None:
        for param in full_set:
            for rule in custom_ex_rules:
                if param[rule] == custom_ex_rules[rule][0]:
                    for excluded in custom_ex_rules[rule][1]:
                        param.pop(excluded)

    # Exclude redudant param sets
    full_set = [dict(s) for s in set(frozenset(d.items()) for d in full_set)]
    # full_set = list(np.unique(np.array(full_set)))
    return full_set


parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('data', metavar='d', type=str, help='data')
parser.add_argument('features', metavar='t', type=str, help='features')
parser.add_argument('goldStandard', metavar='t', type=str, help='goldStandard')
parser.add_argument('--nruns', metavar='j', type=int, help="num of runs", default=-1)
parser.add_argument('--gpus', help="Use gpu", type=int, default=0)
parser.add_argument('--njobs', metavar='j', type=int,  help="num of jobs", default=-1)
parser.add_argument('--numberHiddenLayersD', metavar='t', type=str, help='numberHiddenLayersD')
parser.add_argument('--tsv', help="TSV file", action='store_true')    
parser.add_argument('--nv', help="No verbose", action='store_true')  

args = parser.parse_args()   

if args.tsv:
    sep = "\t"
else:
    sep = ","
       
if args.nv:
    verbose = False
else:
    verbose = True

features = pd.read_csv(args.features, sep = sep)
list_features = features["features"].tolist()

dataset = pd.read_csv(args.data, sep = sep)
list_targets = list(dataset.columns)



goldStandard = np.zeros((len(list_features), len(list_targets)))

df = pd.read_csv(args.goldStandard, sep=sep)

for idx, row in df.iterrows():
    if len(row) < 3 or int(row[2]):
        if((row[0] in list_features) and (row[1] in list_targets)):
            goldStandard[list_features.index(row[0]), list_targets.index(row[1])] = 1


print(goldStandard)

df_goldStandard = pd.DataFrame(data = goldStandard, index=list_features, columns=list_targets)




gpus = int(args.gpus)
njobs = int(args.njobs)
numberHiddenLayersD = int(args.numberHiddenLayersD)


nb_targets = len(list_features)


nb_run = int(int(args.nruns)*len(list_targets)/nb_targets)
print("nb_run " + str(nb_run))

parameters={'lr':[0.01],
            'dlr':[0.001],
            'lambda1': [10],
            'lambda1_increase': [0.0],
            'lambda2':[0.001],
            'lambda2_increase':[0.0],
            'startRegul':[0],
            'nh':[20],
            'numberHiddenLayersG':[2],
            'complexPen':[True],
            'dnh': [200],
            'losstype':["fgan"],
            'train_epochs':[3000],
            'test_epochs':[1000],
            'bootstrap_ratio':[1],
            'linear':[False]}
      

   


           
rules_exclusive = {}
rules_exclusive_custom = {}
        
paramset = parameter_set(parameters, rules_exclusive, rules_exclusive_custom)


order = []
p = paramset[0]
for key in sorted(p.keys()):
    order.append(key)

order.append("nb_run")
order.append("aupr")

freport = pd.DataFrame(columns=order)


dateTime = datetime.now().isoformat()

for p in paramset:

    sam = SAMPlus(**p, numberHiddenLayersD=numberHiddenLayersD, njobs=njobs, gpus=gpus, verbose=verbose, nruns=nb_run, )

    df_results, df_cpt, _ = sam.predict(dataset, list_features, list_targets, nb_targets=nb_targets)
 
    df_results = df_results.sort_index(axis=0)
    df_goldStandard = df_goldStandard.sort_index(axis=0)

    df_results = df_results.sort_index(axis=1)
    df_goldStandard = df_goldStandard.sort_index(axis=1)
        
    AUPR = average_precision_score(df_goldStandard.values.ravel(), df_results.values.ravel())
    
    fileResult = args.data.split('/')[0] + "/results/result_aupr_"
    fileCpt = args.data.split('/')[0] + "/results/cpt_aupr_"
    
    fileResult+= str(AUPR)
    fileCpt+= str(AUPR)
    
    line = []
    
    for key in sorted(p.keys()):
        line.append(p[key])
    
    line.append(nb_run)
    line.append(AUPR)
    
    freport.loc[freport.shape[0]+1] = line
    
    freport.to_csv(args.data.split('/')[0] + "/reports/reports_" + str(dateTime) + "_" + args.data.split('/')[1])
    
    fileResult+= "_" + str(dateTime) + ".csv"
    fileCpt+= "_" + str(dateTime) + ".csv"
    
    df_results.to_csv(fileResult)
    df_cpt.to_csv(fileCpt)
    






                   
                  

    
