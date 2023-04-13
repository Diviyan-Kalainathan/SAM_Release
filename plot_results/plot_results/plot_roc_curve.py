import pandas as pd
from sklearn.metrics import roc_auc_score
import numpy as np
import random
import sklearn
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
#import seaborn
#from sklearn.utils.fixes import signature
from sklearn.metrics import average_precision_score

dataset = "cyto_test"
data_file = "data/cyto_full_data.csv"
target_file = "data/cyto_full_target.csv"
algos = [ "MMPC","CAM","SAM","PC-G", "PC-RCIT",  "PC-RCoT", "GES", "GIES", "CCDr" ]
datasetName = "Cyto"


print(sklearn.__version__)
#
# dataset = "syntren_20_test"
# data_file = "data/syntrenHop1_20_0_data.csv"
# target_file = "data/syntrenHop1_20_0_target.csv"
# datasetName = "Syntren 20 nodes"
#
# algos = [ "MMPC","CAM","SAM","PC-G", "PC-RCIT",  "PC-RCoT", "GES", "GIES", "CCDr" ]


# dataset = "syntren_100_test"
# data_file = "data/syntrenHop1_100_5_data.csv"
# target_file = "data/syntrenHop1_100_5_target.csv"
# datasetName = "Syntren 100 nodes"
#
# algos = [ "MMPC","CAM","SAM","PC-G", "PC-RCIT",  "PC-RCoT", "GES", "GIES", "CCDr" ]


data = pd.read_csv(data_file, sep=",")
tardata = pd.read_csv(target_file, sep=",")


plt.plot([0, 1], [0, 1], 'k--')


for idx, algo in enumerate(algos) :


    matrix_pred = np.loadtxt("results/Matrix-result-" + algo + "-" + dataset + "_.csv", delimiter = ",")

    #matrix_pred = np.loadtxt("results/Matrix-result-" + algo + "-cyto_test_.csv", delimiter=",")
    print(algo)
    print(matrix_pred.shape)
    y_scores = np.ravel(matrix_pred)


    target = np.zeros((len(data.columns), len(data.columns)))
    lstcols = list(data.columns.values)

    for idx, row in tardata.iterrows():
        if len(row) < 3 or int(row[2]):
            target[lstcols.index(row[0]), lstcols.index(row[1])] = 1



    y_true = np.ravel(target)


    average_precision = average_precision_score(y_true, y_scores)

    print("avg precision score " + str(average_precision))

    precision, recall, _ = precision_recall_curve(y_true, y_scores)

    # if algo == "bivariateFit":
    #     algo = "Best mse"
    # if algo == "Lingam":
    #     algo = "LiNGAM"
    # if algo == "reci_mono3":
    #     algo = "RECI"

    #step_kwargs = ({'step': 'post'}
    #               if 'step' in signature(plt.fill_between).parameters
    #               else {})

    # if(algo == "SAM"):
    #     plt.step(recall, precision, color = "g", where = "post", label=algo)
    # elif(algo =="PC-RCoT"):
    #     plt.step(recall, precision, color="y", where="post", label=algo)
    # else:
    plt.step(recall, precision, where="post", label=algo)


    #plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)



    plt.legend()



plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title(datasetName)
plt.show()

