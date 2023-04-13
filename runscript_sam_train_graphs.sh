#!/bin/bash


python main.py 'train_graphs/gp_add-train-100_data.csv' 'train_graphs/gp_add-train-100_target.csv' --nruns 2  --njobs 2 --gpus 1 --dag  --adjmatrix --header --log 'train_graphs/results'


#python main.py 'train_graphs/gp_mix-train-20_data.csv' 'train_graphs/gp_mix-train-20_target.csv' --nruns 4  --njobs 4 --gpus 1 --adjmatrix --header --log 'train_graphs/results'

#python main.py 'train_graphs/syntrenHop1_20_train_data.csv' 'train_graphs/syntrenHop1_20_train_target.csv' --nruns 48  --njobs 16 --gpus 2 --dag --log 'train_graphs/results'

#python main.py 'train_graphs/gp_add-train-20_data.csv' 'train_graphs/gp_add-train-20_target.csv' --nruns 48  --njobs 16 --gpus 2 --dag --adjmatrix --header --log 'train_graphs/results'





#python main.py 'train_graphs/NN-train-20_data.csv' 'train_graphs/NN-train-20_target.csv' --nruns 48  --njobs 16 --gpus 2 --dag --adjmatrix --header --log 'train_graphs/results'
#python main.py 'train_graphs/sigmoid_add-train-20_data.csv' 'train_graphs/sigmoid_add-train-20_target.csv' --nruns 48  --njobs 16 --gpus 2 --dag --adjmatrix --header --log 'train_graphs/results'
#python main.py 'train_graphs/sigmoid_mix-train-20_data.csv' 'train_graphs/sigmoid_mix-train-20_target.csv' --nruns 2  --njobs 2 --gpus 1 --dag --adjmatrix --header --log 'train_graphs/results'
#python main.py 'train_graphs/linear-train-20_data.csv' 'train_graphs/linear-train-20_target.csv' --nruns 48  --njobs 16 --gpus 2 --dag --adjmatrix --header --log 'train_graphs/results'





