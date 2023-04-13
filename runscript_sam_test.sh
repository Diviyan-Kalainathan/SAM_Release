#!/bin/bash

#python main.py 'dream4/insilico_size100_1_multifactorial_data.tsv' 'dream4/insilico_size100_1_multifactorial_target.tsv' --nruns 32  --njobs 16 --gpus 2 --tsv --log 'train_graphs/results'
#python main.py 'dream4/insilico_size100_2_multifactorial_data.tsv' 'dream4/insilico_size100_2_multifactorial_target.tsv' --nruns 32  --njobs 16 --gpus 2 --tsv --log 'train_graphs/results'
#python main.py 'dream4/insilico_size100_3_multifactorial_data.tsv' 'dream4/insilico_size100_3_multifactorial_target.tsv' --nruns 32  --njobs 16 --gpus 2 --tsv --log 'train_graphs/results'
#python main.py 'dream4/insilico_size100_4_multifactorial_data.tsv' 'dream4/insilico_size100_4_multifactorial_target.tsv' --nruns 32  --njobs 16 --gpus 2 --tsv --log 'train_graphs/results'
#python main.py 'dream4/insilico_size100_5_multifactorial_data.tsv' 'dream4/insilico_size100_5_multifactorial_target.tsv' --nruns 32  --njobs 16 --gpus 2 --tsv --log 'train_graphs/results'


#python main.py 'train_graphs/syntrenHop1_20_1_data.csv' 'train_graphs/syntrenHop1_20_1_target.csv' --nruns 16  --njobs 16 --gpus 2 --dag --log 'train_graphs/results'

#python main.py 'train_graphs/insilico_size100_1_multifactorial_data.tsv' 'train_graphs/insilico_size100_1_multifactorial_target.tsv' --nruns 16  --njobs 16 --gpus 2 --tsv --log 'train_graphs/results'


python main.py 'synthetic_graphs/gp_add/20/gp_add-20-0_data.csv' 'synthetic_graphs/gp_add/20/gp_add-20-0_target.csv' --nruns 48 --njobs 16 --gpus 2 --dag --adjmatrix --header --log 'variance_analysis'

#python main.py 'synthetic_graphs/sigmoid_add/20/sigmoid_add-20-0_data.csv' 'synthetic_graphs/sigmoid_add/20/sigmoid_add-20-0_target.csv' --nruns 16  --njobs 16 --gpus 2 --dag --adjmatrix --header
#python main.py 'synthetic_graphs/sigmoid_mix/20/sigmoid_mix-20-0_data.csv' 'synthetic_graphs/sigmoid_mix/20/sigmoid_mix-20-0_target.csv' --nruns 16  --njobs 16 --gpus 2 --dag --adjmatrix --header
#python main.py 'synthetic_graphs/NN/20/NN-20-0_data.csv' 'synthetic_graphs/NN/20/NN-20-0_target.csv' --nruns 16  --njobs 16 --gpus 2 --dag --adjmatrix --header
#python main.py 'synthetic_graphs/gp_mix/20/gp_mix-20-0_data.csv' 'synthetic_graphs/gp_mix/20/gp_mix-20-0_target.csv' --nruns 16  --njobs 16 --gpus 2 --dag --adjmatrix --header

