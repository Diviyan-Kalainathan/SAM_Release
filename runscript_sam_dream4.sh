#!/bin/bash

python main.py 'dream4/insilico_size100_1_multifactorial_data.tsv' 'dream4/insilico_size100_1_multifactorial_target.tsv' --nruns 16  --njobs 8 --gpus 4 --tsv
python main.py 'dream4/insilico_size100_2_multifactorial_data.tsv' 'dream4/insilico_size100_1_multifactorial_target.tsv' --nruns 16  --njobs 8 --gpus 4 --tsv
python main.py 'dream4/insilico_size100_3_multifactorial_data.tsv' 'dream4/insilico_size100_1_multifactorial_target.tsv' --nruns 16  --njobs 8 --gpus 4 --tsv
python main.py 'dream4/insilico_size100_4_multifactorial_data.tsv' 'dream4/insilico_size100_1_multifactorial_target.tsv' --nruns 16  --njobs 8 --gpus 4 --tsv
python main.py 'dream4/insilico_size100_5_multifactorial_data.tsv' 'dream4/insilico_size100_1_multifactorial_target.tsv' --nruns 16  --njobs 8 --gpus 4 --tsv
