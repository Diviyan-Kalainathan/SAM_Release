#!/bin/bash


python main_dream5.py dream5/net1_expression_data.tsv dream5/net1_transcription_factors.tsv dream5/DREAM5_NetworkInference_GoldStandard_Network1.tsv --nruns 2 --njobs 2 --gpus 2 --numberHiddenLayersD 2 --tsv

