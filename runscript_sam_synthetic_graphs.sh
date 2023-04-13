#!/bin/bash


for folder in `find $1 -maxdepth 1 -type d`
do
for file in $folder/20/*data*
do
    fname=${file/data/log}
    mkdir -p ${fname/.csv/}
    python main.py $file ${file/data/target} "--nv" "--log" ${fname/.csv/} "${@:2}"        
done
done
