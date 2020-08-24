#!/bin/bash

process=$1
shift
experiment_name=$1
cd $experiment_name
shift
python_script_name=$1
python_script_path="$python_script_name"
shift
python $python_script_path $process $*
