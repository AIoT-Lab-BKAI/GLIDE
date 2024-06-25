#!/bin/bash

# you have to cd to your workdir first
cd /vinserver_user/hung.nn184118/workspace/CDICP/


# /home/admin/miniconda3/envs/easyFL/bin/python baseline.py --baseline "DAS" --dataname "notears"  --ntype "linear" --d 20 --output "res/baseline-notears-extreme.csv" --repeat 4
# /home/admin/miniconda3/envs/easyFL/bin/python baseline.py --baseline "DAS" --dataname "notears"  --ntype "linear" --d 700 --output "res/baseline-notears-extreme.csv" --repeat 5
# /home/admin/miniconda3/envs/easyFL/bin/python baseline.py --baseline "DAS" --dataname "notears"  --ntype "linear" --d 800 --output "res/baseline-notears-extreme.csv" --repeat 5
# /home/admin/miniconda3/envs/easyFL/bin/python baseline.py --baseline "DAS" --dataname "notears"  --ntype "linear" --d 900 --output "res/baseline-notears-extreme.csv" --repeat 5
# /home/admin/miniconda3/envs/easyFL/bin/python baseline.py --baseline "DAS" --dataname "notears"  --ntype "linear" --d 1000 --output "res/baseline-notears-extreme.csv" --repeat 5


# /home/admin/miniconda3/envs/easyFL/bin/python baseline.py --baseline "CDNOD" --dataname "sachs" --folder "m3_d1_n10" --output "res/baseline_generic.csv" 
# /home/admin/miniconda3/envs/easyFL/bin/python baseline.py --baseline "CDNOD" --dataname "insurance" --folder "m3_d1_n10" --output "res/baseline_generic.csv" 
# /home/admin/miniconda3/envs/easyFL/bin/python baseline.py --baseline "CDNOD" --dataname "water" --folder "m3_d1_n10" --output "res/baseline_generic.csv" 
# /home/admin/miniconda3/envs/easyFL/bin/python baseline.py --baseline "CDNOD" --dataname "alarm" --folder "m3_d1_n10" --output "res/baseline_generic.csv" 
# /home/admin/miniconda3/envs/easyFL/bin/python baseline.py --baseline "CDNOD" --dataname "barley" --folder "m3_d1_n10" --output "res/baseline_generic.csv" 
# /home/admin/miniconda3/envs/easyFL/bin/python baseline.py --baseline "CDNOD" --dataname "pathfinder" --folder "m3_d5_n10" --output "res/baseline_generic.csv" 
/home/admin/miniconda3/envs/easyFL/bin/python federated.py