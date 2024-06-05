#!/bin/bash

# you have to cd to your workdir first
cd /vinserver_user/hung.nn184118/workspace/CDICP/

# /home/admin/miniconda3/envs/easyFL/bin/python proposal.py --dataname "erdos_renyi/d20_p0.1" --folder "m5_d1.0_n10" --output "main_erdosrenyi_50_05.csv" --gamma2 '0.5' --num_env 50 --capsize 8 --mode "aS"
# /home/admin/miniconda3/envs/easyFL/bin/python proposal.py --dataname "erdos_renyi/d40_p0.1" --folder "m5_d1.0_n10" --output "main_erdosrenyi_50_05.csv" --gamma2 '0.5' --num_env 50 --capsize 8 --mode "aS"
# /home/admin/miniconda3/envs/easyFL/bin/python proposal.py --dataname "erdos_renyi/d60_p0.1" --folder "m5_d1.0_n10" --output "main_erdosrenyi_50_05.csv" --gamma2 '0.5' --num_env 50 --capsize 8 --mode "aS"
# /home/admin/miniconda3/envs/easyFL/bin/python proposal.py --dataname "erdos_renyi/d80_p0.1" --folder "m5_d1.0_n10" --output "main_erdosrenyi_50_05.csv" --gamma2 '0.5' --num_env 50 --capsize 8 --mode "aS"



/home/admin/miniconda3/envs/easyFL/bin/python proposal.py --dataname "asia" --folder "m3_d1_n10" --output "proposal.csv" --gamma2 '0.5' --num_env 10 --TMB 1 --capsize 8 --mode "aS"
/home/admin/miniconda3/envs/easyFL/bin/python proposal.py --dataname "asia" --folder "m3_d1_n10" --output "proposal.csv" --gamma2 '0.5' --num_env 10 --TMB 1 --capsize 8 --mode "aL"
# /home/admin/miniconda3/envs/easyFL/bin/python proposal.py --dataname "asia" --folder "m3_d1_n10" --output "proposal.csv" --gamma2 '0.5' --num_env 10 --TMB 1 --capsize 8 --mode "aS"
# /home/admin/miniconda3/envs/easyFL/bin/python proposal.py --dataname "survey" --folder "m3_d1_n10" --output "proposal.csv" --gamma2 '0.5' --num_env 50 --capsize 8 --mode "aS"
# /home/admin/miniconda3/envs/easyFL/bin/python proposal.py --dataname "cancer" --folder "m3_d1_n10" --output "proposal.csv" --gamma2 '0.5' --num_env 50 --capsize 8 --mode "aS"
# /home/admin/miniconda3/envs/easyFL/bin/python proposal.py --dataname "child" --folder "m3_d1_n10" --output "proposal.csv" --gamma2 '0.5' --num_env 50 --capsize 8 --mode "aS"
# /home/admin/miniconda3/envs/easyFL/bin/python proposal.py --dataname "sachs" --folder "m3_d1_n10" --output "proposal.csv" --gamma2 '0.5' --num_env 50 --capsize 8 --mode "aS"
# /home/admin/miniconda3/envs/easyFL/bin/python proposal.py --dataname "water" --folder "m3_d1_n10" --output "proposal.csv" --gamma2 '0.5' --num_env 50 --capsize 8 --mode "aS"
# /home/admin/miniconda3/envs/easyFL/bin/python proposal.py --dataname "alarm" --folder "m3_d1_n10" --output "proposal.csv" --gamma2 '0.5' --num_env 50 --capsize 8 --mode "aS"
# /home/admin/miniconda3/envs/easyFL/bin/python proposal.py --dataname "barley" --folder "m3_d1_n10" --output "proposal.csv" --gamma2 '0.5' --num_env 50 --capsize 8 --mode "aS"
# /home/admin/miniconda3/envs/easyFL/bin/python proposal.py --dataname "insurance" --folder "m3_d1_n10" --output "proposal.csv" --gamma2 '0.5' --num_env 50 --capsize 8 --mode "aS"
