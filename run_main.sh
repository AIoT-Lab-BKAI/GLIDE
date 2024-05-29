#!/bin/bash

# you have to cd to your workdir first
cd /vinserver_user/hung.nn184118/workspace/PG-causal/

# /home/admin/miniconda3/envs/easyFL/bin/python main.py --dataname "erdos_renyi/d20_p0.1" --folder "m5_d1.0_n10" --output "main_erdosrenyi_50_05.csv" --confidence '0.01' --gamma2 '0.5' --num_env 50 --max_markov_size 8 --causal_search_repeat 5
/home/admin/miniconda3/envs/easyFL/bin/python main.py --dataname "erdos_renyi/d40_p0.1" --folder "m5_d1.0_n10" --output "main_erdosrenyi_50_05.csv" --confidence '0.01' --gamma2 '0.5' --num_env 50 --max_markov_size 8 --causal_search_repeat 5
# /home/admin/miniconda3/envs/easyFL/bin/python main.py --dataname "erdos_renyi/d60_p0.1" --folder "m5_d1.0_n10" --output "main_erdosrenyi_50_05.csv" --confidence '0.01' --gamma2 '0.5' --num_env 50 --max_markov_size 8 --causal_search_repeat 5
# /home/admin/miniconda3/envs/easyFL/bin/python main.py --dataname "erdos_renyi/d80_p0.1" --folder "m5_d1.0_n10" --output "main_erdosrenyi_50_05.csv" --confidence '0.01' --gamma2 '0.5' --num_env 50 --max_markov_size 8 --causal_search_repeat 5



# /home/admin/miniconda3/envs/easyFL/bin/python main.py --dataname "asia" --folder "m3_d1_n10" --output "all_res_50_05.csv" --confidence '0.01' --gamma2 '0.5' --num_env 50 --max_markov_size 8 --causal_search_repeat 5
# /home/admin/miniconda3/envs/easyFL/bin/python main.py --dataname "survey" --folder "m3_d1_n10" --output "all_res_50_05.csv" --confidence '0.01' --gamma2 '0.5' --num_env 50 --max_markov_size 8 --causal_search_repeat 5
# /home/admin/miniconda3/envs/easyFL/bin/python main.py --dataname "cancer" --folder "m3_d1_n10" --output "all_res_50_05.csv" --confidence '0.01' --gamma2 '0.5' --num_env 50 --max_markov_size 8 --causal_search_repeat 5
# /home/admin/miniconda3/envs/easyFL/bin/python main.py --dataname "child" --folder "m3_d1_n10" --output "all_res_50_05.csv" --confidence '0.01' --gamma2 '0.5' --num_env 50 --max_markov_size 8 --causal_search_repeat 5
# /home/admin/miniconda3/envs/easyFL/bin/python main.py --dataname "sachs" --folder "m3_d1_n10" --output "all_res_50_05.csv" --confidence '0.01' --gamma2 '0.5' --num_env 50 --max_markov_size 8 --causal_search_repeat 5
# /home/admin/miniconda3/envs/easyFL/bin/python main.py --dataname "water" --folder "m3_d1_n10" --output "all_res_50_05.csv" --confidence '0.01' --gamma2 '0.5' --num_env 50 --max_markov_size 8 --causal_search_repeat 5
# /home/admin/miniconda3/envs/easyFL/bin/python main.py --dataname "alarm" --folder "m3_d1_n10" --output "all_res_50_05.csv" --confidence '0.01' --gamma2 '0.5' --num_env 50 --max_markov_size 8 --causal_search_repeat 5
# /home/admin/miniconda3/envs/easyFL/bin/python main.py --dataname "barley" --folder "m3_d1_n10" --output "all_res_50_05.csv" --confidence '0.01' --gamma2 '0.5' --num_env 50 --max_markov_size 8 --causal_search_repeat 5
# /home/admin/miniconda3/envs/easyFL/bin/python main.py --dataname "insurance" --folder "m3_d1_n10" --output "all_res_50_05.csv" --confidence '0.01' --gamma2 '0.5' --num_env 50 --max_markov_size 8 --causal_search_repeat 5
