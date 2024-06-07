#!/bin/bash

# you have to cd to your workdir first
cd /vinserver_user/hung.nn184118/workspace/CDICP/

/home/admin/miniconda3/envs/easyFL/bin/python baseline.py --baseline "PC" --dataname "hailfinder" --folder "m3_d5_n10" --output "baseline_res.csv" 
/home/admin/miniconda3/envs/easyFL/bin/python baseline.py --baseline "PC" --dataname "pathfinder" --folder "m3_d5_n10" --output "baseline_res.csv" 
/home/admin/miniconda3/envs/easyFL/bin/python baseline.py --baseline "PC" --dataname "munin1" --folder "m3_d5_n10" --output "baseline_res.csv" 


# /home/admin/miniconda3/envs/easyFL/bin/python baseline.py --baseline "PC" --dataname "erdos_renyi/d20_p0.11" --folder "m5_d1.0_n10" --output "baseline_res.csv" 
# /home/admin/miniconda3/envs/easyFL/bin/python baseline.py --baseline "PC" --dataname "erdos_renyi/d20_p0.16" --folder "m5_d1.0_n10" --output "baseline_res.csv" 
# /home/admin/miniconda3/envs/easyFL/bin/python baseline.py --baseline "PC" --dataname "erdos_renyi/d20_p0.22" --folder "m5_d1.0_n10" --output "baseline_res.csv" 

# /home/admin/miniconda3/envs/easyFL/bin/python baseline.py --baseline "PC" --dataname "erdos_renyi/d40_p0.05" --folder "m5_d1.0_n10" --output "baseline_res.csv" 
# /home/admin/miniconda3/envs/easyFL/bin/python baseline.py --baseline "PC" --dataname "erdos_renyi/d40_p0.07" --folder "m5_d1.0_n10" --output "baseline_res.csv" 
# /home/admin/miniconda3/envs/easyFL/bin/python baseline.py --baseline "PC" --dataname "erdos_renyi/d40_p0.1" --folder "m5_d1.0_n10" --output "baseline_res.csv" 
# 
# /home/admin/miniconda3/envs/easyFL/bin/python baseline.py --baseline "PC" --dataname "erdos_renyi/d60_p0.03" --folder "m5_d1.0_n10" --output "baseline_res.csv" 
# /home/admin/miniconda3/envs/easyFL/bin/python baseline.py --baseline "PC" --dataname "erdos_renyi/d60_p0.05" --folder "m5_d1.0_n10" --output "baseline_res.csv" 
# /home/admin/miniconda3/envs/easyFL/bin/python baseline.py --baseline "PC" --dataname "erdos_renyi/d60_p0.07" --folder "m5_d1.0_n10" --output "baseline_res.csv" 

# /home/admin/miniconda3/envs/easyFL/bin/python baseline.py --baseline "PC" --dataname "erdos_renyi/d80_p0.03" --folder "m5_d1.0_n10" --output "baseline_res.csv" 
# /home/admin/miniconda3/envs/easyFL/bin/python baseline.py --baseline "PC" --dataname "erdos_renyi/d80_p0.04" --folder "m5_d1.0_n10" --output "baseline_res.csv" 
# /home/admin/miniconda3/envs/easyFL/bin/python baseline.py --baseline "PC" --dataname "erdos_renyi/d80_p0.05" --folder "m5_d1.0_n10" --output "baseline_res.csv" 

# /home/admin/miniconda3/envs/easyFL/bin/python baseline.py --baseline "PC" --dataname "erdos_renyi/d100_p0.02" --folder "m5_d1.0_n10" --output "baseline_res.csv" 
