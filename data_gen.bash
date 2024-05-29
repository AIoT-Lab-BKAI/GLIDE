# CUDA_VISIBLE_DEVICES=0 /home/admin/miniconda3/envs/easyFL/bin/python data_gen.py --dataname "water" --n 10 --s 10000
# CUDA_VISIBLE_DEVICES=0 /home/admin/miniconda3/envs/easyFL/bin/python data_gen.py --dataname "alarm" --n 10 --s 10000
# CUDA_VISIBLE_DEVICES=0 /home/admin/miniconda3/envs/easyFL/bin/python data_gen.py --dataname "barley" --n 10 --s 10000


CUDA_VISIBLE_DEVICES=0 /home/admin/miniconda3/envs/easyFL/bin/python data_synthesized.py --num_node 20 --p '0.11' --n 10 --s 2500 --mi 5 --di '1.0'
CUDA_VISIBLE_DEVICES=0 /home/admin/miniconda3/envs/easyFL/bin/python data_synthesized.py --num_node 20 --p '0.16' --n 10 --s 2500 --mi 5 --di '1.0'
CUDA_VISIBLE_DEVICES=0 /home/admin/miniconda3/envs/easyFL/bin/python data_synthesized.py --num_node 20 --p '0.22' --n 10 --s 2500 --mi 5 --di '1.0'

CUDA_VISIBLE_DEVICES=0 /home/admin/miniconda3/envs/easyFL/bin/python data_synthesized.py --num_node 40 --p '0.05' --n 10 --s 2500  --mi 5 --di '1.0'
CUDA_VISIBLE_DEVICES=0 /home/admin/miniconda3/envs/easyFL/bin/python data_synthesized.py --num_node 40 --p '0.07' --n 10 --s 2500  --mi 5 --di '1.0'
CUDA_VISIBLE_DEVICES=0 /home/admin/miniconda3/envs/easyFL/bin/python data_synthesized.py --num_node 40 --p '0.1' --n 10 --s 2500  --mi 5 --di '1.0'

CUDA_VISIBLE_DEVICES=0 /home/admin/miniconda3/envs/easyFL/bin/python data_synthesized.py --num_node 60 --p '0.03' --n 10 --s 2500  --mi 5 --di '1.0'
CUDA_VISIBLE_DEVICES=0 /home/admin/miniconda3/envs/easyFL/bin/python data_synthesized.py --num_node 60 --p '0.05' --n 10 --s 2500  --mi 5 --di '1.0'
CUDA_VISIBLE_DEVICES=0 /home/admin/miniconda3/envs/easyFL/bin/python data_synthesized.py --num_node 60 --p '0.07' --n 10 --s 2500  --mi 5 --di '1.0'

CUDA_VISIBLE_DEVICES=0 /home/admin/miniconda3/envs/easyFL/bin/python data_synthesized.py --num_node 80 --p '0.03' --n 10 --s 2500  --mi 5 --di '1.0'
CUDA_VISIBLE_DEVICES=0 /home/admin/miniconda3/envs/easyFL/bin/python data_synthesized.py --num_node 80 --p '0.04' --n 10 --s 2500  --mi 5 --di '1.0'
CUDA_VISIBLE_DEVICES=0 /home/admin/miniconda3/envs/easyFL/bin/python data_synthesized.py --num_node 80 --p '0.05' --n 10 --s 2500  --mi 5 --di '1.0'

CUDA_VISIBLE_DEVICES=0 /home/admin/miniconda3/envs/easyFL/bin/python data_synthesized.py --num_node 100 --p '0.02' --n 10 --s 2500  --mi 5 --di '1.0'
CUDA_VISIBLE_DEVICES=0 /home/admin/miniconda3/envs/easyFL/bin/python data_synthesized.py --num_node 100 --p '0.03' --n 10 --s 2500  --mi 5 --di '1.0'
CUDA_VISIBLE_DEVICES=0 /home/admin/miniconda3/envs/easyFL/bin/python data_synthesized.py --num_node 100 --p '0.04' --n 10 --s 2500  --mi 5 --di '1.0'



