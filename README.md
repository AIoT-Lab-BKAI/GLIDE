# Causal Graph Learning via Distributional Invariance of Cause-Effect Relationship

This repository contains the Python implementation for the paper "Causal Graph Learning via Distributional Invariance of Cause-Effect Relationship".

The runnable file is proposal.py and can be executed by the following command to run the *sachs* dataset:<br>
```bash
python proposal.py --dataname "sachs" --folder "m3_d1_n10" --output "res/proposal-generic.csv" --gamma2 '0.5' --num_env 10 --TMB 1
```

Say, you wish to run the notears dataset, then:<br>
```bash
/home/admin/miniconda3/envs/easyFL/bin/python proposal.py --dataname "notears" --d 100 --b 4 --ntype "linear" --output "res/proposal-notears.csv" --gamma2 '0.5' --num_env 20 --TMB 1 --exp_repeat 5
```


Please inquiry us for the data folder, or you can modify the code so that it can read data at your will. The current code functions on the specific data folder structure as follows.<br>

```bash
--> data
    -- categorical
        -- sachs
        -- ...
    -- notears
        -- linearGaussian
        -- ...
```
The generating code for data in the folder **categorical** can be found in the ./utils folder. The code for data in the folder **notears** can be fetched from the following github: https://github.com/xunzheng/notears

[1] Zheng, X., Aragam, B., Ravikumar, P., & Xing, E. P. (2018). [DAGs with NO TEARS: Continuous optimization for structure learning](https://arxiv.org/abs/1803.01422) ([NeurIPS 2018](https://nips.cc/Conferences/2018/), Spotlight).

