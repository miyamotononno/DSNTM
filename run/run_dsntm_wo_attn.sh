#!/bin/bash

#$-l h_rt=12:00:00
#$-o ./log/
#$-j y
#$-cwd
#$-m a
#$-m b
#$-m e

source ~/.bashrc
source ~/.bash_profile
conda activate py39

# train
env CUDA_VISIBLE_DEVICES=1 python main.py --seed 1111 --dataset acl --data_path dataset/acl --mode train --wo_attention --model_path models/acl/model_1111.pth>> logs/acl_wo_attn/log_1111.txt
env CUDA_VISIBLE_DEVICES=1 python main.py --seed 2222 --dataset acl --data_path dataset/acl --mode train --wo_attention --model_path models/acl/model_2222.pth >> logs/acl_wo_attn/log_2222.txt
env CUDA_VISIBLE_DEVICES=1 python main.py --seed 3333 --dataset acl --data_path dataset/acl --mode train --wo_attention --model_path models/acl/model_3333.pth >> logs/acl_wo_attn/log_3333.txt
env CUDA_VISIBLE_DEVICES=1 python main.py --seed 4444 --dataset acl --data_path dataset/acl --mode train --wo_attention --model_path models/acl/model_4444.pth >> logs/acl_wo_attn/log_4444.txt
env CUDA_VISIBLE_DEVICES=1 python main.py --seed 5555 --dataset acl --data_path dataset/acl --mode train --wo_attention --model_path models/acl/model_5555.pth >> logs/acl_wo_attn/log_5555.txt
env CUDA_VISIBLE_DEVICES=1 python main.py --seed 6666 --dataset acl --data_path dataset/acl --mode train --wo_attention --model_path models/acl/model_6666.pth >> logs/acl_wo_attn/log_6666.txt
env CUDA_VISIBLE_DEVICES=1 python main.py --seed 7777 --dataset acl --data_path dataset/acl --mode train --wo_attention --model_path models/acl/model_7777.pth >> logs/acl_wo_attn/log_7777.txt
env CUDA_VISIBLE_DEVICES=1 python main.py --seed 8888 --dataset acl --data_path dataset/acl --mode train --wo_attention --model_path models/acl/model_8888.pth >> logs/acl_wo_attn/log_8888.txt

env CUDA_VISIBLE_DEVICES=0 python main.py --seed 1111 --dataset cs --data_path dataset/cs --mode train --wo_attention --model_path models/cs/model_1111.pth>> logs/cs_wo_attn/log_1111.txt
env CUDA_VISIBLE_DEVICES=0 python main.py --seed 2222 --dataset cs --data_path dataset/cs --mode train --wo_attention --model_path models/cs/model_2222.pth >> logs/cs_wo_attn/log_2222.txt
env CUDA_VISIBLE_DEVICES=0 python main.py --seed 3333 --dataset cs --data_path dataset/cs --mode train --wo_attention --model_path models/cs/model_3333.pth >> logs/cs_wo_attn/log_3333.txt
env CUDA_VISIBLE_DEVICES=0 python main.py --seed 4444 --dataset cs --data_path dataset/cs --mode train --wo_attention --model_path models/cs/model_4444.pth >> logs/cs_wo_attn/log_4444.txt
env CUDA_VISIBLE_DEVICES=0 python main.py --seed 5555 --dataset cs --data_path dataset/cs --mode train --wo_attention --model_path models/cs/model_5555.pth >> logs/cs_wo_attn/log_5555.txt
env CUDA_VISIBLE_DEVICES=0 python main.py --seed 6666 --dataset cs --data_path dataset/cs --mode train --wo_attention --model_path models/cs/model_6666.pth >> logs/cs_wo_attn/log_6666.txt
env CUDA_VISIBLE_DEVICES=0 python main.py --seed 7777 --dataset cs --data_path dataset/cs --mode train --wo_attention --model_path models/cs/model_7777.pth >> logs/cs_wo_attn/log_7777.txt
env CUDA_VISIBLE_DEVICES=0 python main.py --seed 8888 --dataset cs --data_path dataset/cs --mode train --wo_attention --model_path models/cs/model_8888.pth >> logs/cs_wo_attn/log_8888.txt


# test
env CUDA_VISIBLE_DEVICES=1 python main.py --seed 1111 --dataset acl --data_path dataset/acl --mode test --wo_attention --model_path models/acl/model_1111.pth >> logs/acl_wo_attn/log_test.txt
env CUDA_VISIBLE_DEVICES=1 python main.py --seed 2222 --dataset acl --data_path dataset/acl --mode test --wo_attention --model_path models/acl/model_2222.pth >> logs/acl_wo_attn/log_test.txt
env CUDA_VISIBLE_DEVICES=1 python main.py --seed 3333 --dataset acl --data_path dataset/acl --mode test --wo_attention --model_path models/acl/model_3333.pth >> logs/acl_wo_attn/log_test.txt
env CUDA_VISIBLE_DEVICES=1 python main.py --seed 4444 --dataset acl --data_path dataset/acl --mode test --wo_attention --model_path models/acl/model_4444.pth >> logs/acl_wo_attn/log_test.txt
env CUDA_VISIBLE_DEVICES=1 python main.py --seed 5555 --dataset acl --data_path dataset/acl --mode test --wo_attention --model_path models/acl/model_5555.pth >> logs/acl_wo_attn/log_test.txt
env CUDA_VISIBLE_DEVICES=1 python main.py --seed 6666 --dataset acl --data_path dataset/acl --mode test --wo_attention --model_path models/acl/model_6666.pth >> logs/acl_wo_attn/log_test.txt
env CUDA_VISIBLE_DEVICES=1 python main.py --seed 7777 --dataset acl --data_path dataset/acl --mode test --wo_attention --model_path models/acl/model_7777.pth >> logs/acl_wo_attn/log_test.txt
env CUDA_VISIBLE_DEVICES=1 python main.py --seed 8888 --dataset acl --data_path dataset/acl --mode test --wo_attention --model_path models/acl/model_8888.pth >> logs/acl_wo_attn/log_test.txt

env CUDA_VISIBLE_DEVICES=0 python main.py --seed 1111 --dataset cs --data_path dataset/cs --mode test --wo_attention --model_path models/cs/model_1111.pth >> logs/cs_wo_attn/log_test.txt
env CUDA_VISIBLE_DEVICES=0 python main.py --seed 2222 --dataset cs --data_path dataset/cs --mode test --wo_attention --model_path models/cs/model_2222.pth >> logs/cs_wo_attn/log_test.txt
env CUDA_VISIBLE_DEVICES=0 python main.py --seed 3333 --dataset cs --data_path dataset/cs --mode test --wo_attention --model_path models/cs/model_3333.pth >> logs/cs_wo_attn/log_test.txt
env CUDA_VISIBLE_DEVICES=0 python main.py --seed 4444 --dataset cs --data_path dataset/cs --mode test --wo_attention --model_path models/cs/model_4444.pth >> logs/cs_wo_attn/log_test.txt
env CUDA_VISIBLE_DEVICES=0 python main.py --seed 5555 --dataset cs --data_path dataset/cs --mode test --wo_attention --model_path models/cs/model_5555.pth >> logs/cs_wo_attn/log_test.txt
env CUDA_VISIBLE_DEVICES=0 python main.py --seed 6666 --dataset cs --data_path dataset/cs --mode test --wo_attention --model_path models/cs/model_6666.pth >> logs/cs_wo_attn/log_test.txt
env CUDA_VISIBLE_DEVICES=0 python main.py --seed 7777 --dataset cs --data_path dataset/cs --mode test --wo_attention --model_path models/cs/model_7777.pth >> logs/cs_wo_attn/log_test.txt
env CUDA_VISIBLE_DEVICES=0 python main.py --seed 8888 --dataset cs --data_path dataset/cs --mode test --wo_attention --model_path models/cs/model_8888.pth >> logs/cs_wo_attn/log_test.txt