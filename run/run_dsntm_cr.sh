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
env CUDA_VISIBLE_DEVICES=0 python main.py --seed 1111 --dataset acl --data_path dataset/acl --mode train --model_path models/acl_cr/model_1111.pth --citation >> logs/acl_cr/log_1111.txt
env CUDA_VISIBLE_DEVICES=0 python main.py --seed 2222 --dataset acl --data_path dataset/acl --mode train --model_path models/acl_cr/model_2222.pth --citation >> logs/acl_cr/log_2222.txt
env CUDA_VISIBLE_DEVICES=0 python main.py --seed 3333 --dataset acl --data_path dataset/acl --mode train --model_path models/acl_cr/model_3333.pth --citation >> logs/acl_cr/log_3333.txt
env CUDA_VISIBLE_DEVICES=0 python main.py --seed 4444 --dataset acl --data_path dataset/acl --mode train --model_path models/acl_cr/model_4444.pth --citation >> logs/acl_cr/log_4444.txt
env CUDA_VISIBLE_DEVICES=0 python main.py --seed 5555 --dataset acl --data_path dataset/acl --mode train --model_path models/acl_cr/model_5555.pth --citation >> logs/acl_cr/log_5555.txt
env CUDA_VISIBLE_DEVICES=0 python main.py --seed 6666 --dataset acl --data_path dataset/acl --mode train --model_path models/acl_cr/model_6666.pth --citation >> logs/acl_cr/log_6666.txt
env CUDA_VISIBLE_DEVICES=0 python main.py --seed 7777 --dataset acl --data_path dataset/acl --mode train --model_path models/acl_cr/model_7777.pth --citation >> logs/acl_cr/log_7777.txt
env CUDA_VISIBLE_DEVICES=0 python main.py --seed 8888 --dataset acl --data_path dataset/acl --mode train --model_path models/acl_cr/model_8888.pth --citation >> logs/acl_cr/log_8888.txt

env CUDA_VISIBLE_DEVICES=0 python main.py --seed 1111 --dataset cs --data_path dataset/cs --mode train --model_path models/cs_cr/model_1111.pth --citation >> logs/cs_cr/log_1111.txt
env CUDA_VISIBLE_DEVICES=0 python main.py --seed 2222 --dataset cs --data_path dataset/cs --mode train --model_path models/cs_cr/model_2222.pth --citation >> logs/cs_cr/log_2222.txt
env CUDA_VISIBLE_DEVICES=0 python main.py --seed 3333 --dataset cs --data_path dataset/cs --mode train --model_path models/cs_cr/model_3333.pth --citation >> logs/cs_cr/log_3333.txt
env CUDA_VISIBLE_DEVICES=0 python main.py --seed 4444 --dataset cs --data_path dataset/cs --mode train --model_path models/cs_cr/model_4444.pth --citation >> logs/cs_cr/log_4444.txt
env CUDA_VISIBLE_DEVICES=0 python main.py --seed 5555 --dataset cs --data_path dataset/cs --mode train --model_path models/cs_cr/model_5555.pth --citation >> logs/cs_cr/log_5555.txt
env CUDA_VISIBLE_DEVICES=0 python main.py --seed 6666 --dataset cs --data_path dataset/cs --mode train --model_path models/cs_cr/model_6666.pth --citation >> logs/cs_cr/log_6666.txt
env CUDA_VISIBLE_DEVICES=0 python main.py --seed 7777 --dataset cs --data_path dataset/cs --mode train --model_path models/cs_cr/model_7777.pth --citation >> logs/cs_cr/log_7777.txt
env CUDA_VISIBLE_DEVICES=0 python main.py --seed 8888 --dataset cs --data_path dataset/cs --mode train --model_path models/cs_cr/model_8888.pth --citation >> logs/cs_cr/log_8888.txt


# test
env CUDA_VISIBLE_DEVICES=0 python main.py --seed 1111 --dataset acl --data_path dataset/acl --mode test --model_path models/acl_cr/model_1111.pth --citation >> logs/acl_cr/log_test.txt
env CUDA_VISIBLE_DEVICES=0 python main.py --seed 2222 --dataset acl --data_path dataset/acl --mode test --model_path models/acl_cr/model_2222.pth --citation >> logs/acl_cr/log_test.txt
env CUDA_VISIBLE_DEVICES=0 python main.py --seed 3333 --dataset acl --data_path dataset/acl --mode test --model_path models/acl_cr/model_3333.pth --citation >> logs/acl_cr/log_test.txt
env CUDA_VISIBLE_DEVICES=0 python main.py --seed 4444 --dataset acl --data_path dataset/acl --mode test --model_path models/acl_cr/model_4444.pth --citation >> logs/acl_cr/log_test.txt
env CUDA_VISIBLE_DEVICES=0 python main.py --seed 5555 --dataset acl --data_path dataset/acl --mode test --model_path models/acl_cr/model_5555.pth --citation >> logs/acl_cr/log_test.txt
env CUDA_VISIBLE_DEVICES=0 python main.py --seed 6666 --dataset acl --data_path dataset/acl --mode test --model_path models/acl_cr/model_6666.pth --citation >> logs/acl_cr/log_test.txt
env CUDA_VISIBLE_DEVICES=0 python main.py --seed 7777 --dataset acl --data_path dataset/acl --mode test --model_path models/acl_cr/model_7777.pth --citation >> logs/acl_cr/log_test.txt
env CUDA_VISIBLE_DEVICES=0 python main.py --seed 8888 --dataset acl --data_path dataset/acl --mode test --model_path models/acl_cr/model_8888.pth --citation >> logs/acl_cr/log_test.txt

env CUDA_VISIBLE_DEVICES=0 python main.py --seed 1111 --dataset cs --data_path dataset/cs --mode test --model_path models/cs_cr/model_1111.pth --citation >> logs/cs_cr/log_test.txt
env CUDA_VISIBLE_DEVICES=0 python main.py --seed 2222 --dataset cs --data_path dataset/cs --mode test --model_path models/cs_cr/model_2222.pth --citation >> logs/cs_cr/log_test.txt
env CUDA_VISIBLE_DEVICES=0 python main.py --seed 3333 --dataset cs --data_path dataset/cs --mode test --model_path models/cs_cr/model_3333.pth --citation >> logs/cs_cr/log_test.txt
env CUDA_VISIBLE_DEVICES=0 python main.py --seed 4444 --dataset cs --data_path dataset/cs --mode test --model_path models/cs_cr/model_4444.pth --citation >> logs/cs_cr/log_test.txt
env CUDA_VISIBLE_DEVICES=0 python main.py --seed 5555 --dataset cs --data_path dataset/cs --mode test --model_path models/cs_cr/model_5555.pth --citation >> logs/cs_cr/log_test.txt
env CUDA_VISIBLE_DEVICES=0 python main.py --seed 6666 --dataset cs --data_path dataset/cs --mode test --model_path models/cs_cr/model_6666.pth --citation >> logs/cs_cr/log_test.txt
env CUDA_VISIBLE_DEVICES=0 python main.py --seed 7777 --dataset cs --data_path dataset/cs --mode test --model_path models/cs_cr/model_7777.pth --citation >> logs/cs_cr/log_test.txt
env CUDA_VISIBLE_DEVICES=0 python main.py --seed 8888 --dataset cs --data_path dataset/cs --mode test --model_path models/cs_cr/model_8888.pth --citation >> logs/cs_cr/log_test.tx