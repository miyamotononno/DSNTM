# Dynamic Structured Neural Topic Model with Self-Attention Mechanism

This is code that accompanies the paper titled "Dynamic Structured Neural Topic Model with Self-Attention Mechanism" in Findings of ACL2023.

## Dependencies
+ python 3.9.13
+ pytorch 1.9.0

## Datasets

The s2orc datasets can be found below:  
https://github.com/allenai/s2orc

The pre-fitted embeddings can be found below:  
https://bitbucket.org/diengadji/embeddings/src

## Preprocessing
1. Download the s2orc data and put it into `metadata/` directory
2. Run `save_s2orc.py` and save targeted datasets of both ACL and CS into `dataset/`
3. Run `create_s2orc_dataset.py` and save datasets for model training or evaluating into `dataset/` 

## Training
To run the DSNTM, you can run the command below. You can specify different values for other arguments, peek at the arguments list in main.py.

```
python main.py --dataset <acl/cs> --data_path PATH_TO_DATA --model_path PATH_TO_MODEL --mode train
```
The DSTNM with Citation-Regularization is trained with the argument `--citation`.

The DSTNM without Self-Attention is trained with the argument `--wo_attention`.

## Validation/Test
To validate or to test the saved DSNTM, you can run the command below.
```
python main.py --dataset <acl/cs> --data_path PATH_TO_DATA --model_path PATH_TO_MODEL --mode <valid/test>
```

## Acknowledgement
The module to calculate NPMI (`cohrence.py`) is based on the code:  
https://github.com/jhlau/topic_interpretability





