# DyCCF

## Introduction
This repository includes the implementation for Dynamic Causal Collaborative Filtering

> Paper: Dynamic Causal Collaborative Filtering <br>
> Paper Link: [https://dl.acm.org/doi/abs/10.1145/3511808.3557300](https://dl.acm.org/doi/abs/10.1145/3511808.3557300)

## Environment

Environment requirements can be found in `./requirement.txt`

## Datasets
  
- **Electronics**: The origin dataset can be found [here](https://nijianmo.github.io/amazon/index.html.). 

- **MovieLens-1M**: The origin dataset can be found [here](https://grouplens.org/datasets/movielens/).

- The data processing code can be found in `./src/data_processing/'

## Example to run the codes

For example:

```
# DyCCF on Electronics dataset based on GRU4Rec model
> cd ./src/
> python main-3phases.py --model GRU4Rec --dataset Electronics-3 --epoch 100 --phase1 0 --batch_size 256 --dccf 1 --eval_batch_size 5000 --gpu 3 --ctf_num 3 --load 1 --train 0 --metrics nDCG@10,hit@10,unbiasedndcg@10,unbiasedhit@10
```

## Citation

```
@inproceedings{xu2022dynamic,
  title={Dynamic causal collaborative filtering},
  author={Xu, Shuyuan and Tan, Juntao and Fu, Zuohui and Ji, Jianchao and Heinecke, Shelby and Zhang, Yongfeng},
  booktitle={Proceedings of the 31st ACM International Conference on Information \& Knowledge Management},
  pages={2301--2310},
  year={2022}
}
```
