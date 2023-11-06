# Supported Value Regularization for Offline Reinforcement Learning

Original PyTorch implementation of the SVR algorithm from [Supported Value Regularization for Offline Reinforcement Learning](https://openreview.net/forum?id=fze7P9oy6l).

## Environment
Paper results were collected with [MuJoCo 210](https://mujoco.org/) (and [mujoco-py 2.1.2.14](https://github.com/openai/mujoco-py)) in [OpenAI gym 0.23.1](https://github.com/openai/gym) with the [D4RL datasets](https://github.com/Farama-Foundation/D4RL). Networks are trained using [PyTorch 1.11.0](https://github.com/pytorch/pytorch) and [Python 3.7](https://www.python.org/).

## Usage

### Pretrained Models

We have uploaded pretrained behavior models in SVR_bcmodels/ to facilitate experiment reproduction. 

You can also pretrain behavior models by running:
```
./run_pretrain.sh
```

### Offline RL


You can train SVR on D4RL datasets by running:
```
./run_experiments.sh
```

### Logging

This codebase uses tensorboard. You can view saved runs with:

```
tensorboard --logdir <run_dir>
```

## Bibtex
```
@inproceedings{mao2023supported,
	title={Supported Value Regularization for Offline Reinforcement Learning},
	author={Yixiu Mao and Hongchang Zhang and Chen Chen and Yi Xu and Xiangyang Ji},
	booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
	year={2023},
	url={https://openreview.net/forum?id=fze7P9oy6l}
}
```

## Acknowledgement

This repo borrows heavily from [TD3+BC](https://github.com/sfujim/TD3_BC) and [SPOT](https://github.com/thuml/SPOT).
