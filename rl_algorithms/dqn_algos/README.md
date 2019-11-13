# Deep Q Networks (DQN)
## Instructions
1. Train the agents, can use flag `--use-dueling` and `--use-double-net` to try the Double DQN or Dueling Network Architecture:
```bash
python train.py --env-name='<env name>' --cuda (if you have a GPU) --<other flags>
```
2. Play the demo - Please use the same algorithm flag as training:
```bash
python demo.py --env-name='<env name>' --<algo flags>
```
## Results
![](../../figures/01_dqn.png)
