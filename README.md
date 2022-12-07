# Privacy-Preserving-Reinforcement-Learning-Beyond-Expectation
Privacy-Preserving Reinforcement Learning Beyond Expectation:

In this paper, we incorporated cumulative prospect theory (CPT) into the objective of a reinforcement learning (RL) problem to quantify risk and used differential privacy to keep decision making hidden from external parties.

```ruby
require 'matplotlib, gym, torch and PIL, cv2'
conda install matplotlib
conda install -c conda-forge pytorch-gpu
conda install -c conda-forge gym
conda install -c menpo opencv
```

To run the exprimetns:

```ruby

python CPT-TwoDimensionQLearning.py --DP  1 --CPT 1 --sigma 1

--DF 1 to turn on differential privacy otherwise 0
--CPT 1 to turn on risk-neutral otherwise 0
--sigma (float): the hyperparameter for differential privacy
```

To cite this paper:

```ruby

@article{rajabi2022privacy,
  title={Privacy-Preserving Reinforcement Learning Beyond Expectation},
  author={Rajabi, Arezoo and Ramasubramanian, Bhaskar and Maruf, Abdullah Al and Poovendran, Radha},
  journal={the 61st IEEE Conference on Decision and Control (CDC)},
  year={2022}
}
```


![plot](./figs/Env2.png)
![plot](./figs/LossConvergence.png)
