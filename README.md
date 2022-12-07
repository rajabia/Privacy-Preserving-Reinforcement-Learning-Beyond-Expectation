# Privacy-Preserving-Reinforcement-Learning-Beyond-Expectation
Privacy-Preserving Reinforcement Learning Beyond Expectation

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

```
--DF 1 to turn on differential privacy otherwise 0
--CPT 1 to turn on risk-neutral otherwise 0
--sigma (float): the hyperparameter for differential privacy
