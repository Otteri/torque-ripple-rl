# torque-ripple-rl
This repository contains gym interface that can be utilized for prototyping. The interface is built for underlying C++ simulator, which is not distributed.

## pulsar.py
Torque pulsation can be learned and then compensated. The agent in `pulsar.py` learns simulated pulsations using Q-learning algorithm and then reduces the torque ripple. Figures below visualize the progress after training 0, 500, 2000 and 4900 episodes.
<div>
    <img src="images/0-episodes.gif" width="49%" />
    <img src="images/500-episodes.gif" width="49%" /> 
    <img src="images/2000-episodes.gif" width="49%" />
    <img src="images/4900-episodes.gif" width="49%" />
</div>

### Reward history  
![reward-history](images/reward-history.svg)

## Installation
`activate virtuan environment`  
`pip install -e ilmarinen_gym/`
