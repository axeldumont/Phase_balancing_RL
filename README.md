# Phase balancing on low voltage distribution networks using Reinforcement Learning

This repo prensents an offline method to model the balancing of a low voltage distribution network using implemented algorithms of reinforcement learning.

## Usage

Example of usage of the ```main.py``` script :
```
python3 main.py --steps 10000 --model a2c --loads 5 --timesteps 10
```
* ```steps```: Number of steps to perform the training of the model
* ```model```: What model to use (implemented by the ```stable-baselines3``` library) between : ```a2c```, ```dqn```, ```ppo```
* ```loads```: The number of loads on the distribution network
* ```timesteps```: Number of timesteps of the data of the load profiles, used to calculate the imbalance of the network on a fixed period of time. 
