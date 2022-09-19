import numpy as np
import tensorflow as tf
import sys
sys.path.append('./')


from tf_agents.policies import random_py_policy
from agents.bs_local_policy import BSLocalPolicy
from financial_envs.black_scholes_env import BSEnv

# Market
r = 0.05
mu = 0.1
sigma_S = 0.5
sigma_V = 0.4
S0 = 1
strike = 1
maturity = 1

# Create environment
environment = BSEnv(S0 = S0,mu = mu,r = r,sigma_S = sigma_S,sigma_V = sigma_V, 
        strike = strike, maturity = maturity,window = 1)

# Create agent
my_random_py_policy = random_py_policy.RandomPyPolicy(time_step_spec=None,action_spec=environment.action_spec())
bs_local_policy = BSLocalPolicy(time_step_spec=environment.time_step_spec(),action_spec=environment.action_spec(),
        maturity = maturity,strike = strike,r = r)

# Run episode
#environment.play_episode(my_random_py_policy)
#N_episodes = 10000
#for i in range(N_episodes):
environment.play_episode(bs_local_policy)
#environment.play_episode(my_random_py_policy)

# Render results
environment.render()


