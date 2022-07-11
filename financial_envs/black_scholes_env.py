import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ndtr
import tensorflow as tf
from tf_agents.specs import array_spec
from tf_agents.specs import tensor_spec

from tf_agents.environments import py_environment
from tf_agents.environments import utils
from tf_agents.trajectories import time_step as ts

class BSEnv(py_environment.PyEnvironment):

    def __init__(self,S0 = 1,mu = 0.05,r = 0.05,sigma_S = 1.,sigma_V = 2.,strike = 1.,maturity = 1.,window = 10):
        self.window = window # Number of quotes available
        self.len_episodes = int(maturity*365) # A price per day
        self.discount = 1.

        self.S0 = S0 # Initial spot price
        self.T = maturity
        self.K = strike

        self.mu = mu # Real drift
        self.r = r # Risk free drift
        self.sigma_S = sigma_S # Volatility of the asset given by the market
        self.sigma_V = sigma_V # Volatility of the derivative given by the market


            # The observables are an array of asset prices, an array of V prices, the time to maturity and the value
            # of the portfolio
        self._action_spec = array_spec.BoundedArraySpec(
                shape=(2,), dtype=np.float32, minimum=-1., maximum=1., name='action')
        minimum = np.empty((4,window))
        minimum[0,:] = 0*np.ones(window) # tau
        minimum[1,:] = 0*np.ones(window) # s
        minimum[2,:] = 0*np.ones(window) # v
        minimum[3,:] = -np.inf*np.ones(window) # Portfolio
        maximum = np.empty((4,window))
        maximum[0,:] = maturity*np.ones(window)
        maximum[1,:] = 4*S0/strike*np.ones(window)
        maximum[2,:] = np.inf*np.ones(window)
        maximum[3,:] = np.inf*np.ones(window)
        self._observation_spec = array_spec.BoundedArraySpec(
                shape=(4,window), dtype=np.float32, minimum=minimum,maximum=maximum, name='observation')
        
        self._reward_spec = array_spec.BoundedArraySpec(
                shape=(1,), dtype=np.float32, minimum=-np.inf,maximum=np.inf, name='reward')
        
        self._time_step_spec = ts.time_step_spec(self.observation_spec(),self.reward_spec())

        self._state = np.empty((4,window),dtype = np.float32)
        self._episode_ended = False
        self._reset()

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec
    
    def reward_spec(self):
        return self._reward_spec
    
    def time_step_spec(self):
        return self._time_step_spec

    def EM_step(self,S):
        dt = self.T/self.len_episodes
        dW = np.random.randn()*np.sqrt(dt)
        return S+self.mu*S*dt+self.sigma_S*S*dW

    def BS_formula(self,t,S):
        d1 = 1/(self.sigma_V*np.sqrt(self.T-t))*(np.log(S/self.K)+(self.r+self.sigma_V*self.sigma_V/2)*(self.T-t))
        d2 = d1-self.sigma_V*np.sqrt(self.T-t)
        V = ndtr(d1)*S-ndtr(d2)*self.K*np.exp(-self.r*(self.T-t))
        return V

    def update_state(self):
        self._state[0,:] = self._tau[self.counter-self.window+1:self.counter+1]
        self._state[1,:] = self._s[self.counter-self.window+1:self.counter+1]
        self._state[2,:] = self._v[self.counter-self.window+1:self.counter+1]
        self._state[3,:] = self._portfolio[self.counter-self.window+1:self.counter+1]

        return self._state

    def _reset(self, seed = 123):
        if seed is not None:
            np.random.seed(seed)

        # Keep track of current position in time
        self.counter = 0

        # Information of the episode 
        self._t = np.linspace(0,self.T,self.len_episodes) 
        self._portfolio = np.zeros(self.len_episodes) 
        self._tau = self.T-self._t
        self._S =  np.zeros(self.len_episodes)
        self._s =  np.zeros(self.len_episodes)
        self._V = np.zeros(self.len_episodes) 
        self._v = np.zeros(self.len_episodes) 

        # Initialize
        self._S[0] = self.S0
        self._s[0] = self.S0/self.K
        self._V[0] = self.BS_formula(self._t[0],self._S[0])
        self._v[0] = self._V[0]/self.K

        # Generate window
        for i in range(0,self.window-1):
            self._S[i+1] = self.EM_step(self._S[i])
            self._s[i+1] = self._S[i+1]/self.K 
            self._V[i+1] = self.BS_formula(self._t[i],self._S[i]) 
            self._v[i+1] = self._V[i+1]/self.K
            self.counter = self.counter+1

        self._episode_ended = False
        reward = 0.0
        self.update_state()
        return ts.restart(self._state)



    def _step(self, action):

        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self.reset()
        action_S = action[0]
        action_V = action[1]

        # Update observations
        i = self.counter
        self._S[i+1] = self.EM_step(self._S[i])
        self._s[i+1] = self._S[i+1]/self.K 
        self._V[i+1] = self.BS_formula(self._t[i+1],self._S[i+1]) 
        self._v[i+1] = self._V[i+1]/self.K

        # Compute change in the portfolio and reward 
        delta_V = self._V[i+1]-self._V[i]
        delta_S = self._S[i+1]-self._S[i]
        delta_portfolio = action_V*delta_V-action_S*delta_S\
                +self.r*(action_S*self._S[i]-action_V*self._V[i])*(self._t[i+1]-self._t[i])
        self._portfolio[i+1] = self._portfolio[i]+delta_portfolio
        self.counter = self.counter+1

        # (action,R,O)
        observation = self.update_state()
        reward = delta_portfolio 
        self._episode_ended = bool(self._t[self.counter] == self.T)
        
        
        if self._episode_ended:
            return ts.termination(self._state, reward)
        else:
            return ts.transition(self._state,reward = reward,discount = self.discount)

    
    def play_episode(self,agent):
        time_step = self._reset()
        while not time_step.is_last():
            action = agent.action(time_step)
            time_step = self._step(action.action)
        return None

    def render(self):
        plt.plot(self._t[0:self.counter+1],self._S[0:self.counter+1],label = "S")
        plt.plot(self._t[0:self.counter+1],self._V[0:self.counter+1],label = "V")
        plt.plot(self._t[0:self.counter+1],self._portfolio[0:self.counter+1],label = "Portfolio")
        plt.xlabel(r"$t$")
        plt.ylabel("Price")
        plt.grid()
        plt.legend()
        plt.show()

