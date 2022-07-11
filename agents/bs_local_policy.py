import numpy as np
import tensorflow as tf
from scipy.special import ndtr
from py_lets_be_rational import implied_volatility_from_a_transformed_rational_guess
from tf_agents.distributions import masked
from tf_agents.policies import py_policy
from tf_agents.specs import array_spec
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types
from tf_agents.utils import nest_utils


class BSLocalPolicy(py_policy.PyPolicy):
    """Returns local samples of the given action_spec."""

    def __init__(self,
            time_step_spec: ts.TimeStep,
            action_spec: types.NestedArraySpec,
            policy_state_spec: types.NestedArraySpec = (),
            maturity = 1.,
            strike = 1.,
            r = 0.05):
        """ Initializes the RandomPyPolicy.
        Args:
            time_step_spec: Reference `time_step_spec`. If not None and outer_dims
                is not provided this is used to infer the outer_dims required for the
                given time_step when action is called.
            action_spec: A nest of BoundedArraySpec representing the actions to sample
                from.
            policy_state_spec: Nest of `tf.TypeSpec` representing the data in the
                policy state field.
        """

        super(BSLocalPolicy, self).__init__(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            policy_state_spec=policy_state_spec)

        self.T = maturity
        self.K = strike
        self.r = r
        self.sigma = None

    def _reset(maturity = None,strike = None):
        if maturity is not None:
            self.T = maturity
        if K is not None:
            self.K = strike
        
        self.sigma = None

    
    def delta(self,tau,S):
        d1 = 1/(self.sigma*np.sqrt(tau))*(np.log(S/self.K)+(self.r+self.sigma*self.sigma/2)*tau)
        delta = ndtr(d1)
        return delta

    def calibrate(self,tau,S,V):
        self.sigma = implied_volatility_from_a_transformed_rational_guess(V,S*np.exp(self.r*tau),self.K,tau, q=1)

    def _action(self, time_step,policy_state=()):
        tau = time_step.observation[0,-1] 
        S = self.K*time_step.observation[1,-1] 
        V = self.K*time_step.observation[2,-1] 

        if self.sigma is None:
            tau_0 = time_step.observation[0,0] 
            S_0 = self.K*time_step.observation[1,0] 
            V_0 = self.K*time_step.observation[2,0] 
            self.calibrate(tau_0,S_0,V_0)

        new_action = (self.delta(tau,S),1)
        policy_state = ()
        info = ()
        

        return policy_step.PolicyStep(new_action, policy_state, info)
