import numpy as np
import scipy
import scipy.special
from bayesflow.simulators.benchmark_simulators.benchmark_simulator import BenchmarkSimulator

class PRLSimulator(BenchmarkSimulator):
    def __init__(
        self,
        n_trials: int = 500,
        pval: float = 0.8,
        min_switch: int = 10,
        n_actions: int = 3,
        rng: np.random.Generator = None
    ):
        """
        Implements the Probabilistic Reversal Learning Task RL model, as described in [1], [2].
        This model accepts two free parameters in theta:
            1) learning_rate
            2) softmax_beta * 10
        On simulate, the model will simulate a number of trials while updating Q-Action values for a single agent
        
        The simulator returns the onehot actions and corresponding rewards over the number of trials
        This takes the shape (1, n_trials, n_actions + rewards_dim) -> (1, n_trials, n_actions + 1)
            
        [1] - https://github.com/MilenaCCNlab/MI-PEstNets/blob/main/Simulation/PRL/PRL_CognitiveModels_Utils.py (PRLtask_2ParamRL)  
        [2] - https://github.com/MilenaCCNlab/MI-PEstNets/blob/main/rnn/utils/data_utils.py (get_features, get_mixed_trials_features)  
        
        """
        self.n_trials = n_trials
        self.pval = pval
        self.min_switch = min_switch
        self.n_actions = n_actions
        self.rng = rng
        if self.rng is None:
            self.rng = np.random.default_rng()
    
    def _one_hot_encode(self, data, num_classes):
        one_hot = np.zeros((data.size, num_classes))
        one_hot[np.arange(data.size), data] = 1
        return one_hot
    
    def prior(self):
        params = self.rng.beta(a=[2, 2], b=[5, 5])
        return params
    
    def observation_model(self, params):
        alpha, beta = params # learning rate, qval scaling
        beta *= 10
        qvals = np.array([1 / self.n_actions] * self.n_actions)
        
        curr_correct = np.random.choice([0, 1]) # action that is more likely to be rewarded first
        curr_length = self.min_switch + np.random.randint(0, 5) # number of correct trials before switch
        cum_reward = 0
        
        actions = np.zeros((self.n_trials,), dtype="int32")
        rewards = np.zeros((self.n_trials,))
        
        # Simulate over n_trials
        for trial in range(self.n_trials):
            # Select random action over learned qvals. Check for correctness
            action_prob = scipy.special.softmax(beta * qvals)
            action = np.random.choice(self.n_actions, p=action_prob)
            correct = int(action == curr_correct)
            if np.random.uniform(0, 1, 1)[0] < self.pval:
                reward = correct
            else:
                reward = 1 - correct
            cum_reward += reward
            
            # Get reward prediction error and update Q Vals
            rpe = reward - qvals[action]
            qvals[action] += alpha * rpe
            
            # Update remaining Q Vals via counterfactual learning
            rpe_counter = (1 - reward) - qvals[1 - action]
            qvals[1 - action] += alpha * rpe_counter
            
            # Switch correct action if necessary
            if reward == 1 and cum_reward >= curr_length:
                curr_correct = 1 - curr_correct
                curr_length = self.min_switch + np.random.randint(0, 5)
                cum_reward = 0
            
            # Track actions and rewards
            actions[trial] = action
            rewards[trial] = reward
        
        # Broadcast to feature tensor
        actions = self._one_hot_encode(actions, self.n_actions)
        rewards = np.expand_dims(rewards, -1)
        x = np.concatenate([actions, rewards], axis=-1)
                
        return x