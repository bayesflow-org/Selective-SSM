import numpy as np
import scipy
import scipy.special
import keras

class PRL2:
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
    
    @staticmethod
    def simulate(theta, n_trials: int, pval: float = 0.8, min_switch: int = 10, n_actions: int = 3):
        # Init Q Vals and simulator params
        lr_alpha = theta[0] # alpha
        softmax_beta = theta[1] # beta
        softmax_beta = softmax_beta * 10
        q_vals = np.array([1 / n_actions] * n_actions)
        
        curr_correct = np.random.choice([0, 1]) # action that is more likely to be rewarded first
        curr_length = min_switch + np.random.randint(0, 5) # number of correct trials before switch
        cum_reward = 0
        
        actions = []
        rewards = []
        
        # Simulate over n_trials
        for _ in range(n_trials):
            # Select random action over learned qvals. Check for correctness
            action_prob = scipy.special.softmax(softmax_beta * q_vals)
            action = np.random.choice(n_actions, p=action_prob)
            correct = int(action == curr_correct)
            if np.random.uniform(0, 1, 1)[0] < pval:
                reward = correct
            else:
                reward = 1 - correct
            cum_reward += reward
            
            # Get reward prediction error and update Q Vals
            rpe = reward - q_vals[action]
            q_vals[action] += lr_alpha * rpe
            
            # Update remaining Q Vals via counterfactual learning
            rpe_counter = (1 - reward) - q_vals[1 - action]
            q_vals[1 - action] += lr_alpha * rpe_counter
            
            # Switch correct action if necessary
            if reward == 1 and cum_reward >= curr_length:
                curr_correct = 1 - curr_correct
                curr_length = min_switch + np.random.randint(0, 5)
                cum_reward = 0
            
            # Track actions and rewards
            actions.append(action)
            rewards.append(reward)
        
        # Broadcast to feature tensor
        actions = np.asarray(actions).reshape((1, n_trials))
        actions = keras.ops.one_hot(actions, n_actions)
        rewards = np.asarray(rewards).reshape((1, n_trials))
        features = keras.ops.concatenate([rewards[:, :, np.newaxis], actions], axis=2)
    
        return features