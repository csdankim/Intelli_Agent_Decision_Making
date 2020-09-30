# Distributed Deep Q-Learning 

The goal of this assignment is to implement and experiment with both single-core and distributed versions of the deep reinforcement learning algorithm Deep Q Networks (DQN). In particular, DQN will be run in the classic RL benchmark Cart-Pole and abblation experiments will be run to observe the impact of the different DQN components. 

The relevant content about DQN can be found Q-Learning and SARSA are in the following course notes from CS533.

https://oregonstate.instructure.com/courses/1719746/files/75047394/download?wrap=1

The full pseudo-code for DQN is on slide 45 with prior slides introducing the individual components. 


## Recap of DQN 

From the course slides it can be seen that DQN is simply the standard table-based Q-learning algorithm but with three extensions:

1) Use of function approximation via a neural network instead of a Q-table. 
2) Use of experience replay. 
3) Use of a target network. 

Extension (1) allows for scaling to problems with enormous state spaces, such as when the states correspond to images or sequences of images. Extensions (2) and (3) are claimed to improve the robustness and effectiveness of DQN compared. 

(2) adjusts Q-learning so that updates are not just performed on individual experiences as they arrive. But rather, experiences are stored in a memory buffer and updates are performed by sampling random mini-batches of experience tuples from the memory buffer and updating the network based on the mini-batch. This allows for reuse of experience as well as helping to reduce correlation between successive updates, which is claimed to be beneficial. 

(3) adjusts the way that target values are computed for the Q-learning updates. Let $Q_{\theta}(s,a)$ be the function approximation network with parameters $\theta$ for representing the Q-function. Given an experience tuple $(s, a, r, s')$ the origional Q-learning algorithm updates the parameters so that $Q_{\theta}(s,a)$ moves closer to the target value: 
\begin{equation}
r + \beta \max_a' Q_{\theta}(s',a') 
\end{equation}
Rather, DQN stores two function approximation networks. The first is the update network with parameters $\theta$, which is the network that is continually updated during learning. The second is a target network with parameters $\theta'$. Given the same experience tuple, DQN will update the parameters $\theta$ so that $Q_{\theta}(s,a)$ moves toward a target value based on the target network:
\begin{equation}
r + \beta \max_a' Q_{\theta'}(s',a') 
\end{equation}
Periodically the target network is updated with the most recent parameters $\theta' \leftarrow \theta$. This use of a target network is claimed to stabilize learning.

In the assignment you will get to see an example of the impact of both the target network and experience replay.

Further reading about DQN and its application to learning to play Atari games can be found in the following paper. 

Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A.A., Veness, J., Bellemare, M.G., Graves, A., Riedmiller, M., Fidjeland, A.K., Ostrovski, G. and Petersen, S., 2015. Human-level control through deep reinforcement learning. Nature, 518(7540), p.529.
https://oregonstate.instructure.com/courses/1719746/files/75234294/download?wrap=1


```python
# !pip3 install --user gym[Box2D]
# !pip3 install --user torch
# !pip3 install --user JSAnimation
# !pip3 install --user matplotlib
```

Install the packages for enviroment


```python
import gym
import torch
import time
import os
import ray
import numpy as np

from tqdm import tqdm
from random import uniform, randint

import io
import base64
from IPython.display import HTML

from dqn_model import DQNModel
from dqn_model import _DQNModel
from memory import ReplayBuffer

import matplotlib.pyplot as plt
%matplotlib inline

FloatTensor = torch.FloatTensor
```

## Useful PyTorch functions

### Tensors

This assignment will use the PyTorch library for the required neural network functionality. You do not need to be familiar with the details of PyTorch or neural network training. However, the assignment will require dealing with data in the form of tensors.  

The mini-batches used to train the PyTorch neural network is expected to be represented as a tensor matrix. The function `FloatTensor` can convert a list or NumPy matrix into a tensor matrix if needed. 

You can find more infomation here: https://pytorch.org/docs/stable/tensors.html


```python
# list
m = [[3,2,1],[6,4,5],[7,8,9]]
print(m)

# tensor matrix
m_tensor = FloatTensor(m)
print(type(m_tensor))
print(m_tensor)
```

    [[3, 2, 1], [6, 4, 5], [7, 8, 9]]
    <class 'torch.Tensor'>
    tensor([[3., 2., 1.],
            [6., 4., 5.],
            [7., 8., 9.]])


### Tensor.max()
Once you have a tenosr maxtrix, you can use torch.max(m_tensor, dim) to get the max value and max index corresponding to the dimension you choose.
```
>>> a = torch.randn(4, 4)
>>> a
tensor([[-1.2360, -0.2942, -0.1222,  0.8475],
        [ 1.1949, -1.1127, -2.2379, -0.6702],
        [ 1.5717, -0.9207,  0.1297, -1.8768],
        [-0.6172,  1.0036, -0.6060, -0.2432]])
>>> torch.max(a, 1)
torch.return_types.max(values=tensor([0.8475, 1.1949, 1.5717, 1.0036]), indices=tensor([3, 0, 0, 1]))
```
You can find more infomation here: https://pytorch.org/docs/stable/torch.html#torch.max


```python
max_value, index = torch.max(m_tensor, dim = 1)
print(max_value, index)
```

    tensor([3., 6., 9.]) tensor([0, 0, 2])


## Initialize Environment
### CartPole-v0:  
CartPole is a classic control task that is often used as an introductory reinforcement learning benchmark. The environment involves controlling a 2d cart that can move in either the left or right direction on a frictionless track. A pole is attached to the cart via an unactuated joint. The system is controlled by applying a force of +1 or -1 to the cart. The pendulum starts upright, and the goal is to prevent it from falling over. A reward of +1 is provided for every timestep that the pole remains upright. The episode ends when the pole is more than 15 degrees from vertical, or the cart moves more than 2.4 units from the center.  
(You can find more infomation by this Link: https://gym.openai.com/envs/CartPole-v0/)  
  



```python
# Set the Env name and action space for CartPole
ENV_NAME = 'CartPole-v0'
# Move left, Move right
ACTION_DICT = {
    "LEFT": 0,
    "RIGHT":1
}
# Register the environment
env_CartPole = gym.make(ENV_NAME)
```


```python
# Set result saveing floder
result_floder = ENV_NAME
result_file = ENV_NAME + "/results.txt"
if not os.path.isdir(result_floder):
    os.mkdir(result_floder)
```

## Helper Function
Plot results.


```python
def plot_result(total_rewards ,learning_num, legend):
    print("\nLearning Performance:\n")
    episodes = []
    for i in range(len(total_rewards)):
        episodes.append(i * learning_num + 1)
        
    plt.figure(num = 1)
    fig, ax = plt.subplots()
    plt.plot(episodes, total_rewards)
    plt.title('performance')
    plt.legend(legend)
    plt.xlabel("Episodes")
    plt.ylabel("total rewards")
    plt.show()
```

## Hyperparams
When function approximation is involves, especially neural networks, additional hyper parameters are inroduced and setting the parameters can require experience. Below is a list of the hyperparameters used in this assignment and values for the parameters that have worked well for a basic DQN implementation. You will adjust these values for particular parts of the assignment. For example, experiments that do not use the target network will set 'use_target_model' to False. 

You can find the more infomation about these hyperparameters in the notation of DQN_agent.init() function.


```python
hyperparams_CartPole = {
    'epsilon_decay_steps' : 100000, 
    'final_epsilon' : 0.1,
#     'batch_size' : 32,
    'batch_size' : 1,
#     'update_steps' : 10, 
    'update_steps' : 1, 
#     'memory_size' : 2000, 
    'memory_size' : 1, 
    'beta' : 0.99, 
    'model_replace_freq' : 2000,
    'learning_rate' : 0.0003,
#     'use_target_model': True
    'use_target_model': False
}
```

***
# Part 1: Non-distributed DQN

In this part, you will complete an implementation of DQN and run experiments on the CartPole environment from OpenAI Gym.  
Note that OpenAI Gym has many other environments that use the same interface---so this experience will allow the curious student to easily explore these algorithms more widely. 

Below you need to fill in the missing code for the DQN implementation. 

The Run function below can then be used to generate learning curves. 

You should conduct the following experiments involving different features of DQN. 

1. DQN without a replay buffer and without a target network. This is just standard Q-learning with a function approximator.
    The corresponding parameters are: memory_size = 1, update_steps = 1, batch_size = 1, use_target_model = False  
    
2. DQN without a replay buffer (but including the target network).   
    The corresponding parameters are: memory_size = 1, update_steps = 1, batch_size = 1, use_target_model = True  

3. DQN with a replay buffer, but without a target network.   
    Here you set use_target_model = False and otherwise set the replay memory parameters to the above suggested values 
   
4. Full DQN

For each experiment, record the parameters that you used, plot the resulting learning curves, and give a summary of your observations regarding the differences you observed. 
***



## DQN Agent

The full DQN agent involves a number of functions, the neural network, and the replay memory. Interfaces to a neural network model and memory are provided. 

Some useful information is below:   
- Neural Network Model: The network is used to represent the Q-function $Q(s,a)$. It takes a state $s$ as input and returns a vector of Q-values, one value for each action. The following interface functions are used for predicting Q-values, actions, and updating the neural network model parameters. 
    1. Model.predict(state) --- Returns the action that has the best Q-value in 'state'.
    2. Model.predict_batch(states) --- This is used to predict both the Q-values and best actions for a batch of states. Given a batch of states, the function returns: 1) 'best_actions' a vector containing the best action for each input state, and 2) 'q_values' a matrix where each row gives the Q-value for all actions of each state (one row per state).   
    3. Model.fit(q_values, q_target) --- It is used to update the neural network (via back-propagation). 'q_values' is a vector containing the Q-value predictions for a list of state-action pairs (e.g. from a batch of experience tuples). 'q_target' is a vector containing target values that we would like the correspoinding predictions to get closer to. This function updates the network in a way that the network predictions will ideally be closer to the targets. There is no return value.  
    4. Model.replace(another_model) --- It takes another model as input, and replace the weight of itself by the input model.
- Memory: This is the buffer used to store experience tuples for experience replay.
    1. Memory.add(state, action, reward, state', is_terminal) --- It takes one example as input, and store it into its storage.  
    2. Memory.sample(batch_size) --- It takes a batch_size int number as input. Return 'batch_size' number of randomly selected examples from the current memory buffer. The batch takes the form (states, actions, rewards, states', is_terminals) with each component being a vector/list of size equal to batch_size. 


```python
class DQN_agent(object):
    def __init__(self, env, hyper_params, action_space = len(ACTION_DICT)):
        
        self.env = env
        self.max_episode_steps = env._max_episode_steps
        
        """
            beta: The discounted factor of Q-value function
            (epsilon): The explore or exploit policy epsilon. 
            initial_epsilon: When the 'steps' is 0, the epsilon is initial_epsilon, 1
            final_epsilon: After the number of 'steps' reach 'epsilon_decay_steps', 
                The epsilon set to the 'final_epsilon' determinately.
            epsilon_decay_steps: The epsilon will decrease linearly along with the steps from 0 to 'epsilon_decay_steps'.
        """
        self.beta = hyper_params['beta']
        self.initial_epsilon = 1
        self.final_epsilon = hyper_params['final_epsilon']
        self.epsilon_decay_steps = hyper_params['epsilon_decay_steps']

        """
            episode: Record training episode
            steps: Add 1 when predicting an action
            learning: The trigger of agent learning. It is on while training agent. It is off while testing agent.
            action_space: The action space of the current environment, e.g 2.
        """
        self.episode = 0
        self.steps = 0
        self.best_reward = 0
        self.learning = True
        self.action_space = action_space

        """
            input_len The input length of the neural network. It equals to the length of the state vector.
            output_len: The output length of the neural network. It is equal to the action space.
            eval_model: The model for predicting action for the agent.
            target_model: The model for calculating Q-value of next_state to update 'eval_model'.
            use_target_model: Trigger for turn 'target_model' on/off
        """
        state = env.reset()
        input_len = len(state)
        output_len = action_space
        self.eval_model = DQNModel(input_len, output_len, learning_rate = hyper_params['learning_rate'])
        self.use_target_model = hyper_params['use_target_model']
        if self.use_target_model:
            self.target_model = DQNModel(input_len, output_len)
#         memory: Store and sample experience replay.
        self.memory = ReplayBuffer(hyper_params['memory_size'])
        
        """
            batch_size: Mini batch size for training model.
            update_steps: The frequence of traning model
            model_replace_freq: The frequence of replacing 'target_model' by 'eval_model'
        """
        self.batch_size = hyper_params['batch_size']
        self.update_steps = hyper_params['update_steps']
        self.model_replace_freq = hyper_params['model_replace_freq']
        
    # Linear decrease function for epsilon
    def linear_decrease(self, initial_value, final_value, curr_steps, final_decay_steps):
        decay_rate = curr_steps / final_decay_steps
        if decay_rate > 1:
            decay_rate = 1
        return initial_value - (initial_value - final_value) * decay_rate
    
    def explore_or_exploit_policy(self, state):
        p = uniform(0, 1)
        # Get decreased epsilon
        epsilon = self.linear_decrease(self.initial_epsilon, 
                               self.final_epsilon,
                               self.steps,
                               self.epsilon_decay_steps)
        
        if p < epsilon:
            #return action
            return randint(0, self.action_space - 1)
        else:
            #return action
            return self.greedy_policy(state)
        
    def greedy_policy(self, state):
        return self.eval_model.predict(state)
    
    # This next function will be called in the main RL loop to update the neural network model given a batch of experience
    # 1) Sample a 'batch_size' batch of experiences from the memory.
    # 2) Predict the Q-value from the 'eval_model' based on (states, actions)
    # 3) Predict the Q-value from the 'target_model' base on (next_states), and take the max of each Q-value vector, Q_max
    # 4) If is_terminal == 1, q_target = reward + discounted factor * Q_max, otherwise, q_target = reward
    # 5) Call fit() to do the back-propagation for 'eval_model'.
    def update_batch(self):
        if len(self.memory) < self.batch_size or self.steps % self.update_steps != 0:
            return

        batch = self.memory.sample(self.batch_size)

        (states, actions, reward, next_states,
         is_terminal) = batch
        
        states = states
        next_states = next_states
        terminal = FloatTensor([1 if t else 0 for t in is_terminal])
        reward = FloatTensor(reward)
        batch_index = torch.arange(self.batch_size,
                                   dtype=torch.long)
        
        # Current Q Values
        _, q_values = self.eval_model.predict_batch(states)
        q_values = q_values[batch_index, actions]
        
        # Calculate target
        if self.use_target_model:
            actions, q_next = self.target_model.predict_batch(next_states)
        else:
            actions, q_next = self.eval_model.predict_batch(next_states)
            
        #INSERT YOUR CODE HERE --- neet to compute 'q_targets' used below
#         q_max, index = torch.max(q_next, dim = 1)
        q_target = reward + self.beta * torch.max(q_next, dim = 1)[0] * (1 - terminal)

        # update model
        self.eval_model.fit(q_values, q_target)
    
    def learn_and_evaluate(self, training_episodes, test_interval):
        test_number = training_episodes // test_interval
        all_results = []
        
        for i in range(test_number):
            # learn
            self.learn(test_interval)
            
            # evaluate
            avg_reward = self.evaluate()
            all_results.append(avg_reward)
            
        return all_results
    
    def learn(self, test_interval):
        for episode in tqdm(range(test_interval), desc="Training"):
            state = self.env.reset()
            done = False
            steps = 0

            while steps < self.max_episode_steps and not done:
                #INSERT YOUR CODE HERE
                steps += 1
                self.steps += 1
                # add experience from explore-exploit policy to memory
                action = self.explore_or_exploit_policy(state)
                new_state, reward, done, _ = self.env.step(action)
                self.memory.add(state, action, reward, new_state, done)
                state = new_state
                
                # update the model every 'update_steps' of experience
                if self.steps % self.update_steps == 0:
                    self.update_batch()
                
                # update the target network (if the target network is being used) every 'model_replace_freq' of experiences               
                if self.use_target_model and self.steps % self.model_replace_freq == 0:
                    self.target_model.replace(self.eval_model)
                    
    
    def evaluate(self, trials = 30):
        total_reward = 0
        for _ in tqdm(range(trials), desc="Evaluating"):
            state = self.env.reset()
            done = False
            steps = 0

            while steps < self.max_episode_steps and not done:
                steps += 1
                action = self.greedy_policy(state)
                state, reward, done, _ = self.env.step(action)
                total_reward += reward

        avg_reward = total_reward / trials
        f = open(result_file, "a+")
        f.write(str(avg_reward) + "\n")
        f.close()
        if avg_reward >= self.best_reward:
            self.best_reward = avg_reward
            self.save_model()
        return avg_reward

    # save model
    def save_model(self):
        self.eval_model.save(result_floder + '/best_model.pt')
        
    # load model
    def load_model(self):
        self.eval_model.load(result_floder + '/best_model.pt')
```

## Run function


```python
training_episodes, test_interval = 10000, 50
agent = DQN_agent(env_CartPole, hyperparams_CartPole)
result = agent.learn_and_evaluate(training_episodes, test_interval)
plot_result(result, test_interval, ["batch_update with target_model"])
```

    Training: 100%|██████████| 50/50 [00:01<00:00, 30.22it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 620.56it/s]
    Training: 100%|██████████| 50/50 [00:02<00:00, 21.80it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 527.46it/s]
    Training: 100%|██████████| 50/50 [00:03<00:00, 10.32it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 450.29it/s]
    Training: 100%|██████████| 50/50 [00:02<00:00, 26.44it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 490.16it/s]
    Training: 100%|██████████| 50/50 [00:02<00:00, 17.90it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 258.49it/s]
    Training: 100%|██████████| 50/50 [00:02<00:00, 18.18it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 355.55it/s]
    Training: 100%|██████████| 50/50 [00:02<00:00, 18.89it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 89.28it/s]
    Training: 100%|██████████| 50/50 [00:03<00:00, 16.16it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 46.63it/s]
    Training: 100%|██████████| 50/50 [00:02<00:00, 18.60it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 532.81it/s]
    Training: 100%|██████████| 50/50 [00:02<00:00, 17.64it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 270.90it/s]
    Training: 100%|██████████| 50/50 [00:04<00:00, 12.16it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 511.70it/s]
    Training: 100%|██████████| 50/50 [00:02<00:00, 21.30it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 142.79it/s]
    Training: 100%|██████████| 50/50 [00:02<00:00, 20.96it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 90.05it/s]
    Training: 100%|██████████| 50/50 [00:02<00:00, 17.53it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 224.31it/s]
    Training: 100%|██████████| 50/50 [00:03<00:00, 16.37it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 363.70it/s]
    Training: 100%|██████████| 50/50 [00:02<00:00, 17.11it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 338.76it/s]
    Training: 100%|██████████| 50/50 [00:03<00:00, 16.44it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 308.21it/s]
    Training: 100%|██████████| 50/50 [00:02<00:00, 17.97it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 397.75it/s]
    Training: 100%|██████████| 50/50 [00:03<00:00, 16.35it/s]
    Evaluating: 100%|██████████| 30/30 [00:01<00:00, 27.45it/s]
    Training: 100%|██████████| 50/50 [00:03<00:00,  8.71it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 295.72it/s]
    Training: 100%|██████████| 50/50 [00:03<00:00, 14.76it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 294.99it/s]
    Training: 100%|██████████| 50/50 [00:03<00:00, 16.61it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 81.10it/s]
    Training: 100%|██████████| 50/50 [00:03<00:00, 12.85it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 30.54it/s]
    Training: 100%|██████████| 50/50 [00:04<00:00, 11.04it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 48.19it/s]
    Training: 100%|██████████| 50/50 [00:03<00:00,  9.79it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 175.16it/s]
    Training: 100%|██████████| 50/50 [00:04<00:00,  8.99it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 37.46it/s]
    Training: 100%|██████████| 50/50 [00:06<00:00,  7.38it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 28.64it/s]
    Training: 100%|██████████| 50/50 [00:04<00:00,  8.64it/s]
    Evaluating: 100%|██████████| 30/30 [00:01<00:00, 25.42it/s]
    Training: 100%|██████████| 50/50 [00:04<00:00, 11.16it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 99.85it/s]
    Training: 100%|██████████| 50/50 [00:04<00:00, 10.05it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 33.80it/s]
    Training: 100%|██████████| 50/50 [00:05<00:00,  8.83it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 131.33it/s]
    Training: 100%|██████████| 50/50 [00:05<00:00,  8.79it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 170.82it/s]
    Training: 100%|██████████| 50/50 [00:04<00:00,  7.77it/s]
    Evaluating: 100%|██████████| 30/30 [00:01<00:00, 29.77it/s]
    Training: 100%|██████████| 50/50 [00:05<00:00,  8.41it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 219.39it/s]
    Training: 100%|██████████| 50/50 [00:06<00:00,  6.33it/s]
    Evaluating: 100%|██████████| 30/30 [00:01<00:00, 28.17it/s]
    Training: 100%|██████████| 50/50 [00:05<00:00,  9.11it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 45.43it/s]
    Training: 100%|██████████| 50/50 [00:07<00:00,  6.81it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 39.46it/s]
    Training: 100%|██████████| 50/50 [00:06<00:00,  6.90it/s]
    Evaluating: 100%|██████████| 30/30 [00:01<00:00, 26.92it/s]
    Training: 100%|██████████| 50/50 [00:08<00:00,  4.20it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 350.76it/s]
    Training: 100%|██████████| 50/50 [00:08<00:00,  6.53it/s]
    Evaluating: 100%|██████████| 30/30 [00:01<00:00, 26.12it/s]
    Training: 100%|██████████| 50/50 [00:10<00:00,  5.88it/s]
    Evaluating: 100%|██████████| 30/30 [00:01<00:00, 28.96it/s]
    Training: 100%|██████████| 50/50 [00:08<00:00,  4.56it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 61.79it/s]
    Training: 100%|██████████| 50/50 [00:10<00:00,  4.59it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 319.65it/s]
    Training: 100%|██████████| 50/50 [00:11<00:00,  3.11it/s]
    Evaluating: 100%|██████████| 30/30 [00:01<00:00, 29.05it/s]
    Training: 100%|██████████| 50/50 [00:11<00:00,  4.38it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 207.96it/s]
    Training: 100%|██████████| 50/50 [00:13<00:00,  5.27it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 35.11it/s]
    Training: 100%|██████████| 50/50 [00:13<00:00,  4.14it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 33.11it/s]
    Training: 100%|██████████| 50/50 [00:10<00:00,  4.68it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 43.84it/s]
    Training: 100%|██████████| 50/50 [00:11<00:00,  4.35it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 331.24it/s]
    Training: 100%|██████████| 50/50 [00:12<00:00,  3.94it/s]
    Evaluating: 100%|██████████| 30/30 [00:01<00:00, 27.95it/s]
    Training: 100%|██████████| 50/50 [00:14<00:00,  4.58it/s]
    Evaluating: 100%|██████████| 30/30 [00:01<00:00, 28.02it/s]
    Training: 100%|██████████| 50/50 [00:12<00:00,  4.58it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 40.23it/s]
    Training: 100%|██████████| 50/50 [00:11<00:00,  3.37it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 153.75it/s]
    Training: 100%|██████████| 50/50 [00:10<00:00,  4.38it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 47.90it/s]
    Training: 100%|██████████| 50/50 [00:10<00:00,  4.38it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 30.58it/s]
    Training: 100%|██████████| 50/50 [00:09<00:00,  5.03it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 59.36it/s]
    Training: 100%|██████████| 50/50 [00:13<00:00,  3.93it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 80.86it/s]
    Training: 100%|██████████| 50/50 [00:13<00:00,  3.07it/s]
    Evaluating: 100%|██████████| 30/30 [00:01<00:00, 28.56it/s]
    Training: 100%|██████████| 50/50 [00:14<00:00,  2.84it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 125.04it/s]
    Training: 100%|██████████| 50/50 [00:11<00:00,  3.66it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 51.02it/s]
    Training: 100%|██████████| 50/50 [00:14<00:00,  3.47it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 101.73it/s]
    Training: 100%|██████████| 50/50 [00:13<00:00,  3.66it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 185.03it/s]
    Training: 100%|██████████| 50/50 [00:13<00:00,  3.64it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 61.16it/s]
    Training: 100%|██████████| 50/50 [00:13<00:00,  2.94it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 31.03it/s]
    Training: 100%|██████████| 50/50 [00:13<00:00,  3.08it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 43.17it/s]
    Training: 100%|██████████| 50/50 [00:13<00:00,  3.62it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 32.73it/s]
    Training: 100%|██████████| 50/50 [00:13<00:00,  3.78it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 35.90it/s]
    Training: 100%|██████████| 50/50 [00:16<00:00,  3.02it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 425.72it/s]
    Training: 100%|██████████| 50/50 [00:14<00:00,  5.43it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 114.58it/s]
    Training: 100%|██████████| 50/50 [00:12<00:00,  2.75it/s]
    Evaluating: 100%|██████████| 30/30 [00:01<00:00, 28.59it/s]
    Training: 100%|██████████| 50/50 [00:14<00:00,  3.93it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 66.17it/s]
    Training: 100%|██████████| 50/50 [00:11<00:00,  4.26it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 87.00it/s]
    Training: 100%|██████████| 50/50 [00:11<00:00,  4.48it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 81.14it/s]
    Training: 100%|██████████| 50/50 [00:09<00:00,  2.88it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 33.75it/s]
    Training: 100%|██████████| 50/50 [00:13<00:00,  3.60it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 78.98it/s]
    Training: 100%|██████████| 50/50 [00:11<00:00,  3.38it/s]
    Evaluating: 100%|██████████| 30/30 [00:01<00:00, 28.29it/s]
    Training: 100%|██████████| 50/50 [00:16<00:00,  2.57it/s]
    Evaluating: 100%|██████████| 30/30 [00:01<00:00, 28.66it/s]
    Training: 100%|██████████| 50/50 [00:17<00:00,  4.37it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 47.35it/s]
    Training: 100%|██████████| 50/50 [00:15<00:00,  3.33it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 151.64it/s]
    Training: 100%|██████████| 50/50 [00:10<00:00,  4.60it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 530.62it/s]
    Training: 100%|██████████| 50/50 [00:14<00:00,  3.38it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 47.22it/s]
    Training: 100%|██████████| 50/50 [00:09<00:00,  5.46it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 78.78it/s]
    Training: 100%|██████████| 50/50 [00:13<00:00,  4.06it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 446.77it/s]
    Training: 100%|██████████| 50/50 [00:12<00:00,  4.15it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 250.48it/s]
    Training: 100%|██████████| 50/50 [00:21<00:00,  2.33it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 457.38it/s]
    Training: 100%|██████████| 50/50 [00:17<00:00,  2.93it/s]
    Evaluating: 100%|██████████| 30/30 [00:01<00:00, 26.37it/s]
    Training: 100%|██████████| 50/50 [00:15<00:00,  3.60it/s]
    Evaluating: 100%|██████████| 30/30 [00:01<00:00, 28.29it/s]
    Training: 100%|██████████| 50/50 [00:13<00:00,  4.39it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 41.42it/s]
    Training: 100%|██████████| 50/50 [00:12<00:00,  3.85it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 71.74it/s]
    Training: 100%|██████████| 50/50 [00:11<00:00,  4.53it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 536.62it/s]
    Training: 100%|██████████| 50/50 [00:12<00:00,  3.95it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 55.40it/s]
    Training: 100%|██████████| 50/50 [00:11<00:00,  4.07it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 552.70it/s]
    Training: 100%|██████████| 50/50 [00:07<00:00,  6.44it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 341.78it/s]
    Training: 100%|██████████| 50/50 [00:05<00:00, 10.18it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 266.79it/s]
    Training: 100%|██████████| 50/50 [00:16<00:00,  3.03it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 85.40it/s]
    Training: 100%|██████████| 50/50 [00:17<00:00,  3.89it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 32.94it/s]
    Training: 100%|██████████| 50/50 [00:14<00:00,  3.41it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 165.52it/s]
    Training: 100%|██████████| 50/50 [00:17<00:00,  2.56it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 40.33it/s]
    Training: 100%|██████████| 50/50 [00:19<00:00,  2.74it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 35.25it/s]
    Training: 100%|██████████| 50/50 [00:13<00:00,  3.79it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 33.61it/s]
    Training: 100%|██████████| 50/50 [00:15<00:00,  3.23it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 43.23it/s]
    Training: 100%|██████████| 50/50 [00:12<00:00,  7.87it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 93.74it/s]
    Training: 100%|██████████| 50/50 [00:16<00:00,  5.13it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 268.36it/s]
    Training: 100%|██████████| 50/50 [00:20<00:00,  2.47it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 30.66it/s]
    Training: 100%|██████████| 50/50 [00:22<00:00,  2.36it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 393.25it/s]
    Training: 100%|██████████| 50/50 [00:23<00:00,  2.19it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 39.11it/s]
    Training: 100%|██████████| 50/50 [00:28<00:00,  1.09it/s]
    Evaluating: 100%|██████████| 30/30 [00:01<00:00, 23.71it/s]
    Training: 100%|██████████| 50/50 [00:16<00:00,  4.50it/s]
    Evaluating: 100%|██████████| 30/30 [00:01<00:00, 30.05it/s]
    Training: 100%|██████████| 50/50 [00:12<00:00,  3.00it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 41.78it/s]
    Training: 100%|██████████| 50/50 [00:21<00:00,  2.60it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 38.29it/s]
    Training: 100%|██████████| 50/50 [00:17<00:00,  2.90it/s]
    Evaluating: 100%|██████████| 30/30 [00:01<00:00, 29.62it/s]
    Training: 100%|██████████| 50/50 [00:14<00:00,  2.77it/s]
    Evaluating: 100%|██████████| 30/30 [00:01<00:00, 22.25it/s]
    Training: 100%|██████████| 50/50 [00:11<00:00,  4.31it/s]
    Evaluating: 100%|██████████| 30/30 [00:01<00:00, 29.60it/s]
    Training: 100%|██████████| 50/50 [00:15<00:00,  4.70it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 60.01it/s]
    Training: 100%|██████████| 50/50 [00:18<00:00,  2.75it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 217.23it/s]
    Training: 100%|██████████| 50/50 [00:11<00:00,  3.58it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 74.87it/s]
    Training: 100%|██████████| 50/50 [00:19<00:00,  2.60it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 478.71it/s]
    Training: 100%|██████████| 50/50 [00:12<00:00,  3.72it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 89.90it/s]
    Training: 100%|██████████| 50/50 [00:12<00:00,  3.57it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 34.36it/s]
    Training: 100%|██████████| 50/50 [00:13<00:00,  6.09it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 80.90it/s]
    Training: 100%|██████████| 50/50 [00:08<00:00,  6.79it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 94.98it/s]
    Training: 100%|██████████| 50/50 [00:17<00:00,  2.82it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 31.93it/s]
    Training: 100%|██████████| 50/50 [00:14<00:00,  6.55it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 84.37it/s]
    Training: 100%|██████████| 50/50 [00:19<00:00,  1.99it/s]
    Evaluating: 100%|██████████| 30/30 [00:01<00:00, 28.25it/s]
    Training: 100%|██████████| 50/50 [00:12<00:00,  4.47it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 55.45it/s]
    Training: 100%|██████████| 50/50 [00:11<00:00,  5.21it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 61.45it/s]
    Training: 100%|██████████| 50/50 [00:11<00:00,  4.25it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 88.09it/s]
    Training: 100%|██████████| 50/50 [00:08<00:00,  4.50it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 78.99it/s]
    Training: 100%|██████████| 50/50 [00:11<00:00,  4.49it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 316.52it/s]
    Training: 100%|██████████| 50/50 [00:10<00:00,  4.63it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 93.89it/s]
    Training: 100%|██████████| 50/50 [00:13<00:00,  3.83it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 31.46it/s]
    Training: 100%|██████████| 50/50 [00:11<00:00,  4.38it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 125.49it/s]
    Training: 100%|██████████| 50/50 [00:17<00:00,  3.31it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 445.89it/s]
    Training: 100%|██████████| 50/50 [00:15<00:00,  2.73it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 57.38it/s]
    Training: 100%|██████████| 50/50 [00:15<00:00,  4.41it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 58.84it/s]
    Training: 100%|██████████| 50/50 [00:18<00:00,  4.54it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 36.32it/s]
    Training: 100%|██████████| 50/50 [00:14<00:00,  2.91it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 529.71it/s]
    Training: 100%|██████████| 50/50 [00:12<00:00,  4.04it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 39.98it/s]
    Training: 100%|██████████| 50/50 [00:17<00:00,  2.90it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 94.19it/s]
    Training: 100%|██████████| 50/50 [00:16<00:00,  2.22it/s]
    Evaluating: 100%|██████████| 30/30 [00:01<00:00, 26.68it/s]
    Training: 100%|██████████| 50/50 [00:20<00:00,  4.15it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 120.18it/s]
    Training: 100%|██████████| 50/50 [00:13<00:00,  3.62it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 54.62it/s]
    Training: 100%|██████████| 50/50 [00:11<00:00,  3.18it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 30.26it/s]
    Training: 100%|██████████| 50/50 [00:17<00:00,  2.25it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 45.75it/s]
    Training: 100%|██████████| 50/50 [00:16<00:00,  3.03it/s]
    Evaluating: 100%|██████████| 30/30 [00:01<00:00, 27.61it/s]
    Training: 100%|██████████| 50/50 [00:18<00:00,  3.09it/s]
    Evaluating: 100%|██████████| 30/30 [00:01<00:00, 27.88it/s]
    Training: 100%|██████████| 50/50 [00:13<00:00,  4.04it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 72.16it/s]
    Training: 100%|██████████| 50/50 [00:12<00:00,  4.02it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 38.85it/s]
    Training: 100%|██████████| 50/50 [00:16<00:00,  3.75it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 63.11it/s]
    Training: 100%|██████████| 50/50 [00:16<00:00,  4.70it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 30.47it/s]
    Training: 100%|██████████| 50/50 [00:08<00:00,  5.97it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 494.91it/s]
    Training: 100%|██████████| 50/50 [00:15<00:00,  3.33it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 65.11it/s]
    Training: 100%|██████████| 50/50 [00:15<00:00,  2.58it/s]
    Evaluating: 100%|██████████| 30/30 [00:01<00:00, 27.56it/s]
    Training: 100%|██████████| 50/50 [00:15<00:00,  1.18it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 30.16it/s]
    Training: 100%|██████████| 50/50 [00:15<00:00,  2.84it/s]
    Evaluating: 100%|██████████| 30/30 [00:01<00:00, 26.92it/s]
    Training: 100%|██████████| 50/50 [00:25<00:00,  1.77it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 262.99it/s]
    Training: 100%|██████████| 50/50 [00:15<00:00,  1.55it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 30.25it/s]
    Training: 100%|██████████| 50/50 [00:17<00:00,  3.45it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 33.64it/s]
    Training: 100%|██████████| 50/50 [00:21<00:00,  2.93it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 188.41it/s]
    Training: 100%|██████████| 50/50 [00:22<00:00,  1.78it/s]
    Evaluating: 100%|██████████| 30/30 [00:01<00:00, 26.21it/s]
    Training: 100%|██████████| 50/50 [00:21<00:00,  2.28it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 57.58it/s]
    Training: 100%|██████████| 50/50 [00:16<00:00,  2.32it/s]
    Evaluating: 100%|██████████| 30/30 [00:01<00:00, 26.16it/s]
    Training: 100%|██████████| 50/50 [00:17<00:00,  2.57it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 109.12it/s]
    Training: 100%|██████████| 50/50 [00:14<00:00,  8.16it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 94.18it/s]
    Training: 100%|██████████| 50/50 [00:10<00:00,  4.67it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 59.94it/s]
    Training: 100%|██████████| 50/50 [00:12<00:00,  1.43it/s]
    Evaluating: 100%|██████████| 30/30 [00:01<00:00, 25.07it/s]
    Training: 100%|██████████| 50/50 [00:14<00:00,  3.51it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 165.96it/s]
    Training: 100%|██████████| 50/50 [00:17<00:00,  2.88it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 31.15it/s]
    Training: 100%|██████████| 50/50 [00:14<00:00,  5.79it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 136.58it/s]
    Training: 100%|██████████| 50/50 [00:13<00:00,  1.87it/s]
    Evaluating: 100%|██████████| 30/30 [00:01<00:00, 26.48it/s]
    Training: 100%|██████████| 50/50 [00:11<00:00,  4.28it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 42.38it/s]
    Training: 100%|██████████| 50/50 [00:14<00:00,  2.15it/s]
    Evaluating: 100%|██████████| 30/30 [00:01<00:00, 26.28it/s]
    Training: 100%|██████████| 50/50 [00:18<00:00,  2.69it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 153.13it/s]
    Training: 100%|██████████| 50/50 [00:15<00:00,  3.92it/s]
    Evaluating: 100%|██████████| 30/30 [00:01<00:00, 28.45it/s]
    Training: 100%|██████████| 50/50 [00:19<00:00,  2.45it/s]
    Evaluating: 100%|██████████| 30/30 [00:01<00:00, 26.62it/s]
    Training: 100%|██████████| 50/50 [00:15<00:00,  2.96it/s]
    Evaluating: 100%|██████████| 30/30 [00:01<00:00, 26.90it/s]
    Training: 100%|██████████| 50/50 [00:12<00:00,  3.65it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 35.56it/s]
    Training: 100%|██████████| 50/50 [00:18<00:00,  2.77it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 105.84it/s]
    Training: 100%|██████████| 50/50 [00:15<00:00,  2.25it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 44.82it/s]
    Training: 100%|██████████| 50/50 [00:12<00:00,  6.70it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 96.86it/s]
    Training: 100%|██████████| 50/50 [00:19<00:00,  2.00it/s]
    Evaluating: 100%|██████████| 30/30 [00:01<00:00, 28.59it/s]
    Training: 100%|██████████| 50/50 [00:17<00:00,  2.88it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 107.22it/s]
    Training: 100%|██████████| 50/50 [00:20<00:00,  2.69it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 38.00it/s]
    Training: 100%|██████████| 50/50 [00:14<00:00,  2.46it/s]
    Evaluating: 100%|██████████| 30/30 [00:01<00:00, 28.39it/s]
    Training: 100%|██████████| 50/50 [00:20<00:00,  2.38it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 48.29it/s]
    Training: 100%|██████████| 50/50 [00:13<00:00,  4.76it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 47.49it/s]
    Training: 100%|██████████| 50/50 [00:19<00:00,  1.77it/s]
    Evaluating: 100%|██████████| 30/30 [00:01<00:00, 25.98it/s]
    Training: 100%|██████████| 50/50 [00:18<00:00,  2.36it/s]
    Evaluating: 100%|██████████| 30/30 [00:01<00:00, 26.95it/s]
    Training: 100%|██████████| 50/50 [00:16<00:00,  2.66it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 34.25it/s]
    Training: 100%|██████████| 50/50 [00:14<00:00,  3.42it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 135.87it/s]
    Training: 100%|██████████| 50/50 [00:24<00:00,  2.88it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 29.43it/s]
    Training: 100%|██████████| 50/50 [00:22<00:00,  1.98it/s]
    Evaluating: 100%|██████████| 30/30 [00:01<00:00, 29.28it/s]
    Training: 100%|██████████| 50/50 [00:21<00:00,  3.88it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 31.66it/s]
    Training: 100%|██████████| 50/50 [00:16<00:00,  3.75it/s]
    Evaluating: 100%|██████████| 30/30 [00:01<00:00, 25.68it/s]
    Training: 100%|██████████| 50/50 [00:18<00:00,  1.84it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 36.18it/s]
    Training: 100%|██████████| 50/50 [00:20<00:00,  1.52it/s]
    Evaluating: 100%|██████████| 30/30 [00:01<00:00, 27.03it/s]
    Training: 100%|██████████| 50/50 [00:16<00:00,  2.62it/s]
    Evaluating: 100%|██████████| 30/30 [00:01<00:00, 24.60it/s]
    Training: 100%|██████████| 50/50 [00:28<00:00,  1.20it/s]
    Evaluating: 100%|██████████| 30/30 [00:01<00:00, 18.80it/s]
    Training: 100%|██████████| 50/50 [00:20<00:00,  1.37it/s]
    Evaluating: 100%|██████████| 30/30 [00:01<00:00, 28.93it/s]
    Training: 100%|██████████| 50/50 [00:22<00:00,  1.83it/s]
    Evaluating: 100%|██████████| 30/30 [00:00<00:00, 34.93it/s]

    
    Learning Performance:
    


    



    <Figure size 432x288 with 0 Axes>



![png](output_20_4.png)


***
# Part 2: Distributed DQN
***

Here you will implement a distributed version of the above DQN approach. The distribution approach can be the same as that used for the table-based distribution Q-learning algorithm from homework 3.

## init Ray


```python
ray.shutdown()
ray.init(include_webui=False, ignore_reinit_error=True, redis_max_memory=500000000, object_store_memory=5000000000)
```

## Distributed DQN agent
The idea is to speedup learning by creating actors to collect data and a model_server to update the neural network model.
- Collector: There is a simulator inside each collector. Their job is to collect exprience from the simulator, and send them to the memory server. They follow the explore_or_exploit policy, getting greedy action from model server. Also, call update function of model server to update the model.  
- Evaluator: There is a simulator inside the evaluator. It is called by the the Model Server, taking eval_model from it, and test its performance.
- Model Server: Stores the evalation and target networks. It Takes experiences from Memory Server and updates the Q-network, also replacing target Q-network periodically. It also interfaces to the evaluator periodically. 
- Memory Server: It is used to store/sample experience relays.

An image of this architecture is below. 

For this part, you should use our custom_cartpole as your enviroment. This version of cartpole is slower, which allows for the benefits of distributed experience collection to be observed. In particular, the time to generate an experience tuple needs to be non-trivial compared to the time needed to do a neural network model update. 

<span style="color:green">It is better to run the distributed DQN agent in exclusive node, not in Jupyter notebook</span>
```
Store all of your distrited DQN code into a python file.
ssh colfax (get access to the Devcloud on terminal)
qsub -I -lselect=1
python3 distributed_dqn.py
```

<img src="distributed DQN.png">

For this part of the homework you need to submit your code for distributed DQN and run experiments that vary the number of workers involved. Produce some learning curves and timing results and discuss your observations. 


```python
from memory_remote import ReplayBuffer_remote
from dqn_model import _DQNModel
import torch
from custom_cartpole import CartPoleEnv
```


```python
# Set the Env name and action space for CartPole
ENV_NAME = 'CartPole_distributed' 

# Set result saveing floder
result_floder = ENV_NAME + "_distributed"
result_file = ENV_NAME + "/results.txt"
if not os.path.isdir(result_floder):
    os.mkdir(result_floder)
torch.set_num_threads(12)
```
