from memory_remote import ReplayBuffer_remote
from custom_cartpole import CartPoleEnv
from dqn_model import _DQNModel
from dqn_model import DQNModel

from random import uniform, randint

import gym
import torch
import time
import os
import ray
import numpy as np

ENV_NAME = 'CartPole-distributed3'
result_floder = ENV_NAME
if not os.path.isdir(result_floder):
    os.mkdir(result_floder)

FloatTensor = torch.FloatTensor
ray.shutdown()
ray.init(include_webui=False, ignore_reinit_error=True, redis_max_memory=300000000, object_store_memory=3000000000)

ACTION_DICT = {
    "LEFT": 0,
    "RIGHT": 1
}


def plot_result(total_rewards, learning_num, legend):
    print("\nLearning Performance:\n")
    episodes = []
    for i in range(len(total_rewards)):
        episodes.append(i * learning_num + 1)

    plt.figure(num=1)
    fig, ax = plt.subplots()
    plt.plot(episodes, total_rewards)
    plt.title('performance')
    plt.legend(legend)
    plt.xlabel("Episodes")
    plt.ylabel("total rewards")
    plt.show()


@ray.remote
class RLAgent_model_server():
    def __init__(self, env, hyper_params, memo_server):
        self.memory_server = memo_server
        self.env = env
        self.max_episode_steps = env._max_episode_steps

        self.beta = hyper_params['beta']
        self.training_episodes = hyper_params['training_episodes']
        self.test_interval = hyper_params['test_interval']

        action_space = len(ACTION_DICT)
        self.episode = 0
        self.steps = 0
        self.best_reward = 0
        self.learning = True
        self.action_space = action_space

        state = env.reset()
        input_len = len(state)
        output_len = action_space
        self.eval_model = DQNModel(input_len, output_len, learning_rate=hyper_params['learning_rate'])
        self.use_target_model = hyper_params['use_target_model']
        if self.use_target_model:
            self.target_model = DQNModel(input_len, output_len)

        self.batch_size = hyper_params['batch_size']
        self.update_steps = hyper_params['update_steps']
        self.model_replace_freq = hyper_params['model_replace_freq']
        self.collector_done = False
        self.results = []

        self.initial_epsilon = 1
        self.final_epsilon = hyper_params['final_epsilon']
        self.epsilon_decay_steps = hyper_params['epsilon_decay_steps']
        self.replace_targe_cnt = 0
        self.epsilon = 1
        self.eval_models_seq = 1

    def update_batch(self):
        # Get memory sample
        batch = ray.get(self.memory_server.sample.remote(self.batch_size))
        if not batch:
            return
        (states, actions, reward, next_states, is_terminal) = batch

        # Setting torch value
        states = states
        next_states = next_states
        terminal = FloatTensor([0 if t else 1 for t in is_terminal])
        reward = FloatTensor(reward)
        batch_index = torch.arange(self.batch_size, dtype=torch.long)

        # Current Q Values
        _, q_values = self.eval_model.predict_batch(states)
        q_values = q_values[batch_index, actions]

        # Calculate target
        if self.use_target_model:
            actions, q_next = self.target_model.predict_batch(next_states)
        else:
            actions, q_next = self.eval_model.predict_batch(next_states)
        max_q_next, index = torch.max(q_next, dim=1)
        q_target = reward + self.beta * max_q_next * terminal
        # Update model
        self.eval_model.fit(q_values, q_target)

    def replace_target_model(self):
        if self.use_target_model and self.steps % self.model_replace_freq == 0:
            self.target_model.replace(self.eval_model)

    def evaluate_result(self):
#         print(self.episode, self.training_episodes)
        self.episode += 1
        if self.episode % self.test_interval == 0:
            self.save_model()
#             evaluation_worker_gg.remote(self.env, self.memory_server, self.eval_model, self.test_interval)

    def save_model(self):
        filename = "/best_model{0}.pt".format(self.eval_models_seq)
        self.eval_model.save(result_floder + filename)
        self.memory_server.add_evamodel_dir.remote(result_floder + filename)
        self.eval_models_seq += 1

    def ask_evaluate(self):
        if len(self.eval_models) == 0:
            return None, self.episode >= self.training_episodes

        eval_model, is_done = self.eval_models[0]
        del self.eval_models[0]
        return eval_model, is_done

    def get_collector_done(self):
        return self.episode >= self.training_episodes

    def linear_decrease(self, initial_value, final_value, curr_steps, final_decay_steps):
        decay_rate = curr_steps / final_decay_steps
        if decay_rate > 1:
            decay_rate = 1
        return initial_value - (initial_value - final_value) * decay_rate

    def explore_or_exploit_policy(self, state):
        self.epsilon = self.linear_decrease(self.initial_epsilon, 
                                            self.final_epsilon, 
                                            self.steps,
                                            self.epsilon_decay_steps)
        return randint(0, self.action_space - 1) if uniform(0, 1) < self.epsilon else self.greedy_policy(state)

    def greedy_policy(self, state):
        return self.eval_model.predict(state)

    def add_results(self, result):
        self.results.append(result)

    def get_reuslts(self):
        return self.results

    def update_and_replace_model(self):
        self.steps += 1
        if self.steps % self.update_steps != 0:
            self.update_batch()
        self.replace_target_model()

@ray.remote
def collecting_worker(env, mod_server, mem_server, batch_size):
#     learn_done = False
#     while not learn_done:
    for _ in range(batch_size):
        exp_set, state, done, steps = [], env.reset(), False, 0
        while steps < env._max_episode_steps and not done:
            action = ray.get(mod_server.explore_or_exploit_policy.remote(state))
            new_state, reward, done, info = env.step(action)
            mem_server.add.remote(state, action, reward, new_state, done)
            mod_server.update_and_replace_model.remote()
            state = new_state
            steps += 1
        mod_server.evaluate_result.remote()
#         learn_done = ray.get(mod_server.get_collector_done.remote())

@ray.remote
def evaluation_worker(env, mem_server, trials):
    eval_model = DQNModel(len(env.reset()), len(ACTION_DICT))
    learn_done, filedir = False, ""
    while not learn_done:
        filedir, learn_done = ray.get(mem_server.get_evaluate_filedir.remote())
        if not filedir:
            continue
        eval_model.load(filedir)
        start_time, total_reward = time.time(), 0
        for _ in range(trials):
            state, done, steps = env.reset(), False, 0
            while steps < env._max_episode_steps and not done:
                steps += 1
                state, reward, done, _ = env.step(eval_model.predict(state))
                total_reward += reward
        mem_server.add_results.remote(total_reward / trials)

@ray.remote
def evaluation_worker_test2(env, mem_server, eval_model, trials):
    total_reward = 0
    for _ in range(trials):
        state, done, steps = env.reset(), False, 0
        while steps < env._max_episode_steps and not done:
            steps += 1
            state, reward, done, _ = env.step(eval_model.predict(state))
            total_reward += reward
    return total_reward / trials

@ray.remote
def evaluation_worker_gg(env, mem_server, eval_model, trials=30):
    start_time, total_reward = time.time(), 0
    for _ in range(trials):
        state, done, steps = env.reset(), False, 0
        while steps < env._max_episode_steps and not done:
            steps += 1
            action = eval_model.predict(state)
            state, reward, done, _ = env.step(action)
            total_reward += reward

    print(total_reward / trials, (time.time() - start_time))
    mem_server.add_results.remote(total_reward / trials)

class distributed_RL_agent():
    def __init__(self, env, parms):
        self.memory_server = ReplayBuffer_remote.remote(parms['memory_size'], parms['test_interval'], parms['training_episodes'])
        self.model_server = RLAgent_model_server.remote(env, parms, self.memory_server)
        self.env, self.parms = env, parms

    def learn_and_evaluate(self):
        workers_id = []
        batch_size = self.parms['training_episodes'] // self.parms['workers'][0]
        for _ in range(self.parms['workers'][0]):
            workers_id.append(collecting_worker.remote(self.env, self.model_server, self.memory_server, batch_size))

        all_results = []
        if self.parms['do_test']:
            eval_model = DQNModel(len(env.reset()), len(ACTION_DICT))
            learn_done, filedir = False, ""
            workers_num = self.parms['workers'][1]
            interval = self.parms['test_interval']//workers_num
            while not learn_done:
                filedir, learn_done = ray.get(self.memory_server.get_evaluate_filedir.remote())
                if not filedir:
                    continue
                eval_model.load(filedir)
                start_time, total_reward = time.time(), 0
                eval_workers = []
                for _ in range(workers_num):
                    eval_workers.append(evaluation_worker_test2.remote(self.env, self.memory_server, eval_model, interval))
                    
                avg_reward = sum(ray.get(eval_workers))/workers_num
                print(filedir, avg_reward, (time.time() - start_time))
                all_results.append(avg_reward)

        return all_results


hyperparams = {
    'epsilon_decay_steps': 7000,
    'final_epsilon': 0.1,
    'batch_size': 10,
    'update_steps': 5,
    'memory_size': 2000,
    'beta': 0.99,
    'model_replace_freq': 2000,
    'learning_rate': 0.0003,
    'use_target_model': True,
    'workers': (12, 4),
    'do_test': True,
    'initial_epsilon': 1,
    'steps': 0,
    'training_episodes': 7000,
    'test_interval': 50
}


start_time = time.time()
env = CartPoleEnv()
env.reset()
agent = distributed_RL_agent(env, hyperparams)
result = agent.learn_and_evaluate()
print(result)
print(time.time() - start_time)
# plot_result(result, test_interval, ["batch_update with target_model"])
print("Done!!")
