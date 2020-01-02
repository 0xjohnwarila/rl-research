import ray
import time
from copy import deepcopy
import matplotlib.pyplot as plt
from random import randint, choice
import pickle

from environment import FrozenLakeEnv, MAPS

#TransitionProb = [0.7, 0.1, 0.1, 0.1]

# Map initialization
map_8 = (MAPS["8x8"], 8)
map_16 = (MAPS["16x16"], 16)
map_32 = (MAPS["32x32"], 32)
MAP = map_8
map_size = MAP[1]
run_time = {}

# Helpers

def evaluate_policy(env, policy, trials = 1000):
    total_reward = 0
    for _ in range(trials):
        env.reset()
        done = False
        observation, reward, done, info = env.step(policy[0])
        total_reward += reward
        while not done:
            observation, reward, done, info = env.step(policy[observation])
            total_reward += reward
    return total_reward / trials

def evaluate_policy_discounted(env, policy, discount_factor, trials = 1000):
    total_reward = 0
    for _ in range(trials):
        env.reset()
        done = False
        i = 0
        observation, reward, done, info = env.step(policy[0])
        total_reward += (discount_factor**i)*reward
        while not done:
            i += 1
            observation, reward, done, info = env.step(policy[observation])
            total_reward += (discount_factor**i)*reward
    return total_reward / trials

def print_results(v, pi, map_size, env, beta, name):
    v_np, pi_np = np.array(v), np.array(pi)
    print("\nState Value:\n")
    print(np.array(v_np[:-1]).reshape((map_size,map_size)))
    print("\nPolicy:\n")
    print(np.array(pi_np[:-1]).reshape((map_size,map_size)))
    print("\nAverage reward: {}\n".format(evaluate_policy(env, pi)))
    print("Avereage discounted reward: {}\n".format(evaluate_policy_discounted(env, pi, discount_factor = beta)))
    print("State Value image view:\n")
    plt.imshow(np.array(v_np[:-1]).reshape((map_size,map_size)))
    pickle.dump(v, open(name + "_" + str(map_size) + "_v.pkl", "wb"))
    pickle.dump(pi, open(name + "_" + str(map_size) + "_pi.pkl", "wb"))

# Init Ray
ray.shutdown()
ray.init(include_webui=False, ignore_reinit_error=True, redis_max_memory=10000000, object_store_memory=1008643200)

# Distribute Value Iteration
@ray.remote
class VI_server_v1(object):
    def __init__(self,size):
        self.v_current = [0] * size
        self.pi = [0] * size
        self.v_new = [0] * size

    def get_value_and_policy(self):
        return self.v_current, self.pi

    def update(self, update_index, update_v, update_pi):
        self.v_new[update_index] = update_v
        self.pi[update_index] = update_pi

    def get_error_and_update(self):
        max_error = 0
        for i in range(len(self.v_current)):
            error = abs(self.v_new[i] - self.v_current[i])
            if error > max_error:
                max_error = error
            self.v_current[i] = self.v_new[i]

        return max_error

@ray.remote
def VI_worker(VI_server, data, worker_id, start_state, end_state):
    env, workers_num, beta, epsilon = data
    A = env.GetActionSpace()
    S = env.GetStateSpace()

    V, _ = ray.get(VI_server.get_value_and_policy.remote())

    for update_state in range(start_state, end_state):
        # Bellman Backup
        max_v = float('-inf')
        max_a = 0

        for action in range(A):
            reward = env.GetReward(update_state, action)
            state_sum = 0
            for state_prob in env.GetSuccessors(update_state, action):
                state_sum += state_prob[1] * V[state_prob[0]]
            state_sum *= beta
            reward += state_sum
            if reward > max_v:
                max_v = reward
                max_a = action

        VI_server.update.remote(update_state, max_v, max_a)

def sync_value_iteration_distributed(env, beta = 0.999, epsilon = 0.01, workers_num = 4, stop_steps = 2000):
    S = env.GetStateSpace()
    VI_server = VI_server_v1.remote(S)
    workers_list = []
    data_id = ray.put((env, workers_num, beta, epsilon))

    batch_size = int(S / workers_num)
    start_state = 0
    end_state = batch_size

    for i in range(workers_num):
        w_id = VI_worker.remote(VI_server, data_id, i, start_state, end_state)
        workers_list.append(w_id)
        start_state += batch_size
        if start_state > S:
            start_state = S
        end_state += batch_size
        if end_state > S:
            end_state = S

    error = float('inf')
    while error > epsilon:
        start_state = 0
        end_state = batch_size

        for i in range(workers_num):
            finished_worker_id = ray.wait(workers_list, num_returns = 1, timeout = None)[0][0]
            finish_worker = ray.get(finished_worker_id)
            workers_list.remove(finished_worker_id)

            w_id = VI_worker.remote(VI_server, data_id, finish_worker, start_state, end_state)
            workers_list.append(w_id)

            start_state += batch_size
            if start_state > S:
                start_state = S
            end_state += batch_size
            if end_state > S:
                end_state = S

        error = ray.get(VI_server.get_error_and_update.remote())
    
    v, pi = ray.get(VI_server.get_value_and_policy.remote())
    return v, pi


beta = 0.999
env = FrozenLakeEnv(desc = MAP[0], is_slippery = True)
print("Game Map:")
env.render()

start_time = time.time()
v, pi = sync_value_iteration_distributed(env, beta = beta, workers_num = 4)
v_np, pi_np = np.array(v), np.array(pi)
end_time = time.time()
run_time['Sync dis'] = end_time - start_time
print("time:", run_time['Sync distributed v2'])
print_results(v, pi, map_size, env, beta, 'dist_vi')
