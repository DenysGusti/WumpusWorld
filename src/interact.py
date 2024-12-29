from math import comb

from wumpus import Wumpus
from tqdm import tqdm
import numpy as np
from aufgabe2 import Agent

size = (4, 4)
n_repeat = 10000

if __name__ == "__main__":
    env = Wumpus(seed=2025, size=size)
    agent = Agent(size=size)

    # action_names = ['FORWARD', 'LEFT', 'RIGHT', 'GRAB', 'SHOOT', 'CLIMB']
    # percept_names = '[STENCH, BREEZE, GLITTER, BUMP, SCREAM]'

    if all_worlds := False:
        for p_crit in [0., 0.21, 0.32, 0.45, 0.56, 0.7, 0.87, 1.]:
            print(f"{p_crit = }")
            cum_rewards = np.zeros(2 ** 23)

            for run in tqdm(i for i in range(2 ** 23) if i % 16 != 0):
                # pits_bits = (run >> 8) & 0x7FFF
                # k = pits_bits.bit_count()
                # if not (2 <= k <= 4):
                #     continue

                agent.new_episode()
                agent.kb.p_crit = p_crit
                terminated = False
                percept = env.reset(seed=run)
                reward = 0
                # print(f"{run = }\n{bin(run)[2:]}")
                # print(f"gold: {env.pos_gold}, wumpus: {env.pos_wumpus}, pits: {list(env.pits)}")

                while not terminated:
                    action = agent.get_action(percept, reward)
                    percept, reward, terminated, info = env.step(action)
                    if info:
                        tqdm.write(info)
                        print(run)
                        raise ValueError("Something went wrong")
                    # tqdm.write(f"{action_names[action]}\tpercept: {percept_names} = {str(percept)}")
                    cum_rewards[run] += reward

            # if cum_rewards[run] < -900:
            #     print(agent.kb)
            #     tqdm.write(f"[{run + 1:02d}] Total reward={cum_rewards[run]}")

            pits_sum_reward = [0. for _ in range(16)]
            for seed in range(2 ** 23):
                if seed % 16 != 0:
                    pits_bits = (seed >> 8) & 0x7FFF
                    pits = pits_bits.bit_count()
                    pits_sum_reward[pits] += cum_rewards[seed]

            p = 0.2
            n = 15

            weights = [(p ** k) * ((1 - p) ** (n - k)) for k in range(16)]

            average_total_reward = np.dot(pits_sum_reward, weights) / 240

            for k, s in enumerate(pits_sum_reward):
                avg = s / (comb(n, k) * 240)
                tqdm.write(f"Pits: {k}, Average total reward: {avg}")

            tqdm.write(f"Average total reward: {average_total_reward}")

    elif random_worlds := True:
        cum_rewards = np.zeros(n_repeat)

        for run in tqdm(range(n_repeat)):
            agent.new_episode()
            terminated = False
            percept = env.reset()
            reward = 0
            # if run != 424:
            #     continue
            # print(f"{run = }")
            # print(f"gold: {env.pos_gold}, wumpus: {env.pos_wumpus}, pits: {list(env.pits)}")

            while not terminated:
                action = agent.get_action(percept, reward)
                percept, reward, terminated, info = env.step(action)
                # print(agent.kb)
                # print(agent.t)
                if info:
                    tqdm.write(info)
                    print(run)
                    raise ValueError("Something went wrong")
                # tqdm.write(f"{action_names[action]}\tpercept: {percept_names} = {str(percept)}")
                cum_rewards[run] += reward

            # if cum_rewards[run] < -900:
            #     print(agent.kb)
            #     tqdm.write(f"[{run + 1:02d}] Total reward={cum_rewards[run]}")
        average_total_reward = np.mean(cum_rewards)
        tqdm.write(f"Average total reward: {average_total_reward}")
    else:
        cum_rewards = np.zeros(2 ** 23)

        for run in tqdm([0b0000_0000_0000_000_1111_1111]):
            agent.new_episode()
            terminated = False
            percept = env.reset(run)
            reward = 0
            print(f"{run = }")
            print(f"gold: {env.pos_gold}, wumpus: {env.pos_wumpus}, pits: {list(env.pits)}")

            while not terminated:
                action = agent.get_action(percept, reward)
                percept, reward, terminated, info = env.step(action)
                print(agent.kb)
                print(agent.t)
                if info:
                    tqdm.write(info)
                    print(run)
                    raise ValueError("Something went wrong")
                # tqdm.write(f"{action_names[action]}\tpercept: {percept_names} = {str(percept)}")
                cum_rewards[run] += reward

            tqdm.write(f"[{run + 1:02d}] Total reward={cum_rewards[run]}")
