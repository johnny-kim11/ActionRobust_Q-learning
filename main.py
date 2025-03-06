import gym
import torch
import numpy as np

from replay_buffer import ReplayBuffer
from EB_DDPG import EB_DDPG
from evaluate import parell_evaluate    


if __name__ == '__main__':
    env_name = ['Swimmer-v4', 'Hopper-v4', 'LunarLanderContinuous-v2', 'Reacher-v4', 'InvertedPendulum-v4']
    env_index = 4
    threshold = 990
    
    env = gym.make(env_name[env_index])
    env_evaluate = gym.make(env_name[env_index])  # When evaluating the policy, we need to rebuild an environment
    
    # Set random seed
    seed = 0
    env.action_space.seed(seed)
    env_evaluate.action_space.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    max_episode_steps = 1000 # env._max_episode_steps  # Maximum number of steps per episode
    print("env={}".format(env_name[env_index]))
    print("state_dim={}".format(state_dim))
    print("action_dim={}".format(action_dim))
    print("max_action={}".format(max_action))
    print("max_episode_steps={}".format(max_episode_steps))

    agent = EB_DDPG(state_dim, action_dim, max_action)
    replay_buffer = ReplayBuffer(state_dim, action_dim)
    # Build a tensorboard
    checkpoint = "./ddpg_trained_models/env_{}_step_".format(env_name[env_index])

    noise_std = 0.1 * max_action  # the std of Gaussian noise for exploration
    max_train_steps = 2e6  # Maximum number of training steps
    random_steps = 20e3  # Take the random actions in the beginning for the better exploration #50
    update_freq = 50  # Take 50 steps,then update the networks 50 times
    evaluate_freq = 1e3  # Evaluate the policy every 'evaluate_freq' steps
    evaluate_num = 0  # Record the number of evaluations
    evaluate_rewards = []  # Record the rewards during the evaluating
    total_steps = 0  # Record the total steps during the training
    print("noise_std={}".format(noise_std))
    
    
    while total_steps < max_train_steps:
        s = env.reset()[0]
        episode_steps = 0
        done = False
        while not done:
            episode_steps += 1
            if total_steps < random_steps:  # Take the random actions in the beginning for the better exploration
                a = env.action_space.sample()
            else:
                # Add Gaussian noise to actions for exploration
                a = agent.choose_action(s, 'RS')
                a = (a + np.random.normal(0, noise_std, size=action_dim)).clip(-max_action, max_action)
            s_, r, t1, t2, _ = env.step(a)
            done = t1 or t2

            if done and episode_steps != max_episode_steps:
                dw = True
            else:
                dw = False
            replay_buffer.store(s, a, r, s_, dw)  # Store the transition
            s = s_

            # Take 50 steps,then update the networks 50 times
            if total_steps >= random_steps and total_steps % update_freq == 0:
                for _ in range(update_freq):
                    agent.learn(replay_buffer, 'RS')

            # Evaluate the policy every 'evaluate_freq' steps
            if (total_steps + 1) % evaluate_freq == 0:
                evaluate_num += 1
                evaluate_reward = parell_evaluate(env_evaluate, agent, 'RS')
                evaluate_rewards.append(evaluate_reward)
                print("total_steps:{} \t evaluate_num:{} \t evaluate_reward:{}".format(total_steps+1, evaluate_num, evaluate_reward))
                if evaluate_reward > threshold:
                    input_checkpoint = checkpoint + str(total_steps+1) + ".pth"
                    agent.save(input_checkpoint)
                    print("Agent saved")
                    
            total_steps += 1