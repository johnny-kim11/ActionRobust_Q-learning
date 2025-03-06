import ray

@ray.remote
def one_time_evaluate(env, agent, mode):
    evaluate_reward = 0
    # env.model.body_mass = env.model.body_mass * 2.0
    for _ in range(1):
        s = env.reset(seed=0)[0]
        done = False
        episode_reward = 0
        while not done:
            a = agent.choose_action(s, mode)  # We do not add noise when evaluating
            s_, r, t1, t2, _ = env.step(a)
            done = t1 or t2
            episode_reward += r
            s = s_
        evaluate_reward += episode_reward
    return evaluate_reward

def parell_evaluate(env, agent, mode):
    result_ids = [one_time_evaluate.remote(env, agent, mode) for i in range(5)]
    results = ray.get(result_ids)
    eval_rewards = 0
    for i in range(5):
        eval_rewards += results[i]
                
    return int(eval_rewards/5)
