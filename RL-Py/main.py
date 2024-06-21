from mlagents_envs.environment import UnityEnvironment

if __name__ == '__main__':
    env = UnityEnvironment(file_name='../../Build/RLSample')
    env.reset()
    behavior_name = list(env.behavior_specs.keys())[0]
    print('name of behavior: {}'.format(behavior_name))
    spec = env.behavior_specs[behavior_name]
    
    for ep in range(10):
        env.reset()
        decision_steps, terminal_steps = env.get_steps(behavior_name)
        
        tracked_agent = -1
        done = False
        ep_reward = 0
        
        while not done:
            if tracked_agent == -1 and len(decision_steps) >= 1:
                tracked_agent = decision_steps.agent_id[0]
            
            action = spec.action_spec.random_action(len(decision_steps))
            env.set_actions(behavior_name, action)
            
            env.step()
            
            decision_steps, terminal_steps = env.get_steps(behavior_name)
            if tracked_agent in decision_steps:
                ep_reward += decision_steps[tracked_agent].reward
            if tracked_agent in terminal_steps:
                ep_reward += terminal_steps[tracked_agent].reward
                done = True
            
        print('total reward for ep: {}'.format(ep_reward))
    env.close()
    