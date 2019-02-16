

import gym
import configparser
from dqn import DQN
import numpy as np
import os

def main():

    print(os.getcwd())

    env = gym.make('Breakout-v4')

    config = configparser.ConfigParser()
    config.read("../breakout.conf")
    agent = DQN(env, config)

    for episode in range(config.getint('agent', 'episode')):

        total_reward = 0
        state = env.reset()

        state = agent.image_processor.colormat2bin(state, config)
        state_shadow = np.stack((state, state, state , state), axis=2)

        for step in range(config.getint('agent', 'max_step')):
            env.render()
            action = agent.get_action(state_shadow, config)
            next_state, reward, done, _ = env.step(action)
            # print(state_shadow, action, next_state, reward)
            # print("="*100)
            # next_state = np.reshape(agent.image_processor.colormat2bin(next_state, config),
            #                         (config.getint('agent', 'cnn_input_width'),
            #                          config.getint('agent', 'cnn_input_height'), 1))
            next_state = agent.image_processor.colormat2bin(next_state, config)
            next_state_shadow = np.stack((next_state, state_shadow[:, :, 1],
                                          state_shadow[:, :, 2],
                                          state_shadow[:, :, 3]), axis=2)


            total_reward += reward

            agent.percieve(state_shadow,action,reward, next_state_shadow, done, config)

            state_shadow = next_state_shadow

            if done:
                break

        print('episode {}, point: {}'.format(episode, total_reward))

        # agent.summary_writer.add_summary(dqn.reward_summary, global_step=episode)


if __name__ == "__main__":
    main()
