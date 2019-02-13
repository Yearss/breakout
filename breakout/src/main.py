

import gym
import configparser
from dqn import DQN
import numpy as np

def main():

    env = gym.make('Breakout-v3')

    cf = configparser.ConfigParser()
    cf.read("breakout.conf")
    config = cf.options("agent")
    agent = DQN(config)
    total_reward_decade = 0

    for episode in range(config.episode):

        total_reward = 0
        state = env.reset()

        print(state.shape)
        state = agent.image_processor.colormat2bin(state)
        state_shadow = np.stack((state, state, state , state), axis=2)

        for step in range(config.step):
            env.render()
            action = agent.get_action(state_shadow)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(agent.image_processor.colormat2bin(next_state), (config.cnn_input_width, config.cnn_input_height, 1))
            next_state_shadow = np.append(next_state, state_shadow[:, :, :3], axis=2)

            total_reward += reward

            state_shadow = next_state_shadow

            if done:
                break

        print('episode {}, point: {}'.format(episode, total_reward))
        total_reward_decade += total_reward

        if episode % 10 == 0:
            print("="*20)
            print("decade point; {}".format(total_reward_decade))
            print("="*20)
            total_reward_decade = 0


if __name__ == "__main__":
    main()
