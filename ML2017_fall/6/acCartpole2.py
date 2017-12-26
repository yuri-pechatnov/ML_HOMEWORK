import threading
import numpy as np
import tensorflow as tf
import pylab
import time
import gym
from keras.layers import Dense, Concatenate, Input, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K


K.set_learning_phase(1)

make_env = lambda: gym.make('BipedalWalker-v2')

episode = 0
scores = []

EPISODES = 5000

class A3CAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.actor_lr = 0.1
        self.critic_lr = 0.05
        self.discount_factor = .8
        self.hidden1, self.hidden2 = 48, 48
        self.threads = 4

        self.actor, self.critic = self.build_model()

        self.optimizer = [self.actor_optimizer(), self.critic_optimizer()]

        self.sess = tf.InteractiveSession()
        K.set_session(self.sess)
        self.sess.run(tf.global_variables_initializer())


    def build_model(self):
        state = Input(batch_shape=(None,  self.state_size))
        shared = Dense(self.hidden1, input_dim=self.state_size, activation='relu', kernel_initializer='glorot_uniform')(state)

        actor_hidden = Dense(self.hidden2, activation='tanh', kernel_initializer='glorot_uniform')(shared)
        actor_hidden = Dropout(0.1)(actor_hidden)
        #actor_hidden = Dense(self.hidden2, activation='relu', kernel_initializer='glorot_uniform')(actor_hidden)
        #actor_hidden = Dropout(0.3)(actor_hidden)
        actor_hidden = Dense(self.hidden2, activation='relu', kernel_initializer='glorot_uniform')(actor_hidden)
        action_preoutput = Dense(self.action_size * 2, activation='tanh', kernel_initializer='glorot_uniform')(actor_hidden)
        #action_preoutput = Dense(self.action_size * 2, activation='relu', kernel_initializer='glorot_uniform')(shared)
        action_mean = Dense(self.action_size, activation='softsign', kernel_initializer='zeros')(action_preoutput)
        action_variance = Dense(self.action_size, activation='softplus', bias_initializer='ones')(action_preoutput)
        action_params = Concatenate()([action_mean, action_variance])

        value_hidden = Dense(self.hidden2, activation='relu', kernel_initializer='uniform')(shared)
        state_value = Dense(1, activation='linear', kernel_initializer='zeros')(value_hidden)

        actor = Model(inputs=state, outputs=action_params)
        critic = Model(inputs=state, outputs=state_value)

        actor._make_predict_function()
        critic._make_predict_function()

        actor.summary()
        critic.summary()

        return actor, critic

    def actor_optimizer(self):
        action = K.placeholder(shape=(None, self.action_size))
        advantages = K.placeholder(shape=(None, ))

        policy = self.actor.output

        good_prob = K.sum(((action - policy[:, :self.action_size]) / (policy[:, self.action_size:] + 1e-9)) ** 2 + K.log(policy[:, self.action_size:] + 1e-9), axis=1)
        #good_prob = -K.sum(-2 * ((action - policy[:, :self.action_size]) / (policy[:, self.action_size:] + 1e-9)) - K.log(policy[:, self.action_size:] + 1e-9), axis=1)

        #good_prob = K.sum(action * policy, axis=1)
        #eligibility = K.log(good_prob + 1e-10) * K.stop_gradient(advantages)
        #eligibility = good_prob * K.stop_gradient(advantages)
        eligibility = good_prob * K.stop_gradient(advantages)
        #eligibility = -K.stop_gradient(advantages)
        loss = K.sum(eligibility)

        entropy = K.sum((policy ** 2) * K.log((policy ** 2) + 1e-10), axis=1)

        actor_loss = loss + 0.01 * entropy
        #actor_loss = K.sum((action - policy[:, :self.action_size]))# / (policy[:, self.action_size:] + 1e-9)) + K.sum(K.log(policy[:, self.action_size:] + 1e10))

        optimizer = Adam(lr=self.actor_lr)
        updates = optimizer.get_updates(self.actor.trainable_weights, [], actor_loss)
        train = K.function([self.actor.input, action, advantages], [], updates=updates)
        return train

    def critic_optimizer(self):
        discounted_reward = K.placeholder(shape=(None, ))

        value = self.critic.output

        loss = K.mean(K.square(discounted_reward - value))

        optimizer = Adam(lr=self.critic_lr)
        updates = optimizer.get_updates(self.critic.trainable_weights, [], loss)
        train = K.function([self.critic.input, discounted_reward], [], updates=updates)
        return train

    def train(self):
        # self.load_model('./save_model/cartpole_a3c.h5')
        agents = [Agent(i, self.actor, self.critic, self.optimizer, self.discount_factor,
                        self.action_size, self.state_size) for i in range(self.threads)]

        for agent in agents:
            agent.start()

        global episode
        env = make_env()
        while episode < EPISODES:
            #time.sleep(2)

            plot = scores[:]
            pylab.plot(range(len(plot)), plot, 'b')
            pylab.savefig("./save_graph/cartpole_a3c.png")

            self.save_model('./save_model/cartpole_a3c.h5')



            state = env.reset()
            for i in range(1600):
                action = agents[0].get_action(state)
                next_state, reward, done, _ = env.step(action)
                if i % 10 == 0:
                    env.render()
                if done:
                    break

        del env


        for agent in agents:
            agent.join()

    def save_model(self, name):
        self.actor.save_weights(name + "_actor.h5")
        self.critic.save_weights(name + "_critic.h5")

    def load_model(self, name):
        self.actor.load_weights(name + "_actor.h5")
        self.critic.load_weights(name + "_critic.h5")

class Agent(threading.Thread):
    def __init__(self, index, actor, critic, optimizer, discount_factor, action_size, state_size):
        threading.Thread.__init__(self)

        self.states = []
        self.rewards = []
        self.actions = []
        self.positions = []

        self.index = index
        self.actor = actor
        self.critic = critic
        self.optimizer = optimizer
        self.discount_factor = discount_factor
        self.action_size = action_size
        self.state_size = state_size

    def run(self):
        global episode
        env = make_env()
        max_pos = 0
        while episode < EPISODES:
            self.random_bias_rate = max(0.05, 0.7 * np.exp(-episode / 400))
            state = env.reset()
            score = 0
            steps = 0
            position = 0
            while True:
                action = self.get_action(state)
                next_state, reward, done, _ = env.step(action)
                newPosition = env.unwrapped.hull.position[0]
                max_pos = max(max_pos, position)
                #reward += position / (steps + 1)
                #if reward == -100:
                #    reward += position

                #if done and position + 3 < max_pos:
                #    reward -= 100

                score += reward
                steps += 1

                self.memory(state, position, action, reward)

                state = next_state
                position = newPosition

                if steps > 400 and steps % 101 == 0 and position > 0 and steps / position > 50:
                    self.random_bias_rate *= 1.1
                    #~ donrue
                #if steps > 200 and episode < 1000:
                #    done = True

                if done:
                    episode += 1
                    print("episode: %4d" % episode, "/ random bias rate: %0.2f " % self.random_bias_rate,
                          "steps: %4d " % steps, " position: %2.3f " % position, "/ score: %0.3f " % score)
                    #~ print(self.states)
                    #~ print(self.actions)
                    #~ print(self.rewards)
                    scores.append(score)
                    self.train_episode(score != 500)
                    break

    # In Policy Gradient, Q function is not available.
    # Instead agent uses sample returns for evaluating policy
    def discount_rewards(self, rewards, done=True):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        if not done:
            running_add = self.critic.predict(np.reshape(self.states[-1], (1, self.state_size)))[0]
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.discount_factor + rewards[t]
            posAdvantage = 0
            #~ if len(rewards) > 300 and t + 100 < len(rewards):
                #~ posAdvantage = 100 * self.positions[-100] / (len(rewards) - 100)
            #~ if t + 100 > len(rewards):
                #~ posAdvantage -= 10
            discounted_rewards[t] = running_add + posAdvantage
        return discounted_rewards

    def memory(self, state, position, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.positions.append(position)

    def train_episode(self, done):
        discounted_rewards = self.discount_rewards(self.rewards, done)

        values = self.critic.predict(np.array(self.states))
        values = np.reshape(values, len(values))

        advantages = discounted_rewards - values

        self.optimizer[0]([self.states, self.actions, advantages])
        self.optimizer[1]([self.states, discounted_rewards])
        self.states, self.actions, self.rewards, self.positions = [], [], [], []

    def get_action(self, state):
        try:
            policy = self.actor.predict(np.reshape(state, [1, self.state_size]))[0]
            assert(not np.isnan(policy[0]))
            action = np.random.normal(loc=policy[:self.action_size], scale=policy[self.action_size:] * self.random_bias_rate)
            action = np.min([np.ones_like(action), action], axis=0)
            action = np.max([-np.ones_like(action), action], axis=0)
        except:
            print("=(")
            print(state)
            print(policy)
            print(action)
        return action


env = make_env()

state_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]

env.close()

global_agent = A3CAgent(state_size, action_size)
global_agent.train()
