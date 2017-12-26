import multiprocessing, threading, gym, os, shutil
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
import numpy as np
from matplotlib import pylab
import time

# GAME = 'CartPole-v1'
GAME = 'BipedalWalker-v2'

def env_make():
    return gym.make(GAME)

OUTPUT_GRAPH = False
LOG_DIR = './log'
N_WORKERS = 4
# MAX_GLOBAL_EP = 200
MAX_GLOBAL_EP = 2000000
GLOBAL_NET_SCOPE = 'Global_Net'
UPDATE_GLOBAL_ITER = 10
# GAMMA = 0.96
GAMMA = 0.999
ENTROPY_BETA = 0.005
ACTOR_LEARNING_RATE = 0.00002
CRITIC_LEARNING_RATE = 0.0001
GLOBAL_RUNNING_R = [0]
GLOBAL_EPOCH = 0

env = env_make()

STATE_SIZE = env.observation_space.shape[0]
ACTION_SIZE = env.action_space.shape[0]
ACTION_BOUNDS = [env.action_space.low, env.action_space.high]

#~ OPTIMIZER = tf.train.AdamOptimizer
OPTIMIZER = tf.train.RMSPropOptimizer

ACTOR_OPTIMIZER = OPTIMIZER(ACTOR_LEARNING_RATE, name='AdamActor')
CRITIC_OPTIMIZER = OPTIMIZER(CRITIC_LEARNING_RATE, name='AdamCritic')

class ACNetBase:
    def _build_net(self):
        w_init = tf.contrib.layers.xavier_initializer()
        # predict action
        with tf.variable_scope('actor'):
            nn = InputLayer(self.s, name='in')
            nn = DenseLayer(nn, n_units=500, act=tf.nn.relu6, W_init=w_init, name='la')
            #nn = DenseLayer(nn, n_units=500, act=tf.nn.relu6, W_init=w_init, name='la2')
            nn = DenseLayer(nn, n_units=300, act=tf.nn.relu6, W_init=w_init, name='la2')
            mu = DenseLayer(nn, n_units=ACTION_SIZE, act=tf.nn.tanh, W_init=w_init, name='mu')
            sigma = DenseLayer(nn, n_units=ACTION_SIZE, act=tf.nn.softplus, W_init=w_init, name='sigma')
            self.mu = mu.outputs
            self.sigma = sigma.outputs

        # predict value of state
        with tf.variable_scope('critic'):
            nn = InputLayer(self.s, name='in')
            nn = DenseLayer(nn, n_units=500, act=tf.nn.relu6, W_init=w_init, name='lc')
            nn = DenseLayer(nn, n_units=200, act=tf.nn.relu6, W_init=w_init, name='lc2')
            v = DenseLayer(nn, n_units=1, W_init=w_init, name='v')
            self.v = v.outputs

    def choose_action(self, s):
        s = s[np.newaxis, :]
        return self.session.run(self.A, {self.s: s})[0]

    def get_variance(self, s):
        s = s[np.newaxis, :]
        return self.session.run(self.sigma, {self.s: s})[0]


class ACNetGlobal(ACNetBase):
    def __init__(self, session):
        self.session = session
        with tf.variable_scope('global_net'):
            self.s = tf.placeholder(tf.float32, [None, STATE_SIZE], 'S')
            self._build_net()
            self.actor_params = tl.layers.get_variables_with_name('global_net/actor', True, False)
            self.critic_params = tl.layers.get_variables_with_name('global_net/critic', True, False)

            normal_dist = tf.contrib.distributions.Normal(self.mu, self.sigma)

            with tf.name_scope('choose_a'):
                self.A = tf.clip_by_value(tf.squeeze(normal_dist.sample(1), axis=0), *ACTION_BOUNDS)



class ACNet(ACNetBase):
    def __init__(self, scope, globalAC):
        self.session = globalAC.session
        self.scope = scope
        with tf.variable_scope(scope):
            self.s = tf.placeholder(tf.float32, [None, STATE_SIZE], 'S')
            self.a_his = tf.placeholder(tf.float32, [None, ACTION_SIZE], 'A')
            self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')

            self._build_net()

            td = tf.subtract(self.v_target, self.v, name='TD_error')
            with tf.name_scope('c_loss'):
                self.critic_loss = tf.reduce_mean(tf.square(td))

            with tf.name_scope('wrap_a_out'):
                self.test = self.sigma[0]
                self.mu, self.sigma = self.mu * ACTION_BOUNDS[1], self.sigma + 1e-5

            normal_dist = tf.contrib.distributions.Normal(self.mu, self.sigma) # for continuous action space

            with tf.name_scope('a_loss'):
                log_prob = normal_dist.log_prob(self.a_his)
                exp_v = log_prob * td
                entropy = normal_dist.entropy()
                self.exp_v = ENTROPY_BETA * entropy + exp_v
                self.actor_loss = tf.reduce_mean(-self.exp_v)

            with tf.name_scope('choose_a'):
                self.A = tf.clip_by_value(tf.squeeze(normal_dist.sample(1), axis=0), *ACTION_BOUNDS)

            with tf.name_scope('local_grad'):
                self.actor_params = tl.layers.get_variables_with_name(scope + '/actor', True, False)
                self.critic_params = tl.layers.get_variables_with_name(scope + '/critic', True, False)
                self.actor_grads = tf.gradients(self.actor_loss, self.actor_params)
                self.critic_grads = tf.gradients(self.critic_loss, self.critic_params)

        with tf.name_scope('sync'):
            with tf.name_scope('pull'):
                self.pull_actor_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.actor_params, globalAC.actor_params)]
                self.pull_critic_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.critic_params, globalAC.critic_params)]
            with tf.name_scope('push'):
                self.update_actor_op = ACTOR_OPTIMIZER.apply_gradients(zip(self.actor_grads, globalAC.actor_params))
                self.update_critic_op = CRITIC_OPTIMIZER.apply_gradients(zip(self.critic_grads, globalAC.critic_params))



    def update_global(self, feed_dict):
        _, _, t = self.session.run([self.update_actor_op, self.update_critic_op, self.test], feed_dict)
        return t

    def pull_global(self):
        self.session.run([self.pull_actor_params_op, self.pull_critic_params_op])


class Worker(object):
    def __init__(self, name, globalAC, coordinator):
        self.env = env_make()
        self.name = name
        self.AC = ACNet(name, globalAC)
        self.coordinator = coordinator
        self.session = globalAC.session

    def work(self):
        global GLOBAL_RUNNING_R, GLOBAL_EPOCH
        lep = 0
        stateHist, actionHist, rewardHist = [], [], []
        while not self.coordinator.should_stop() and GLOBAL_EPOCH < MAX_GLOBAL_EP:
            s = self.env.reset()
            ep_r = 0
            rand_bias = 0
            steps = 0
            while True:
                if self.name == 'W0' and steps % 20 == 0:
                    self.env.render()
                a = self.AC.choose_action(s)
                s_, r, done, info = self.env.step(a)
                position = self.env.unwrapped.hull.position[0]

                if r == -100:
                    r = -2

                ep_r += r
                stateHist.append(s)
                actionHist.append(a)
                rewardHist.append(r)

                rand_bias = rand_bias * 0.95 + self.AC.get_variance(s).mean() * 0.05

                s = s_
                steps += 1

                if done:
                    running_add = 0
                    discountedRewards = []
                    for r in rewardHist[::-1]:
                        running_add = r + GAMMA * running_add
                        discountedRewards.append(running_add)
                    discountedRewards.reverse()

                    stateHist, actionHist, discountedRewards = np.vstack(stateHist), np.vstack(actionHist), np.vstack(discountedRewards)
                    feed_dict = {
                        self.AC.s: stateHist,
                        self.AC.a_his: actionHist,
                        self.AC.v_target: discountedRewards,
                    }

                    test = self.AC.update_global(feed_dict)
                    stateHist, actionHist, rewardHist = [], [], []

                    self.AC.pull_global()

                    GLOBAL_RUNNING_R.append(0.95 * GLOBAL_RUNNING_R[-1] + 0.05 * ep_r)
                    print(self.name, "episode: %4d" % GLOBAL_EPOCH, "/ random bias rate: %0.2f " % rand_bias,
                          "steps: %4d " % steps, " position: %2.3f " % position, "/ score: %3.3f " % ep_r)
                    GLOBAL_EPOCH += 1

                    if lep + 0 < GLOBAL_EPOCH and self.name == 'W1':

                        pass

                    break

session = tf.Session()

with tf.device("/cpu:0"):
    actorCriticNet = ACNetGlobal(session)
    coordinator = tf.train.Coordinator()

with tf.device("/cpu:0"):
    workers = [Worker("W%i" % i, actorCriticNet, coordinator) for i in range(N_WORKERS)]

tl.layers.initialize_global_variables(session)

worker_threads = []
for worker in workers:
    t = threading.Thread(target=worker.work)
    t.start()
    worker_threads.append(t)


while True:
    try:
        time.sleep(2)
        with open("./save_graph/scores.txt", 'w') as storage:
            storage.write(" ".join(map(str, GLOBAL_RUNNING_R)))
    except:
        pass

coordinator.join(worker_threads)

