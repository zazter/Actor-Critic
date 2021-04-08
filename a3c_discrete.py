import gym
import time
import datetime
from collections import deque
import random
import threading
import numpy as np
import tensorflow as tf
import multiprocessing
import cv2

from tensorflow.keras.layers import Dense, Input, Flatten
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam, RMSprop
from keras.models import Model


# create actor and critic model with common layers
class Neural_Network():
    def __init__(self, actions, input_shape, learning_rate):
            self.actions =       actions
            self.input_shape =   input_shape
            self.learning_rate = learning_rate
            
    def create_a3c_model(self):        
        in_layer = Input(shape=self.input_shape)
        conv1 = Conv2D(32, 8,
                         strides = (4, 4), 
                         kernel_initializer="he_uniform",
                         activation='relu', 
                         padding = "valid")(in_layer)
        conv2 = Conv2D(64,  4,
                         strides = (2, 2), 
                         kernel_initializer="he_uniform",
                         padding = "valid",
                         activation='relu')(conv1)
        conv3 = Conv2D(64,  4,
                         strides = (2, 2), 
                         kernel_initializer="he_uniform",
                         padding = "valid",
                         activation='relu')(conv2)
        flat = Flatten()(conv3)
        dense = Dense(512, 
                        kernel_initializer="he_uniform",
                        activation='relu')(flat)
        
        action = Dense(self.actions,
                        activation="softmax")(dense)
        value = Dense(1, activation='linear')(dense)
        
        actor = Model(inputs= in_layer, outputs= action)
        
            
        
        actor.compile(optimizer=Adam(lr=self.learning_rate), 
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        
        critic = Model(inputs= in_layer, outputs= value)
        critic.compile(optimizer=Adam(lr=self.learning_rate), 
                      loss='mse',
                      metrics=['accuracy'])
        return actor, critic


# global agent with global actor/critic that starts workers
class Global_Agent:
    def __init__(self, input_shape):
        self.env_name =               "BreakoutDeterministic-v4"
        self.lr =                     0.000025
        self.gamma =                  0.99
        self.env =                    gym.make(self.env_name)
        self.input_shape =            input_shape
        self.actions =                self.env.action_space.n
        self.actor, self.critic  =     Neural_Network(self.actions, 
                                               self.input_shape, 
                                               self.lr).create_a3c_model()

    # start workers
    def train(self):

        agents = [Agent(self.input_shape, 
                        self.actions, 
                        [self.actor, self.critic],
                        self.gamma,
                        self.env_name, i) for i in range(multiprocessing.cpu_count())]
        for i, agent in enumerate(agents):
            print("Staring worker {}".format(i))
            agent.start()
            time.sleep(1)
            
        for agent in agents:
            agent.join()
            


class Agent(threading.Thread):
    def __init__(self, input_shape, actions, networks, gamma, env_name, i):
        threading.Thread.__init__(self)
        self.number = i
        self.input_shape = input_shape
        self.lock = threading.Lock()
        self.lr = 0.000025
        self.actions = actions
        self.actor, self.critic = networks
        self.stack = deque(maxlen=4)
        self.l_actor,  self.l_critic = Neural_Network(self.actions, 
                                               self.input_shape, 
                                               self.lr).create_a3c_model()
        self.gamma = gamma
        self.env_name = env_name
        self.states, self.action_history, self.rewards = [], [], []
        
        #tensorboard
        self.log_dir_a = "summaries/actor/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_dir_c = "summaries/critic/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.callback_a = tf.keras.callbacks.TensorBoard(log_dir=self.log_dir_a, histogram_freq=1)
        self.callback_c = tf.keras.callbacks.TensorBoard(log_dir=self.log_dir_c, histogram_freq=1)
        
    def run(self):
        env = gym.make(self.env_name)
        self.actions = env.action_space.n
        episodes = 10000
        all_rewards = []
        
        for episode in range(episodes):
            state = env.reset()
            reward_total, step = 0, 0
            life_lost = False
            lives = 5
            # no_op
            for _ in range(random.randint(1, 5)):
                state, _, done, _ = env.step(1)

            states = self.add_to_stack(state, True)
            
            while not done:
                step += 1
                #env.render()
                action = self.get_action(states)
                
                # fire to start new life again
                if life_lost:
                    action = 1
                    life_lost = False
        
                next_state, reward, done, info = env.step(action)
                
                if info['ale.lives'] < lives:
                    life_lost = True
                    lives = info['ale.lives']
                
                reward_total += reward
                if life_lost:
                    states = self.add_to_stack(next_state, True)
                else:
                    states = self.add_to_stack(next_state)
                
                self.memory(states, action, reward)
                
                if step % 5 == 0 or done:
                    with self.lock:
                        self.train_model(done)
                        self.update_model()
                        
                
                
            all_rewards.append(reward_total)
            print("Agent {}: Ep. reward: {}    Avg Reward: {:.2f}".format(self.number,
                                                                          reward_total,
                                                                          np.mean(all_rewards[-10:])))
            

        
    def train_model(self, done):
        
        discounted_rewards = self.discounted_rewards(self.rewards, done)
        
        states = np.stack(self.states)
        states = np.float32(states / 255.)
        
        values = self.l_critic.predict(states)
        values = np.reshape(values, len(values))
        
        advantages = discounted_rewards - values
        #print(self.action_history)
        #actions = np.vstack(self.action_history)
        actions = np.stack(self.action_history)
        self.actor.fit(states, actions, sample_weight=advantages, epochs=1, verbose=0, callbacks=[self.callback_a])
        self.critic.fit(states, discounted_rewards, epochs=1, verbose=0, callbacks=[self.callback_c])
        
        self.states, self.action_history, self.rewards = [],[],[]
        
        
    def get_action(self, history):
        history = np.float32(history / 255.)
        history  = history.reshape((1, *history.shape))
        policy = self.l_actor.predict(history)[0]
        action_index = np.random.choice(self.actions, 1, p=policy)[0]
        return action_index

    def memory(self, history, action, reward):
        self.states.append(history)
        act = np.zeros(self.actions)
        act[action] = 1
        self.action_history.append(act)
        self.rewards.append(reward)    
     
    def add_to_stack(self, frame, reset = False):
        frame = self.preprocess_image(frame)
        
        if reset:
            for _ in range(4):
                self.stack.append(frame)
        else:
            self.stack.append(frame)
        
        states = np.stack(self.stack, axis=2)
        return states    
        
    def discounted_rewards(self, rewards, done):
        discounted_rewards = np.zeros_like(rewards)
        state = np.float32(self.states[-1] / 255.)
        state = state.reshape((1, *state.shape))
        running_add = 0
        if not done:
            running_add = self.critic.predict(state)[0]
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.gamma + rewards[t]
            discounted_rewards[t] = running_add
        if not done:
            discounted_rewards -= np.mean(discounted_rewards) # normalizing the result
            discounted_rewards /= np.std(discounted_rewards) # divide by standard deviation 
            discounted_rewards = np.nan_to_num(discounted_rewards)
        
        return discounted_rewards
    
    def preprocess_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = image[34:34+160, :160]  # crop image
        image = cv2.resize(image, (84, 84), interpolation=cv2.INTER_NEAREST)
        return image
    
    def update_model(self):
        self.l_critic.set_weights(self.critic.get_weights())
        self.l_actor.set_weights(self.actor.get_weights()) 
        
agent = Global_Agent((84, 84, 4))
agent.train()
