from collections import deque
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, LSTM, Dot
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import random


class Agent():
    def __init__(self, memory=2048, pos_memory=2048, dr=0.5, lr=0.1,
                 er = 1, ed = 0.99, emin = 0.1):
        self.memory = deque(maxlen = memory)
        self.pos_memory = deque(maxlen = memory)
        self.priority = 0.25
        self.dr = dr
        self.er = er
        self.ed = ed
        self.emin = emin
        self.lr = lr
        self.vocab_size = 2048
        self.qdict = dict()
        self.history = list()
        self.model = self.create_model()

    def create_model(self,lstm_dim=32,dense_dim=16):
        '''
        Builds and compiles a State-Action Neural Network using a shared
        embedding and LSTM
        '''
        # Define input layers with no specific shape for state and action
        input_state = Input(batch_shape=(None,None), name='state')
        input_action = Input(batch_shape=(None,None), name='action')
        
        # Define embedding layer to be shared by the state and action
        embedding = Embedding(self.vocab_size+1, 16, input_length = None,
                              mask_zero = True, name = 'Embedding')
        embed_state = embedding(input_state)
        embed_action = embedding(input_action)
        
        # Generate LSTM layer and add it to the network
        lstm = LSTM(lstm_dim, name = 'LSTM')
        lstm_state = lstm(embed_state)
        lstm_action = lstm(embed_action)

        # Generate two individual dense layers for processing state and action
        # independently
        dense_state = Dense(dense_dim, activation='relu',
                            name='Dense_S')(lstm_state)
        dense_action = Dense(dense_dim, activation='relu', 
                             name='Dense_A')(lstm_action)

        # Generate the dot product layer and its two input layers, the same size
        # of the last layer
        input_dot_state = Input(shape = (dense_dim,))
        input_dot_action = Input(shape = (dense_dim,))
        dot_layer = Dot(axes=-1, normalize=False, name='Dot_Product')
        dot_product = dot_layer([input_dot_state, input_dot_action])

        #Build the individual models and compile
        dp_model = Model(inputs=[input_dot_state,input_dot_action],
                         outputs=dot_product)
        self.dp_model = dp_model
        model_state = Model(inputs=input_state, outputs = dense_state,
                            name = 'state')
        self.state_model = model_state
        model_action = Model(inputs=input_action, outputs=dense_action,
                             name='action')
        self.action_model = model_action
        model = Model(inputs = [model_state.input, model_action.input],
                      outputs=dp_model([model_state.output,model_action.output]))
        model.compile(optimizer='adam', loss='mse')
        
        return model
    
    

    def remember(self,state,state_text,action,reward,next_state,next_state_text,
                 action_dict,finished):
        '''
        Stores the run in memory, including the state, action taken, all
        possible actions, and state arrived to
        '''
        self.memory\
            .append((state, state_text, action, reward, next_state, 
                     next_state_text, action_dict, finished))
        
        # Store successful actions-state pairs in positive memory
        if reward>0:
            self.pos_memory\
                .append((state, state_text, action, reward, next_state, 
                         next_state_text, action_dict, finished))

    def explore(self):
        '''
        Determines whether the agent will explore randomly or predict the 
        next action
        '''
        if np.random.rand()<=self.er:
            return True
        else:
            return False
    
    def calculate_max_q(self, state_text, state_input, action_dict):
        '''
        Estimates the maximum q value using the provided model.
        '''
        # Check if current state has its own q value, else assign it 0
        if state_text in self.qdict:
            q_tgt = self.qdict[state_text]
        else:
            q_tgt=0
            self.qdict[state_text] = 0
        
        # Initialize the variable to estimate in
        q_max = -np.math.inf
        
        # Shuffle all possible actions into a random order
        action_tups = list(action_dict.items())
        index_list = list(range(len(action_tups)))
        random.shuffle(index_list)
        i=0
        # Calculate the best action for the current state
        print("cal-q-lating")
        while (q_max < q_tgt and i< len(action_tups)):
            idx = index_list[i]
            action, data = action_tups[idx]
            action_vector = data[1]
            action_dense = self.action_model.predict([action_vector])[0]
            action_input = action_dense.reshape((1, len(action_dense)))
            with tf.device('/gpu:0'):
                q = self.model\
                        .predict([state_input, action_input], 
                                 batch_size=1)[0][0]
            
            # If beating q_max, set as new benchmark to beat
            if q > q_max:
                q_max = q
                best_action = action
                print(action, q)
            i += 1
        
        # Store q value on the dictionary
        self.qdict[state_text] = q_max
        return best_action, q_max
    
    def predict_actions(self, state_text, state, action_dict):
        '''
        Preprocesses state and runs search for best q
        '''
        state_dense = self.state_model.predict([state])[0]
        state_input = state_dense.reshape((1, len(state_dense)))
        best_action, _ = self.calculate_max_q(state_text, state_input, 
                                                  action_dict)
        return best_action


    def replay(self, batch_size):
        '''
        Replays the last played batch to train the model using rewards
        '''
        
        # Extract states and actions from memory
        states = [None]*batch_size
        actions = [None]*batch_size
        # Initialize zero reward targets
        targets = np.zeros((batch_size, 1))
        # Define a DataFrame with the next state input and future q value
        next_state_df = pd.DataFrame(columns=['next_state', 
                                              'next_state_in', 'future_q'])
        # Select a subset of positive runs and sample from the positive
        # runs and all runs
        batch_positive_size = int(batch_size*self.priority)
        batch_all_size = batch_size - batch_positive_size
        batch_pos = np.random.choice(len(self.pos_memory), 
                                     batch_positive_size)
        batch_all = np.random.choice(len(self.memory), batch_all_size)
        b_p = 0
        b_r = 0 
        for i in range(0, batch_size):  
            if i < batch_positive_size:
                # Extract positive actions
                state, _ ,action, reward, next_state, next_state_text,\
                action_dict, finished = self.pos_memory[batch_pos[b_p]]
                b_p += 1

            else:
                # Extract other actions
                state, _, action, reward, next_state, next_state_text,\
                action_dict, finished = self.memory[batch_all[b_r]]
                b_r += 1
            #Reward becomes new target
            target = reward
            if not finished:
                # extract the future q if not in next state df
                try:
                    next_state_in = next_state_df[next_state_df['next_state'] == next_state]['next_state_in'] 
                    future_q = next_state_df[next_state_df['next_state'] == next_state]['future_q']
                except:
                    with tf.device('/gpu:0'):
                        next_state_dense = self.state_model\
                                                .predict([next_state])[0]
                        next_state_in = next_state_dense\
                                            .reshape((1, len(next_state_dense)))
                        _, future_q = self.calculate_max_q(next_state_text, 
                                                           next_state_in, 
                                                           action_dict)
                        row = len(next_state_df)
                        next_state_df.loc[row, 'next_state'] = next_state
                        next_state_df.loc[row, 'next_state_in'] = next_state_in
                        next_state_df.loc[row, 'future_q'] = future_q

                ## Multiply future q and discount rate
                target = reward + self.dr * future_q
                
                ## Store everything
                states[i] = state[0]
                actions[i] = action[0]
                targets[i] = target
                
        states = pad_sequences(states)
        actions = pad_sequences(actions)
        # Train model on the targets
        history = self.model.fit(x=[states, actions], y=targets, 
                                 batch_size=batch_size, epochs=1, verbose=1)
        # Save model history
        self.history.append(history)
        if self.er > self.emin:
            self.er *= self.ed   

    def save_model_weights(self):
        '''
        Saves model weights into h5 file and history as pickle
        '''
        self.model.save('model.h5')
        self.model.save_weights('model_weights.h5')
        try:
            with open('model_history.pickle', 'wb') as fp:
                pickle.dump(self.model_histories, fp, 
                            protocol=pickle.HIGHEST_PROTOCOL)
        except:
            pass