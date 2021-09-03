from subprocess import Popen, PIPE, STDOUT
from queue import Queue, Empty
from threading import Thread
import numpy as np
import pandas as pd
import re
import spacy
import pickle
from spacy.matcher import Matcher
from spacy.attrs import POS
import itertools
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
from qagent import Agent
from progressbar import ProgressBar
from time import sleep
import traceback
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec

class Game():
    def __init__(self, proc: str = 'dfrotz.exe', game: str = 'zork1.z5',
                 rand_weight = 6, rand_basic=0.4, rand_low=0.1):
        '''
        Allows for opening any text adventure using dfrotz, defaulting to zork1
        '''
        self.agent = Agent()
        self.batch_size = 10
        # Emulator information
        self.process = proc
        self.game = game

        # Process to capture, queue to capture in, thread to run queue
        self.p = None
        self.q = None
        self.t = None

        # Score, moves and other reward info
        self.score = 0
        self.game_score = 0
        self.moves = 0
        self.turn_reward = -1
        self.new_area_reward = 2
        self.move_reward = 0.5
        self.inventory_reward = 3
        self.inventory_not_new_reward_value =0.5

        # Using spacy to analyze POS
        self.nlp = spacy.load('en_core_web_lg')

        # list of game actions
        self.movement = [
           'go north', 'go south', 'go west', 'go east', 'go northeast',
           'go northwest', 'go southeast', 'go southwest', 'go down', 'go up'
        ]
        self.directions = [x.replace("go ","") for x in self.movement]
        self.noun_actions = [
            'open', 'get', 'eat', 'ask', 'make', 'wear', 'move', 'kick', 
            'find', 'play', 'feel', 'read', 'fill', 'pick', 'pour', 'pull',
            'leave', 'break', 'enter', 'shake', 'banish', 'enchant'
        ]
       
        self.two_noun_actions = [
            'pour @1 on @2', 'hide @1 in @2', 'pour @1 in @2', 'move @1 in @2',
            'hide @1 on @2', 'flip @1 for @2', 'fix @1 with @2', 
            'spray @1 on @2', 'dig @1 with @2', 'cut @1 with @2',
            'pick @1 with @2', 'pour @1 from @2', 'fill @1 with @2',
            'burn @1 with @2', 'flip @1 with @2',
            'read @1 with @2', 'hide @1 under @2', 'carry @1 from @2',
            'inflate @1 with @2', 'unlock @1 with @2', 'give @1 to @2', 
            'carry @1 to @2', 'spray @1 with @2'            
        ]

        # Generate action space and add all movement actions
        self.action_space = set()
        for action in self.movement:
            self.action_space.add(action)
        
        # Assigns the weights for similarity and basic actions
        self.rand_weight = rand_weight
        self.rand_basic = rand_basic
        self.rand_low = rand_low

        # Tries to load a tokenizer or generates it
        try:
            with open('tokenizer.pickle','rb') as tk:
                self.tokenizer = pickle.load(tk)
        except: 
            self.tokenizer = Tokenizer(num_words=2048)

        self.state_data = pd.DataFrame(columns = ['state','statevec',
                                                  'actiondata','nouns'])
        self.unique_state = set()
        self.unique_inventory = set()

        # Creates a list of valid or invalid nouns
        self.invalid_nouns = []
        self.valid_nouns = []

        # Loads invalid nouns and valid nouns from previous runs
        self.load_invalid_nouns()
        self.load_valid_nouns()

        # Generates result storage
        self.end_game_scores = pd.DataFrame(columns=['gamenum','score'])
        self.stories = []


    def add_out_to_queue(self, out, queue):
        '''
        Reads an output as a textfile and adds all the lines into the queue
        '''
        for line in iter(out.readline, b''):
            queue.put(line)
        out.close()
    
    def get_score_moves(self,env):
        '''
        Takes current environment text and extracts the score and moves
        '''
        try:
            # Find score in text
            score = int(env[env.index('Score: ') + len('Score: '):][0:3]\
                             .strip())
            # Find moves in text
            moves = int(env[env.index('Moves: ') + len('Moves: '):][0:3]\
                              .strip())
        except: # Generate default values
            score = 0
            moves = 0
        return(score, moves)

    def read_line(self):
        '''
        Reads the queue and joins everything in a space separated string
        '''
        cont = True
        env = ""
        while cont:
            try:  line = self.q.get_nowait()
            except Empty:
                cont = False
            else: 
                env = env + line.decode("utf-8").replace('\n', " ")\
                                    .replace('\r', " ")
        # Needs to be scaled for other text adventures,
        # currently using zork1 rev 88 sn 840726
        if ('840726' in env):
            # For the first environment, jumps to after game credits to take
            # first environment
            env = env[env.index('840726') + len('840726'):]
        try:
            score, moves = self.get_score_moves(env)
            env = env[env.index('Moves: ')+len('Moves:')+5:-1].strip()
        except: # Records invalid moves
            pass
        # Sleep to avoid messing up
        sleep(0.01)
        return(env, score, moves)

    def load_state(self):
        '''
        Looks for a pickle file with the state data, else sticks with the
        default empty dataframe defined in __init__
        '''
        try: 
            self.state_data = pd.read_pickle('state_data.pickle')
        except:
            pass
    
    def start_game(self):
        '''
        Initializes the game
        '''
        self.load_state()
        # Generates a pandas dataframe thet will store all the information
        # in the story
        self.story = pd.DataFrame(columns=['surroundings', 'inventory', 
                                           'action', 'response', 'reward', 
                                           'rewardtype', 'score', 'moves', 
                                           'totmoves', 'game_score'])
        # Generates a dataframe with the number of times playing the game and
        # the score obtained
        self.scores = pd.DataFrame(columns=["gamenum","score"])
        self.score = 0
        self.moves = 0
        self.unique_state = set()
        self.game_score = 0
        # Opens the game using dfrotz, an emulator for text adventures
        # Allows the program to capture both inputs and outputs
        self.p = Popen([self.process, self.game], stdin=PIPE, 
                       stdout=PIPE, stderr=PIPE, shell=True)
        self.q = Queue()
        # Assign a thread to run a particular task: adding the game's output
        # to the queue
        self.t = Thread(target=self.add_out_to_queue, 
                        args=(self.p.stdout, self.q))
        self.t.daemon = True
        self.t.start()
        sleep(0.1)

    def do_action(self, cmd:str):
        '''
        Types a command into the terminal, flushes the input and waits
        for an output
        '''
        self.p.stdin.write(bytes(cmd+ "\n", 'ascii'))
        self.p.stdin.flush()
        sleep(0.01)

    def preprocess(self, line:str):
        '''
        Generates basic preprocessing of any string
        '''
        # Strip output from terminal, rework special characters
        text = line.strip().replace('\\n','').replace('‘', '\'')\
                   .replace('’', '\'').replace('”', '"').replace('“', '"')\
                   .lower()
        
        # Remove text that appears after inventory that weirds out my dude
        flavor_text = 'you hear in the distance the chirping of a song bird'
        if (flavor_text in text):
            text = text.replace(flavor_text, '')
        # Remove all characters that are not alphanumeric, dashes,
        # spaces and quotes
        regex = re.compile('[^ \-\sA-Za-z0-9"\']+')
        text = regex.sub('',text)
        # Convert every multiple whitespace into a single one
        text = re.sub('\s{2,}',' ',text)
        return text
    
    def get_state(self):
        # The state of the player character can be defined by its inventory and
        # surroundings, while the player state includes score and moves
        self.do_action('look')
        surroundings, score, moves = self.read_line()
        surroundings = self.preprocess(surroundings)
        self.do_action('inventory')
        inventory, score, moves = self.read_line()
        inventory = self.preprocess(inventory)

        pc_state = surroundings + ' ' + inventory
        return pc_state, surroundings, inventory, score, moves

    def get_nouns(self, state):
        '''
        Uses Spacy matcher object to extract the nouns from the state   cc
        '''
        matcher = Matcher(self.nlp.vocab)
        pattern = {'POS': 'NOUN'}
        matcher.add('Noun Matcher', [[pattern]])
        doc = self.nlp(state)
        matches = matcher(doc)
        unique_nouns = set()
        for id, start, end in matches:
            noun = doc[start:end].text
            if noun not in self.directions and noun not in self.invalid_nouns:
                unique_nouns.add(noun)
        return unique_nouns
    
    def vectorize_text(self, text, tokenizer):
        '''
        Tokenizes words, converts them into padded sequences for preprocessing
        '''
        words = word_tokenize(text)
        tokenizer.fit_on_texts(words)
        seq = tokenizer.texts_to_sequences(words)
        sent = [x[0] for x in seq]
        padded = pad_sequences([sent], padding = 'post')
        return padded

    def gen_all_pos_actions(self, nouns):
        '''
        Takes all nouns in state to generate the list of possible actions
        '''
        pos_actions = []
        for x in self.movement:
            pos_actions.append(x)

        for noun in nouns:
            for action in self.noun_actions:
                a = action + ' ' + noun
                pos_actions.append(a)

            # Replaces wildcards for actual nouns
            two_noun_perms = list(itertools.permutations(nouns,2))
            for action in self.two_noun_actions:
                for perm in two_noun_perms:
                    a = action.replace('@1',perm[0])
                    a = a.replace('@2', perm[1])
                    pos_actions.append(a)

        return pos_actions

    def gen_action_space(self, action_space:set, actions):
        '''
        Generates a set of unique actions weights them using SpaCuy similarity
        '''
        sims = []

        for action in actions:
            action_space.add(action)
        for action in action_space:
            words = action.split()
            verb = self.nlp(words[0])
            if action in self.movement:
                sims.append(self.rand_basic)
            elif len(words)<3:
                # Looks for non-movement, two word actions.
                noun = self.nlp(words[1])
                try:
                    # Generates probability using similarity
                    similarity = verb.similarity(noun)**self.rand_weight
                    if similarity < 0:
                        similarity = self.rand_basic**self.rand_weight
                    sims.append(similarity)
                except:
                    sims.append(self.rand_low**self.rand_weight)
            else:
                try:
                    noun = self.nlp(words[1])
                    prep = self.nlp(words[2])
                    noun2 = self.nlp(words[3])
                    similarity1 = verb.similarity(noun)
                    similarity2 = prep.similarity(noun2)
                    similarity = ((similarity1+similarity2)/2)**self.rand_weight
                    if similarity < 0:
                        similarity = 0.05
                    sims.append(similarity)
                except:
                    sims.append(self.rand_low**self.rand_weight)
        return action_space, sims
    
    def select_action(self, action_space, sims):
        idx = np.random.choice(len(action_space),p=sims)
        return action_space[idx]

    def do_action_and_read(self, action):
        self.do_action(action)
        response, score, moves = self.read_line()
        response = self.preprocess(response)
        return response, score, moves

    def detect_invalid_nouns(self, action_response):
        word = ''
        ## detect and remove invalid nouns from future turns
        if('know the word' in action_response):
            startIndex = action_response.find('\"')
            endIndex = action_response.find('\"', startIndex + 1)
            word = action_response[startIndex+1:endIndex]
        return word

    def save_invalid_nouns(self):
        ## save invalid nouns to pickled list
        try:
            with open('invalid_nouns.txt', 'wb') as fp:
                pickle.dump(self.invalid_nouns, fp)
        except:
            pass

    def load_invalid_nouns(self):
        ## load previously found invalid nouns from pickled list
        try:
            with open ('invalid_nouns.txt', 'rb') as fp:
                n = pickle.load(fp)
                self.invalid_nouns.extend(n)
        except:
            pass
    
    def test_nouns(self, nouns):
        for noun in nouns:
            if noun in self.invalid_nouns or noun in self.valid_nouns:
                pass
            else:
                action = 'feel ' + noun
                response, current_score, moves = self.do_action_and_read(action)
                if('know the word' in response):
                    self.invalid_nouns.append(noun)
                else:
                    self.valid_nouns.append(noun)  
    def save_valid_nouns(self):
        ## save invalid nouns to pickled list
        try:
            with open('valid_nouns.txt', 'wb') as fp:
                pickle.dump(self.valid_nouns, fp)
        except:
            pass
    
    def load_valid_nouns(self):
        ## load previously found invalid nouns from pickled list
        try:
            with open ('invalid_nouns.txt', 'rb') as fp:
                n = pickle.load(fp)
                self.valid_nouns.extend(n)
        except:
            pass
    def detect_invalid_action(self, state, action, reward, action_dict, invalid_noun):
        ## remember already tried actions that don't change current game state
        invalid_action = ''
        if (reward==-1):
            invalid_action = action
        ## check if we have an invalid noun or action and remove them from the action dictionary
        if invalid_noun:
            for act, data in action_dict.items():
                if invalid_noun in act:
                    del action_dict[invalid_noun]
            self.state_data.loc[self.state_data['state'] == state, 'actiondata'] = [action_dict]
        if invalid_action and invalid_action in action_dict:
            del action_dict[invalid_action]
            ## update state data 
            self.state_data.loc[self.state_data['state'] == state, 'actiondata'] = [action_dict]
        return action_dict

    def reward(self, inventory, old_inventory, moves, old_state,
               new_state, score):
        reward = 0
        reward_msg = ''
        # game score for reward
        if(moves != 0):
            reward += score
            if (score > 0):
                print('Scored ' + str(score) + 'points')
                reward_msg += ' game score: ' + str(reward) + ' '
        #Give a negative reward per turn
        reward += self.turn_reward

        #add inventory reward
        if(moves != 0):
            if inventory.strip().lower() not in old_inventory.strip().lower():
                #avoid pickup + drop
                if(old_inventory + ' - ' + inventory) not in self.unique_inventory:
                    self.unique_inventory.add(old_inventory + ' - ' + inventory)
                    reward = reward + self.inventory_reward
                    print('inventory changed - new')
                    reward_msg += ' inventory score (' + old_inventory + " --- " + inventory + ')'    
                else:
                    reward = reward + self.inventory_not_new_reward_value                

        #add reward for discovery
        if new_state.strip() not in self.unique_state:
            reward += self.new_area_reward 
            self.unique_state.add(new_state.strip())
            reward_msg += ' new area score ---' + new_state.strip()
        
        #add reward for not getting stuck
        if old_state not in new_state:
            reward += self.move_reward
            reward_msg += ' - moved - '
        
        print('Total reward:' + str(reward))
        return reward, reward_msg
    
    def save_tokenizer(self):
        try:
            with open('tokenizer.pickle', 'wb') as fp:
                pickle.dump(self.tokenizer, fp, protocol=pickle.HIGHEST_PROTOCOL)
        except:
            pass
    
    def end_game(self):
        self.save_invalid_nouns()
        self.save_tokenizer()
        self.save_valid_nouns()
        self.agent.save_model_weights()
        self.p.terminate()
        self.p.kill()
    
    def restart(self):
        self.save_invalid_nouns()
        self.save_valid_nouns()
        self.do_action('restart')
        self.read_line()
        self.do_action('y')
        self.read_line()
        self.score = 0
        self.unique_state = set()
        self.unique_inventory = set()
        self.score = 0

    def save_model_weights(self):
        self.agent.model.save_weights('dqn_model_weights.h5')

    def get_data(self,state):
        if(state in list(self.state_data['state'])):
            statevec = list(self.state_data[self.state_data['state'] == state]['statevec'])[0][0]
            try:
                
                nouns = list(self.state_data[self.state_data['state'] == state]['nouns'])[0][0]
            except:
                None
            try:
                actionsvec = []
                actions = []
                probs = []
                action_dict = list(self.state_data[self.state_data['state'] == state]['actiondata'])[0]
                for act, data in action_dict.items():
                    actions.append(act)
                    probs.append(data[0])
                    actionsvec.append(data[1])
                probs = np.array(probs)
            except:
                actionsvec = []
                actions = []
                probs = []
                action_dict = list(self.state_data[self.state_data['state'] == state]['actiondata'])[0][0]
                for act, data in action_dict.items():
                    actions.append(act)
                    probs.append(data[0])
                    actionsvec.append(data[1])
                probs = np.array(probs)
        else: 
            statevec = self.vectorize_text(state,self.tokenizer)
            
            nouns = self.get_nouns(state)
            self.test_nouns(nouns)
            for noun in nouns:
                if noun in self.invalid_nouns:
                    nouns.remove(noun)
            curr_action_space= self.gen_all_pos_actions(nouns)
            action_space = set()
            action_space, probs = self.gen_action_space(action_space, curr_action_space)
    
            actions = []
            for a in action_space:
                actions.append(a)
            probs = np.array(probs)
            actionsvec = []
            for a in actions:
                actionsvec.append(self.vectorize_text(a,self.tokenizer))
            ## create action dictionary
            action_dict = dict()
            for idx, act in enumerate(actions):
                action_dict[act] = (probs[idx], actionsvec[idx])
            ## store state data 
            row = len(self.state_data)
            self.state_data.loc[row, 'state'] = state
            self.state_data.loc[row, 'statevec'] = [statevec]
            self.state_data.loc[row, 'actiondata'] = [action_dict]
            self.state_data.loc[row, 'nouns'] = [nouns]
        return probs, actions, statevec, actionsvec, action_dict

    def run(self, agent, num_games=20, 
            num_rounds=512, batch_size=256, training=True):
        ## set global batch size
        self.batch_size = batch_size
        
        ## initialize progress bar
        pbar = ProgressBar(maxval=num_rounds*num_games)
        pbar.start()
        
        ## initialize game
        self.start_game()

        ## number of games loop
        for game_number in range(0, num_games):
            print('___starting new game___')
            new_state = ''
            inventory = ''
            try:
                for i in range(0, num_rounds):
                    ## get initial state if first round, else grab new state from previous round
                    if (i==0):
                        state, old_surroundings, old_inventory,\
                        _, _ = self.get_state()
                        self.unique_state.add(state)
                        print('generated initial state')
                    else:
                        state  = new_state
                        old_inventory = inventory
                        
                    ## sometimes reading of lines gets backed up, if this happens reset and re-check state
                    invalid_line = True
                    
                    while invalid_line:
                        invalid_line = False
                        if len(state)>1000 or len(state)<5 or 'score' in state:
                            print('encountered line read bug')
                            state, old_surroundings, old_inventory,\
                            _, _ = self.get_state()
                            invalid_line = True
                    #print(state)

                        
                    ## get data for current state
                    probs, actions, statevec,\
                    actionsvec, action_dict = self.get_data(state)
                    ## decide which type of action to perform
                    if (agent.explore()): ## choose random action
                        print('random choice:')
                        probs_norm = probs/(probs.sum())
                        action = self.select_action(actions, probs_norm) 
     
                    else: ## choose predicted max Q value action
                        print('predicted choice:')
                        action = agent.predict_actions(state, statevec, action_dict)

                    print('-- ' + action + ' --')
    
                    ## perform selected action
                    response, current_score,\
                        moves = self.do_action_and_read(action)
                    invalid_noun = self.detect_invalid_nouns(response)
                    
                    ## vectorize selected action
                    actionvec = self.vectorize_text(action,self.tokenizer)
    
                    ## check new state after performing action
                    new_state, surroundings, inventory,\
                    current_score, moves = self.get_state()
                    new_state = self.preprocess(new_state)
                    new_statevec = self.vectorize_text(new_state, self.tokenizer)
                    
                    ## get reward
                    round_score = current_score - self.game_score
                    self.game_score = current_score
                    reward, reward_msg = self.reward(inventory, old_inventory, i, state, new_state, round_score)
    
                    ## update story dataframe
                    self.score += reward
                    total_round_number = i + game_number*num_rounds
                    self.story.loc[total_round_number] = [state, old_inventory, action, response, reward, 
                                  reward_msg, self.score, str(i), total_round_number, self.game_score]
                    
                    ## check if action changed game state
                    action_dict = self.detect_invalid_action(state, action, reward, action_dict, invalid_noun)
                    
                    ## get new state data
                    new_probs, new_actions, new_statevec, new_actionsvec, new_action_dict = self.get_data(new_state)
                    
                    ## remember round data
                    if training:
                        agent.remember(statevec, state, actionvec, reward, new_statevec,
                                       new_state, action_dict, False)
                    
                    ## if enough experiences in batch, replay 
                    if training and (i+1)%self.batch_size == 0 and i>0:  
                        print('Training on mini batch')
                        self.agent.replay(self.batch_size)
                        sleep(0.01)
                              
                    ## update progress bar
                    pbar.update(i + (game_number)*num_rounds) 
                    
                self.end_game_scores.loc[game_number] = [game_number, self.score]
                self.restart()
            except Exception as e:
                print('exception')
                print(e)
                traceback.print_tb(e.__traceback__)
                self.restart()
                #print(e.with_traceback())
            pbar.finish()
            self.stories.append(self.story)
        self.state_data.to_pickle('state_data.pickle')
        self.end_game()
        return True