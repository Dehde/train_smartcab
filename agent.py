

import random
from smartcab.environment import Agent, Environment
from smartcab.planner import RoutePlanner
from smartcab.simulator import Simulator
import numpy as np
from collections import OrderedDict

random.seed(42)

class Utility():
    def __init__(self, valid_actions, valid_inputs, alpha=0.5, gamma=0.2, epsilon=0.1):
        self.Q_values = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.valid_actions = valid_actions
        stateActionSpace = self.init_stateActionSpace(valid_actions, valid_inputs)
        for state_and_action_pair in stateActionSpace:
            self.Q_values[state_and_action_pair] = 0
    
    def update(self, old_state, action, reward, new_state):
        #Q(s,a) = (1-/alpha)Q(s,a) + /alpha ( R(s) + 
        #sum_over_all_possible_next_states_s' /gamma max(Q(s',a') , with regard of a'))
        oldStateKey = self.createStateActionKey(old_state, action)
        Q = self.Q_values[oldStateKey]
        Q_max = 0
        for action in self.valid_actions:
            newStateKey = self.createStateActionKey(new_state, action)
            Q_candidate = self.Q_values[newStateKey]
            if Q_candidate > Q_max:
                Q_max = Q_candidate
            
        Q = (1-self.alpha)*Q + self.alpha*(reward + self.gamma*Q_max)
        
        self.alpha -= self.alpha/200
        print "!!!! alpha = {}".format(self.alpha)
        self.Q_values[oldStateKey] = Q
    
    def policy(self, state):
        randomNumber = random.random()
        if randomNumber > (1-self.epsilon):
            print "!!! The following action is taken randomly !!!"
            return random.choice(self.valid_actions)
        else:
            Q_max = 0
            best_action = []
            for action in self.valid_actions:
                stateActionKey = self.createStateActionKey(state, action)
                Q = self.Q_values[stateActionKey]
                if Q > Q_max:
                    Q_max = Q
                    best_action = [action]
                if Q == Q_max:
                    best_action += [action]
                    
            if len(best_action)== 1:
                return best_action[0]
            else:
                return random.choice(best_action)
    
    def init_stateActionSpace(self, valid_actions, valid_inputs):
        possible_next_waypoints = valid_actions[1:]
        stateActionSpace = []
        state = {}
        for light in ['red','green']:
            for left in valid_inputs['left']:
                for right in valid_inputs['right']:
                    for oncoming in valid_inputs['oncoming']:
                        for next_waypoint in possible_next_waypoints:
                            for action in valid_actions:
                                state = {'light': light, 'next_waypoint': next_waypoint, 'right':right,
                                        'oncoming': oncoming, 'left': left}
                                stateActionSpace += [self.createStateActionKey(
                                    state, action)]
        return stateActionSpace
    
    def createStateActionKey(self, state, action):
        return ("{"+
                ("'light': {0!r}, 'next_waypoint': {1!r}, 'right': {2!r}, 'oncoming': {3!r},"
                "'left': {4!r}").format(state['light'], state['next_waypoint'], state['right'],
                                        state['oncoming'], state['left'])+
                "}" + 
                "action: {0!r}".format(action))
    
        
class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, 
                                                  # next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        
        # TODO: Initialize any additional variables here
        self.valid_actions = env.valid_actions
        self.valid_inputs = env.valid_inputs
        self.utility = Utility(self.valid_actions, self.valid_inputs)
    
    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        
        # TODO: Update state
        self.state = {'next_waypoint':str(self.next_waypoint)}
        self.state.update(inputs)
        old_state = self.state

        #Policy(s) = argmax(Q(s,a) , with regard to a)
        
        action = self.utility.policy(self.state)
    
        # Execute action and get reward
        reward = self.env.act(self, action)
        
        # TODO: Learn policy based on state, action, reward
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        
        #get new state
        inputs = self.env.sense(self)
        self.state = {'next_waypoint':str(self.next_waypoint)}
        self.state.update(inputs)
        new_state = self.state
        
        #update utility
        self.utility.update(old_state, action, reward, new_state)
        
        print "next_waypoint: {}".format(self.next_waypoint)
        print "LearningAgent.update(): deadline = "         "{}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]
        
           

def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.5, display=False)  # create simulator 
    # (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()

