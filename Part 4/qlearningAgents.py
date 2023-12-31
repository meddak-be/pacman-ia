# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from typing import Any, Dict, Optional
from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent
      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update
      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)
      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        ReinforcementAgent.__init__(self, **args)
        self.QValues = {}

    def getQValue(self, state, action) -> float:
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        if (state, action) in self.QValues:
          return self.QValues[(state, action)]
        else:
          return 0.0
        

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        qvalues = []
        for action in self.getLegalActions(state):
          qvalues.append(self.getQValue(state, action))
        if len(qvalues) == 0:
          return 0.0
        else:
          return max(qvalues)

    def computeActionFromQValues(self, state) -> Optional[str]:
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        bestAction = None
        maxQVal = self.getValue(state)
        for action in self.getLegalActions(state):
          qval = self.getQValue(state, action)
          if qval >= maxQVal:
            bestAction=action
            maxQVal = qval
        return bestAction

    def getAction(self, state) -> Optional[str]:
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.
          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        actions = self.getLegalActions(state)
        randomness = util.flipCoin(self.epsilon)
        if len(actions) == 0:
          return None
        if randomness:
          return random.choice(actions)
        else:
          return self.getPolicy(state)

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here
          NOTE: You should never call this function,
          it will be called on your behalf
        """
        
        qval = self.getQValue(state, action)
        alpha = self.alpha
        discount = self.discount
        qvalNext = self.getValue(nextState)
        #Application de l'équation 5
        newVal = (1 - alpha) * qval + alpha * (reward+discount*qvalNext)
        
        self.QValues[(state, action)] = newVal

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1
        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = super().getAction(state)
        self.doAction(state,action)
        return action



class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent
       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        #Application de l'équation 6
        featureVector = self.featExtractor.getFeatures(state, action)
        val=0
        for feature in featureVector:
          val+=self.weights[feature] * featureVector[feature]
        return val

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        #Application de l'equation 8
        featureVector = self.featExtractor.getFeatures(state, action)
        delta = reward + self.discount*self.getValue(nextState) - self.getQValue(state, action)
        for feature in featureVector:
            self.weights[feature] += self.alpha * delta * featureVector[feature]

    def final(self, state):
        """Called at the end of each game."""
        # call the super-class final method
        # you might want to print your weights here for debugging
        super().final(state)
        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            pass