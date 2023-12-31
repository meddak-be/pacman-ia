# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


from typing import Optional

from learningAgents import ValueEstimationAgent
from mdp import MarkovDecisionProcess
import util


class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp: MarkovDecisionProcess, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.values = util.Counter()
        self.runValueIteration(iterations)

    def runValueIteration(self, iterations: int):

        for i in range(iterations):
            states = self.mdp.getStates()
            V = util.Counter()
            for state in states:
                maxQ = float("-inf")
                #parmi les action, on cherche la meilleure
                for action in self.mdp.getPossibleActions(state): 
                    Q = self.getQValue(state, action)
                    if Q > maxQ:
                        maxQ = Q
                    V[state] = maxQ
            #met à jour les valeurs des états
            self.values =  V  

    def getValue(self, state) -> float:
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]
        
    def computeQValueFromValues(self, state, action) -> float:
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        val = 0
        statesProbs = self.mdp.getTransitionStatesAndProbs(state, action)
        for nextState, prob in statesProbs:
            reward = self.mdp.getReward(state, action, nextState)
            val += prob * (reward + self.discount * self.values[nextState])
        return val
        
        util.raiseNotDefined()

    def computeActionFromValues(self, state) -> Optional[str]:
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        best_action = None
        max_val = float("-inf")
        #parmi les actions, retourné la meilleure
        for action in self.mdp.getPossibleActions(state):
          q_value = self.computeQValueFromValues(state, action)
          if q_value > max_val:
            max_val = q_value
            best_action = action
        return best_action
        
    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
