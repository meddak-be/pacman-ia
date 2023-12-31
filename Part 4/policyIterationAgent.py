from valueIterationAgents import ValueIterationAgent
from mdp import MarkovDecisionProcess
import util


class PolicyIterationAgent(ValueIterationAgent):
    def __init__(self, mdp: MarkovDecisionProcess, discount=0.9, iterations=100,value_iteration_tolerance=.01):
        """
        Your policy iteration agent should take an mdp on
        construction, run the indicated number of iterations
        or stop on convergeance and then act according to the
        resulting stored policy.

        Some useful mdp methods you will use:
            mdp.getStates()
            mdp.getPossibleActions(state)
            mdp.getTransitionStatesAndProbs(state, action)
            mdp.getReward(state, action, nextState)
            mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, 0)
        self.policy = {
            state: mdp.getPossibleActions(state)[0]
            if not mdp.isTerminal(state)
            else None
            for state in mdp.getStates()
        }
        self.value_iteration_tolerance = value_iteration_tolerance
        self.runPolicyIteration(iterations)

    def runPolicyIteration(self, iterations: int):
        """ do NOT modify this function """
        converged = False
        self.iterations_to_converge = 0
        while not converged and self.iterations_to_converge < iterations:
            self.iterations_to_converge += 1
            self.updateValues()
            converged = self.updatePolicy()
        
    def updateValues(self):
        """Update the values until convergeance. Instead of 
        maximizing over the arms when computing new values, 
        you should select the action defined by the agent's 
        current policy stored in self.policy. 
        You can consider that the values converged when the
        biggest change in value is smaller than 
        self.value_iteration_tolerance."""
        while True:
            #conserve le plus grand delta pour la convergeance
            maxDelta = 0
            for state in self.mdp.getStates():
                if self.policy[state] != None:
                    val = 0
                    #calcule de la valeur en fonction de la politique actuelle
                    for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, self.policy[state]):
                        reward = self.mdp.getReward(state, self.policy[state] ,nextState)
                        val += prob * (reward +  self.discount*self.values[nextState])
                    #met a jour le delta
                    maxDelta = max(maxDelta, abs(self.values[state] - val))
                    self.values[state] = val
            if maxDelta < self.value_iteration_tolerance:
                break                

    def updatePolicy(self):
        """In this method you should update the agent's policy, 
        stored in self.policy, and return a boolean 
        indicating whether the policy has converged."""
        stable = True
        for state in self.mdp.getStates():
            maxVal = self.values[state]
            for action in self.mdp.getPossibleActions(state):
                val = 0
                #calcule de la valeur pour chaque action
                for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                    reward = self.mdp.getReward(state, action ,nextState)
                    val += prob * (reward +  self.discount*self.values[nextState])
                #si la valeur de l'action est meilleure, on la prend
                if val > maxVal and action != self.policy[state]:
                    self.policy[state] = action
                    maxVal = val
                    #si changement, la politique n'est toujours pas stable
                    stable = False
        
        return stable
       

    def getPolicy(self, state):
        return self.policy[state]

    def getAction(self, state):
        return self.policy[state]
