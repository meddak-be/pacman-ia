# multiAgents.py
# --------------
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


from turtle import pos
from util import manhattanDistance
from game import Directions
from pacman import GameState
import util

from game import Agent



def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, state: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.getNextState(agentIndex, action):
        Returns the child game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        def value(state, nextAgent, currentAgent, currentDepth, maxDepth, action):
            actions = state.getLegalActions(agentIndex=currentAgent)
            if currentDepth == maxDepth or len(actions) == 0:
                return (self.evaluationFunction(state), action)
            else:
                if nextAgent == "MAX":
                    maximum = (-999999, None)
                    for act in actions:
                        nextState = state.getNextState(agentIndex=currentAgent, action=act)
                        val = value(nextState, "MIN", 1, currentDepth, maxDepth, act)
                        maximum = max(val, maximum, key = lambda t: t[0])
                    return maximum

                elif nextAgent == "MIN":
                    minimum = (999999, None)
                    for act in actions:
                        nextState = state.getNextState(agentIndex=currentAgent, action=act)
                        if currentAgent < state.getNumAgents()-1:
                            val = value(nextState, "MIN", currentAgent+1, currentDepth, maxDepth, act) #tous les fantomes jouent
                        else:
                            val = value(nextState, "MAX", 0, currentDepth+1, maxDepth, act) #profondeur suivante
                        val = (val[0], action)
                        minimum = min(val, minimum, key = lambda t: t[0])
                    return minimum
        
        action = value(state, "MAX", 0, 0, self.depth, None)[1]
        return action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, state: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def value(state, nextAgent, currentAgent, currentDepth, maxDepth, action, alpha, beta):
            actions = state.getLegalActions(agentIndex=currentAgent)
            if currentDepth == maxDepth or len(actions) == 0:
                return (self.evaluationFunction(state), action)
            else:
                if nextAgent == "MAX":
                    maximum = (-999999, None)
                    for act in actions:
                        nextState = state.getNextState(agentIndex=currentAgent, action=act)
                        val = value(nextState, "MIN", 1, currentDepth, maxDepth, act, alpha, beta)
                        
                        maximum = max(val, maximum, key = lambda t: t[0])
                        if maximum[0] > beta:
                            return maximum  
                        alpha = max(alpha, maximum[0])
                    return maximum

                elif nextAgent == "MIN":
                    minimum = (999999, None)
                    for act in actions:
                        nextState = state.getNextState(agentIndex=currentAgent, action=act)
                        if currentAgent < state.getNumAgents()-1:
                            val = value(nextState, "MIN", currentAgent+1, currentDepth, maxDepth, act, alpha, beta)
                        else:
                            val = value(nextState, "MAX", 0, currentDepth+1, maxDepth, act, alpha, beta)
                        val = (val[0], action)
                        minimum = min(val, minimum, key = lambda t: t[0])
                        if minimum[0] < alpha:
                            return minimum
                        beta = min(beta, minimum[0])
                    return minimum
        
        action = value(state, "MAX", 0, 0, self.depth, None, -99999, 99999)[1]
        return action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, state: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def value(state, nextAgent, currentAgent, currentDepth, maxDepth, action):
            actions = state.getLegalActions(agentIndex=currentAgent)
            if currentDepth == maxDepth or len(actions) == 0:
                return (self.evaluationFunction(state), action)
            else:
                if nextAgent == "MAX":
                    maximum = (-999999, None)
                    for act in actions:
                        nextState = state.getNextState(agentIndex=currentAgent, action=act)
                        val = value(nextState, "EXP", 1, currentDepth, maxDepth, act)
                        maximum = max(val, maximum, key = lambda t: t[0])
                    return maximum

                elif nextAgent == "EXP":
                    utilities = []
                    for act in actions:
                        nextState = state.getNextState(agentIndex=currentAgent, action=act)
                        if currentAgent < state.getNumAgents()-1:
                            val = value(nextState, "EXP", currentAgent+1, currentDepth, maxDepth, act)
                        else:
                            val = value(nextState, "MAX", 0, currentDepth+1, maxDepth, act)
                        val = (val[0], action)
                        utilities.append(val[0])
                    return (float(sum(utilities)/len(utilities)), action)
        
        action = value(state, "MAX", 0, 0, self.depth, None)[1]
        return action


def betterEvaluationFunction(state: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    L'évalutation prend en compte
        - le nombre de food restant (qu'on veut minimiser)
        - la distance à la food la plus proche (qu'on veut maximiser)
        - le score (qu'on veut maximiser)
        - un indice de danger si un ghost est trop proche (qu'on veut minimiser)
        - un indice de rapprochement des ghost quand pacman peut les manger(qu'on veut maximiser)
        - la distance à la capsule la plus proche (qu'on veut maximiser)
    """
    "*** YOUR CODE HERE ***"     
   #attributs importants
    score = state.getScore()
    capsules = state.getCapsules()
    foods = state.getFood().asList()
    foods.extend(capsules)
    pacman = state.getPacmanPosition()
    ghosts = state.getGhostPositions()


    #la raison pour laquelle il ya 999999999 dans les listes est dans le cas où elle est vide.
    #A chaque utilisation de cette liste, elle se trouve au dénominateur et 
    #la fonction min est appelée dessus. Donc il il n'y a
    #que 999999999, 1/999999999 tendera vers 0 ce qui ne change presque rien à l'éval et si il y a
    #une autre valeur, elle sera forcement plus petite. 
    foodDistances = [999999999]
    for food in foods:
        foodDistances.append(abs(food[0]-pacman[0]) + abs(food[1]-pacman[1]))

    ghostDistances = [999999999]
    for ghost in ghosts:
        ghostDistances.append(abs(ghost[0]-pacman[0]) + abs(ghost[1]-pacman[1]))
    
    capsulesDistances = [999999999]
    for cap in capsules:
        capsulesDistances.append(abs(cap[0]-pacman[0]) + abs(cap[1]-pacman[1]))
    
    #si un ghost est trop proche (si il est vraiment trop proche, on return une grande valeur négative pour fuir)
    danger = 0
    if min(ghostDistances) < 4:
        if min(ghostDistances) < 2:
            return -999999999
        danger = 1/min(ghostDistances)

    #Pour se rapprocher des ghosts quand ils sont effrayés pour les manger
    closestGhost = 0 
    if state.getGhostStates()[0].scaredTimer:
        closestGhost =  1/min(ghostDistances)
        
    #si on peut manger tous les foods, c'est qu'on peut gagner    
    if len(foods) == 0 :
        return 999999999 

    evaluation = -len(foods)*65 + (1/min(foodDistances))*30 + score*100 - danger*50 + closestGhost*65 + (1/min(capsulesDistances))*30
    return evaluation

# Abbreviation
better = betterEvaluationFunction
