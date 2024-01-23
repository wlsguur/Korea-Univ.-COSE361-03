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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        import math

        foodList = newFood.asList()         # list for cordinates of foods
        foodDists = []                      # list for distances to each food

        for food in foodList:
            dx = (newPos[0] - food[0])**2   # x difference between pacman's new position and the food
            dy = (newPos[1] - food[1])**2   # y difference between pacman's new position and the food
            dist = math.sqrt(dx + dy)       # euclidean distance
            foodDists.append(dist)          # add it to list for distances to each food 
        if len(foodDists) != 0:
            foodScore = sum(foodDists) / len(foodDists)
            # get average distance for all foods. We should eat foods as much as possible, so not closest distance, but consider all foods.
        else:
            foodScore = 1       # if there is no food -> the game is ended. no matter what foodScore is.
        
        ghostDist = 9999        # we will find the closest distance to the ghost. so initialize it as inf. first.
        for ghost, scaredTime in zip(newGhostStates, newScaredTimes):
            ghostPos = ghost.getPosition()      # get each ghost's position
            isScared = scaredTime > 0           # get each ghost is scared or not

            if isScared == 1:   # if the ghost is scared, we don't need to consider the ghost
                continue
            else:
                dx = (newPos[0] - ghostPos[0])**2   # x difference between pacman's new position and the ghost
                dy = (newPos[1] - ghostPos[1])**2   # y difference between pacman's new position and the ghost
                temp = math.sqrt(dx + dy)           # get euclidean distance between pacman and the ghost
                ghostDist = min(temp, ghostDist)    # find the minimum distance. We should not to die, so the closest ghost is most important.

        return successorGameState.getScore() + (ghostDist/foodScore)
        # ghost distance(ghostDist) is better to big, average food distance(foodScore) is better to small.
        # so ghostDist to numerator, foodScore to denominator.

def scoreEvaluationFunction(currentGameState):
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
    to the MinimaxPacmanAgent, alphabetaPacmanAgent & ExpectimaxPacmanAgent.

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

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        
        def minimaxSearch(self, state, agent, depth):
            if agent == 0:                  # if agent is pacman
                nextDepth = depth-1         # decrease the depth 1 each pacman's move because pacman and ghosts will move (depth) times. so start with depth = self.depth+1
                optimizer = max             # pacman will choose max value
                bestScore = -9999           # initialize the max value

            else:                           # if agent is ghost
                nextDepth = depth           # depth will not change in ghost's move
                optimizer = min             # ghost will choose min value
                bestScore = 9999            # initialize the min value

            nextAgent = (agent + 1) % state.getNumAgents()          # agent number will rotate [ 0, getNumAgents() )
  
            if nextDepth == 0 or state.isWin() or state.isLose():   # search until the maximum depth or terminal nodes
                return self.evaluationFunction(state), None         # return score using evaluation function
            
            actionList = state.getLegalActions(agent)               # get agent's actions to successor as list
            bestAction = None                                       # best action of the agent

            for action in actionList:
                successor = state.generateSuccessor(agent, action)  # get each successor of the agent's action
                score, nextAction = minimaxSearch(self, successor, nextAgent, nextDepth)    # get successor's score with action using recursion
                bestScore = optimizer(bestScore, score)     # choose best option (best score)
                if bestScore == score:                              
                    bestAction = action     # when choose new option (best score is just changed), choose that action as best action

            return bestScore, bestAction
            # return best score with best action (we need best score when parent node choose best option,
            # but getAction() function should return the action. so both is needed.)

        return minimaxSearch(self, gameState, 0, self.depth + 1)[1]     # return the best action using recursive minimax search
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def minimaxSearchWithAB(self, state, agent, depth, alpha, beta):
            if agent == 0:                  # if agent is pacman
                nextDepth = depth-1         # decrease the depth 1 each pacman's move because pacman and ghosts will move (depth) times. so start with depth = self.depth+1
                optimizer = max             # pacman will choose max value
                bestScore = -9999           # initialize the max value

            else:                           # if agent is ghost
                nextDepth = depth           # depth will not change in ghost's move
                optimizer = min             # ghost will choose min value
                bestScore = 9999            # initialize the min value
            
            nextAgent = (agent + 1) % state.getNumAgents()          # agent number will rotate [ 0, getNumAgents() )
  
            if nextDepth == 0 or state.isWin() or state.isLose():   # search until the maximum depth or terminal nodes
                return self.evaluationFunction(state), None         # return score using evaluation function
            
            actionList = state.getLegalActions(agent)               # get agent's actions to successor as list
            bestAction = None                                       # best action of the agent

            for action in actionList:
                successor = state.generateSuccessor(agent, action)  # get each successor of the agent's action
                score, nextAction = minimaxSearchWithAB(self, successor, nextAgent, nextDepth, alpha, beta) # get successor's score with action using recursion

                bestScore = optimizer(bestScore, score)     # choose best option (best score)
                if bestScore == score:
                    bestAction = action                     # when choose new option (best score is just changed), choose that action as best action

                if agent == 0:                              # alpha beta pruning with max-value (pacman)
                    if bestScore > beta:                    # if this pacman's current best option is biger than min's best,
                        return bestScore, bestAction        # min will never choose this pacman's other successor's option. so just return the action. (no futher search)
                    alpha = optimizer(alpha, bestScore)     # else, update max's best option

                else:                                       # alpha beta pruning with min-value (ghost)
                    if bestScore < alpha:                   # if this ghost's current best option is smaller than max's best,
                        return bestScore, bestAction        # max will never choose this ghost's other successor's option. so just return the aciton. (no futher search)
                    beta = optimizer(beta, bestScore)       # else, update min's best option

            return bestScore, bestAction
            # return best score with best action (we need best score when parent node choose best option,
            # but getAction() function should return the action. so both is needed.)

        return minimaxSearchWithAB(self, gameState, 0, self.depth+1, alpha=-9999, beta=9999)[1]
        # return the best action using recursive minimax search
        # initialize the max's best as -inf.
        # initialize the min's best as inf.
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
