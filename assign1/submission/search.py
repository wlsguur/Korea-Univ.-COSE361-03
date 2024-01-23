# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    start = problem.getStartState() # get start state as start node
    fringe = util.Stack() # use stack as fringe for DFS
    visited = [] # list that will store visited positions

    fringe.push((start, [])) # push start node in the fringe
    # keep the list of actions in the fringe as well, so that we can compute the list of actions until reach that node

    while not fringe.isEmpty():
        pos, actions = fringe.pop() # get position and actions(list of actions) of node from stack
        visited.append(pos) # check the position as visited

        if problem.isGoalState(pos):
            return actions        # if the position is goal, return the list of actions
        
        for successor, action, stepcost in problem.getSuccessors(pos):
            if successor not in visited: # else, push unvisited successor of the node and list of actions from start node to the node's successor
                fringe.push((successor, actions + [action]))
                # list of actions from start to successor
                # = list of actions from start to the node(actions) + action from the node(action)
    
    return [] # there is no pushing actions into the fringe -> return null list


    util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    start = problem.getStartState() # get start state as start node
    fringe = util.Queue() # use queue as fringe for BFS
    visited = [start] # list that will store visited positions

    fringe.push((start, [])) # push start node in the fringe
    # keep the list of actions in the fringe as well, so that we can compute the list of actions until reach that node

    while not fringe.isEmpty():
        pos, actions = fringe.pop() # get position and actions(list of actions) of node from stack

        if problem.isGoalState(pos):
            return actions          # if the position is goal, return the list of actions
        
        for successor, action, stepcost in problem.getSuccessors(pos):
            if successor not in visited: # else, push unvisited successor of the node and list of actions from start node to the node's successor
                fringe.push((successor, actions + [action]))
                # list of actions from start to successor
                # = list of actions from start to the node(actions) + action from the node(action)
                visited.append(successor) # check successor as visited
                # I don't know why, but only DFS needs checking visited after poping the node from fringe.
                # BFS, UCS, A*Search are all correct when check vitisted with pushing the node to fringe.
                # I just tried different cheching options and found this...

    return [] # there is no pushing actions into the fringe -> return null list
    util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    start = problem.getStartState() # get start state as start node
    fringe = util.PriorityQueue() # use priority queue as fringe for UCS
    visited = [start] # list that will store visited positions
    costs = dict() # dictionary that will store node's cost of list of actions
    costs[start] = 0 # start node with zero cost

    fringe.push((start, [], 0), 0) # push node, list of actions, cost from start node to that node, and priority
    # keep list of actions and it's cost, so that we can compute list of actions and it's cost of each nodes
    # priority is cost of list of actions that required to get node from start node

    while not fringe.isEmpty():
        pos, actions, cost = fringe.pop() # get position, list of actions(actions), and cost that required to get that position from fringe

        if problem.isGoalState(pos): 
            return actions          # if the position is goal, return the list of actions
        
        for successor, action, stepcost in problem.getSuccessors(pos):
            if (successor not in visited) or (cost + stepcost < costs[successor]):
                # else, select unvisited successor OR successor that have lower cost than same successor's cost (no matter visited or not).

                costs[successor] = cost + stepcost # compute cost from start to successor
                # cost from start to node's successor = (cost from start to the node) + (cost from the node to it's successor)

                fringe.push((successor, actions + [action], costs[successor]), costs[successor])
                # push that successor, list of actions from start to the node's successor, cost of list of actions, and cost of list of actions as priority.
                # list of actions from start to successor = list of actions from start to the node(actions) + action from the node(action)
                visited.append(successor) # check successor as visited
            
    return [] # there is no pushing actions into the fringe -> return null list
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    start = problem.getStartState() # get start state as start node
    fringe = util.PriorityQueue() # use priority queue as fringe for UCS
    visited = [start] # list that will store visited positions
    costs = dict() # dictionary that will store node's cost of list of actions ( g(n) )
    costs[start] = 0 # start node with zero cost

    fringe.push((start, [], 0), 0 + heuristic(start, problem)) # push node, list of actions, cost from start node to that node, and priority
    # keep list of actions and it's cost, so that we can compute list of actions and it's cost of each nodes
    # priority is f(n) = g(n) + h(n) where
    # g(n) = cost of list of actions that required to get node from start node (backward cost)
    # h(n) = heuristic score of the node (future cost)

    while not fringe.isEmpty():
        pos, actions, cost = fringe.pop() # get position(pos), list of actions(actions), and cost that required to get that position from fringe

        if problem.isGoalState(pos): 
            return actions          # if the position is goal, return the list of actions
        
        for successor, action, stepcost in problem.getSuccessors(pos):
            if (successor not in visited) or (cost + stepcost < costs[successor]):
                # else, select unvisited successor OR successor that have lower cost than same successor's cost (no matter visited or not)

                costs[successor] = cost + stepcost # compute cost from start to successor
                # cost from start to node's successor = (cost from start to the node) + (cost from the node to it's successor)
                # cost of list of actions NOT include heuristic score (h(n)), because costs is backward cost (g(n))

                fringe.push((successor, actions + [action], costs[successor]), costs[successor] + heuristic(successor, problem))
                # push that successor, list of actions from start to the node's successor, cost of list of actions, and priority.
                # priority is f(n) = g(n) + h(n) where
                # g(n) = cost of list of actions that required to get node from start node (backward cost)
                # h(n) = heuristic score of the node (future cost).
                # list of actions from start to successor = list of actions from start to the node(actions) + action from the node(action).
                visited.append(successor) # check successor as visited
    return [] # there is no pushing actions into the fringe -> return null list
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
