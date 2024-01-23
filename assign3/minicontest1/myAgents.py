# myAgents.py
# ---------------
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

from game import Agent
from searchProblems import PositionSearchProblem

import util
import time
import search

from searchProblems import Directions

"""
IMPORTANT
`agent` defines which agent you will use. By default, it is set to ClosestDotAgent,
but when you're ready to test your own agent, replace it with MyAgent
"""

# 팩맨끼리 공유할 정보들이기에 전역변수로 선언
target = []         # 각 팩맨들이 현재 타겟으로 삼은 food의 좌표 리스트
pathToTarget = []   # 각 팩맨들의 타겟까지의 액션 리스트 (pathToTarget[index] = index번째의 팩맨의 타겟까지의 액션 리스트) 
init = True         # initialize 여부 (initialize 메서드에서는 state 변수를 활용할 수 없어 getAction에서 변수를 초기화하기 때문에 필요함)
done = []           # 각 팩맨들의 탐색이 끝났는지 여부
stopTime = []       # 각 팩맨들이 정지해있는 시간 리스트

def createAgents(num_pacmen, agent='MyAgent'):
    return [eval('MyAgent')(index=i) for i in range(num_pacmen)]

class MyAgent(Agent):
    """
    Implementation of your agent.
    """
    # 새로운 BFS 정의
    def newBFS(self, problem, target):
        fringe = util.Queue()
        current = (problem.getStartState(), [])
        fringe.push(current)
        closed = []
            
        while not fringe.isEmpty():
            node, path = fringe.pop()

            if problem.isGoalState(node):
                return path, node       # 팩맨에서 가장 가까운 food의 좌표(node)까지 함께 리턴
                
            if node not in closed:
                closed.append(node)

                for coord, move, _ in problem.getSuccessors(node):
                    if coord not in target:                     # successor의 좌표가 다른 팩맨의 타겟이 아닐때 큐에 넣음
                        fringe.push((coord, path + [move]))     # -> 팩맨끼리의 타겟이 겹치지 않도록 하여 팩맨을 분산시킴

                    else:            # 만약 successor의 좌표가 다른 팩맨의 타겟이라면, 즉 타겟이 겹친다면 (팩맨은 food만을 타겟으로 하기 때문에 자동으로 이 successor는 food임)
                        othersIndex = target.index(coord)   # 해당 successor의 좌표를 타겟으로 하고 있는 팩맨의 인덱스를 찾음
                        if len(pathToTarget[othersIndex]) > len(path) + 1: # 그 팩맨과 현재 팩맨의 해당 타겟까지의 거리를 비교
                            fringe.push((coord, path + [move]))            # 만약 현재 팩맨이 해당 타겟과 더 가깝다면, 해당 팩맨은 이 타겟을 BFS 검색에 포함함
                            pathToTarget[othersIndex], target[othersIndex] = [], (-1, -1)   # 기존에 해당 타겟을 향하던 팩맨은 타겟과 타겟까지의 액션을 초기화 -> 새로운 food 탐색하도록

        return [], (-1, -1)  

        
    def getAction(self, state):
        """
        Returns the next action the agent will take
        """

        "*** YOUR CODE HERE ***"
        
        # 변수 초기화
        global target, pathToTarget, init, done, stopTime
        if init == True:
            init = False
            target = [ (-1, -1) for i in range(state.getNumPacmanAgents()) ]
            pathToTarget = [ [] for i in range(state.getNumPacmanAgents()) ]
            done = [ False for i in range(state.getNumPacmanAgents()) ]
            stopTime = [ 0 for i in range(state.getNumPacmanAgents()) ]

        index = self.index

        if done[index]:                 # 팩맨의 탐색이 끝나서 정지해있던 경우
            if stopTime[index] > 10:    # 팩맨이 멈춰있었던 시간을 계산함
                done[index] = False     # 10이 넘는다면 한번 더 reachable한 food가 있는지 탐색
                stopTime[index] = 0     # 미로에서 좁고 긴 길목이 있을 경우, 한 팩맨이 지나가며 food를 먹으면 다른 팩맨들이 영원히 멈춰버리는 경우를 방지하기 위함
            else:
                stopTime[index] += 1    # 멈춰있었던 시간이 짧으면 시간을 늘려주고 정지 액션 리턴
                return Directions.STOP
        
        if not pathToTarget[index]:     # 팩맨의 현재 타겟까지의 액션 리스트가 비어있다면
            target[index] = (-1, -1)    # 타겟을 새로 지정하기 위해 먼저 초기화
            problem = AnyFoodSearchProblem(state, index)    # 현재 state를 이용해 가장 가까운 food 먹는 문제를 정의
            pathToTarget[index], target[index] = self.newBFS(problem, target)   # 새로운 BFS로 food 탐색하여 팩맨의 액션 리스트와 타겟에 저장
            if not pathToTarget[index]: # 탐색을 해도 액션 리스트가 비어있다면
                done[index] = True      # reachable food가 아직 없으므로 일단 정지
                return Directions.STOP
            
        action = pathToTarget[index][0] # 액션 리스트 중 첫번째 액션 (팩맨이 현재 취해야 할 첫 번째 액션)
        del(pathToTarget[index][0])     # 액션 리스트에서 삭제하면서 리턴
        return action               
        
        raise NotImplementedError()

    def initialize(self):
        """
        Intialize anything you want to here. This function is called
        when the agent is first created. If you don't need to use it, then
        leave it blank
        """

        "*** YOUR CODE HERE"
        global init
        init = True
        return
        raise NotImplementedError()

"""
Put any other SearchProblems or search methods below. You may also import classes/methods in
search.py and searchProblems.py. (ClosestDotAgent as an example below)
"""

class ClosestDotAgent(Agent):

    def findPathToClosestDot(self, gameState):
        """
        Returns a path (a list of actions) to the closest dot, starting from
        gameState.
        """
        # Here are some useful elements of the startState
        startPosition = gameState.getPacmanPosition(self.index)
        food = gameState.getFood()
        walls = gameState.getWalls()
        problem = AnyFoodSearchProblem(gameState, self.index)


        "*** YOUR CODE HERE ***"

        pacmanCurrent = [problem.getStartState(), [], 0]
        visitedPosition = set()
        # visitedPosition.add(problem.getStartState())
        fringe = util.PriorityQueue()
        fringe.push(pacmanCurrent, pacmanCurrent[2])
        while not fringe.isEmpty():
            pacmanCurrent = fringe.pop()
            if pacmanCurrent[0] in visitedPosition:
                continue
            else:
                visitedPosition.add(pacmanCurrent[0])
            if problem.isGoalState(pacmanCurrent[0]):
                return pacmanCurrent[1]
            else:
                pacmanSuccessors = problem.getSuccessors(pacmanCurrent[0])
            Successor = []
            for item in pacmanSuccessors:  # item: [(x,y), 'direction', cost]
                if item[0] not in visitedPosition:
                    pacmanRoute = pacmanCurrent[1].copy()
                    pacmanRoute.append(item[1])
                    sumCost = pacmanCurrent[2]
                    Successor.append([item[0], pacmanRoute, sumCost + item[2]])
            for item in Successor:
                fringe.push(item, item[2])
        return pacmanCurrent[1]

    def getAction(self, state):
        return self.findPathToClosestDot(state)[0]

class AnyFoodSearchProblem(PositionSearchProblem):
    """
    A search problem for finding a path to any food.

    This search problem is just like the PositionSearchProblem, but has a
    different goal test, which you need to fill in below.  The state space and
    successor function do not need to be changed.

    The class definition above, AnyFoodSearchProblem(PositionSearchProblem),
    inherits the methods of the PositionSearchProblem.

    You can use this search problem to help you fill in the findPathToClosestDot
    method.
    """

    def __init__(self, gameState, agentIndex):
        "Stores information from the gameState.  You don't need to change this."
        # Store the food for later reference
        self.food = gameState.getFood()

        # Store info for the PositionSearchProblem (no need to change this)
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition(agentIndex)
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def isGoalState(self, state):
        """
        The state is Pacman's position. Fill this in with a goal test that will
        complete the problem definition.
        """

        x,y = state
        if self.food[x][y] == True:
            return True
        return False

