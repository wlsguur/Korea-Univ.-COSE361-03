# myTeam.py
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

# COSE361(03) Final Project 2/2
# 컴퓨터학과 최진혁
# student no. 2022320006

# scared 고스트를 먹도록 변경
# 공격자여도 우리 진영이면 상대 팩맨 먹도록 변경
# 고스트까지의 거리 feature를 continuous 하도록 변경.


from captureAgents import CaptureAgent
import random, time, util
from game import Directions, Actions
import game
import math

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffensiveOrDefensiveReflexAgent', second = 'DefensiveReflexAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
  """
  A base class for reflex agents that chooses score-maximizing actions
  """
  def __init__(self, index, timeForComputing=0.1):
    super().__init__(index, timeForComputing)

  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)

  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    actions = gameState.getLegalActions(self.index)

    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]
    

    foodLeft = len(self.getFood(gameState).asList())

    if foodLeft <= 2:
      bestDist = 9999
      for action in actions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.start,pos2)
        if dist < bestDist:
          bestAction = action
          bestDist = dist
      return bestAction

    return random.choice(bestActions)

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != util.nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
    return features

  def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'successorScore': 1.0}

class OffensiveOrDefensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
  def __init__(self, index, timeForComputing=0.1):
    super().__init__(index, timeForComputing)
    self.isOffensive = None

  def getFeatures(self, gameState, action):
    curState = gameState.getAgentState(self.index)
    successor = self.getSuccessor(gameState, action)
    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()
    opponents = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    features = util.Counter()

    if self.getScore(gameState) > len(self.getFoodYouAreDefending(successor).asList()) / 5:
      self.isOffensive = False
    elif not curState.isPacman and len([i for i in opponents if i.isPacman]) == 2:
      self.isOffensive = False
    else:
      self.isOffensive = True

    if self.isOffensive:
      # 남은 food들의 갯수 계산
      foodList = self.getFood(successor).asList()   
      features['successorScore'] = -len(foodList)#self.getScore(successor)

      # Compute distance to the nearest food
      if len(foodList) > 0: # This should always be True,  but better safe than sorry
        minDistToFood = min([self.getMazeDistance(myPos, food) for food in foodList])
        features['distanceToFood'] = minDistToFood
      
      # ghost까지의 거리 계산
      ghosts = [i for i in opponents if not i.isPacman]
      ghostsPos = [ghost.getPosition() for ghost in ghosts]
      ghostsScaredTimer = [ghost.scaredTimer for ghost in ghosts]
      if ghostsScaredTimer and min(ghostsScaredTimer) > 5:
          features['distanceToGhost'] = 10 * min(ghostsScaredTimer)
      elif myState.isPacman:
        # 고스트까지의 실제 거리 + scared 시간 = 고스트까지의 거리로 하는건 어떨까??
        distToGhost = [self.getMazeDistance(myPos, gp) for gp in ghostsPos]
        if distToGhost:
          minDistToGhost = min(distToGhost)
          if minDistToGhost <= 10:
              features['distanceToGhost'] = -9 / (minDistToGhost + 0.001)
          elif minDistToGhost <= 20:
              features['distanceToGhost'] = (9 / 100) * minDistToGhost - 9/5
          else:
              features['distanceToGhost'] = 0

      
      # opponents는 successor 기준임. 그래서 액션을 취했을 때 상대 에이전트가 그 좌표에 있는지를 계산 못함. (이미 먹히고 난 뒤일테니까)
      curOpponents = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
      curGhosts = [i for i in curOpponents if not i.isPacman]

      for ghost in curGhosts:
        if myPos == ghost.getPosition() and ghost.scaredTimer > 0:
          features['eatGhost'] = 1

      invaders = [i for i in curOpponents if i.isPacman]
      for invader in invaders:
        if not curState.isPacman and myPos == invader.getPosition():
          features['eatInvader'] = 1
      

      ## 경계선에서 대치만 하는 문제를 해결하기 위해 y축 방향으로 멀어지도록 함
      
      #elif not myState.isPacman:
        #features['distanceToGhost'] = 
      #print(features['distanceToGhost'])
        
      #capsule을 먹을 수 있는지 판단
      curCapsules = self.getCapsules(gameState)
      for capsule in curCapsules:
        if myPos == capsule:
          features['eatCapsule'] = 1
      #capsule까지의 거리 계산
      capsules = self.getCapsules(successor)
      if capsules:
        distToCapsule = [self.getMazeDistance(myPos, capsule) for capsule in capsules]
        minDistToCapsule = min(distToCapsule)
        features['distanceToCapsule'] = minDistToCapsule
      
      # 충분한 food를 먹었다면 myside에 deposite 하도록
      ###
      ### home까지의 거리에다가 numCarrying을 곱한 것을 feature로 해봤음
      ###
      freeEat = (
                  (len([i for i in opponents if i.isPacman]) == 2 and curState.isPacman)
                    or (min(ghostsScaredTimer) > 5)
                )
      homePos = gameState.getInitialAgentPosition(self.index)
      if freeEat:
        features['distanceToHome'] = self.getMazeDistance(myPos, homePos)
      else:
        features['distanceToHome'] = 10 * self.getMazeDistance(myPos, homePos) * myState.numCarrying


    # defensive 역할 수행
    else:
      features['onDefense'] = 1
      if myState.isPacman: features['onDefense'] = 0
      invaders = [i for i in opponents if i.isPacman]
      features['numInvaders'] = len(invaders)

      if invaders:
        invadersPos = [invader.getPosition() for invader in invaders]
        minDistToInvaders = min([self.getMazeDistance(myPos, ip) for ip in invadersPos])
        features['distanceToInvader'] = minDistToInvaders

      if action == Directions.STOP: features['stop'] = 1
      rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
      if action == rev: features['reverse'] = 0


    return features

  def getWeights(self, gameState, action):
    offensive = {'successorScore': 1000, 'distanceToFood': -1, 'distanceToGhost': 10, 'eatGhost': 9999, 'eatInvader': 9999, 'eatCapsule': 9999, 'distanceToCapsule': -10, 'distanceToHome': -1}
    defensive = {'numInvaders': -1000, 'onDefense': 100, 'distanceToInvader': -10, 'stop': -100, 'reverse': -2}
    return offensive if self.isOffensive else defensive

class DefensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """

  def getFeatures(self, gameState, action):
    gameState = gameState
    successor = self.getSuccessor(gameState, action)
    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()
    opponents = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    features = util.Counter()

    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0
    invaders = [i for i in opponents if i.isPacman]
    features['numInvaders'] = len(invaders)

    if invaders:
      invadersPos = [invader.getPosition() for invader in invaders]
      minDistToInvaders = min([self.getMazeDistance(myPos, ip) for ip in invadersPos])
      features['distanceToInvader'] = minDistToInvaders

    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    #여기까진 baseline과 동일
    #capsule 지키기
    capsules = self.getCapsulesYouAreDefending(gameState)
    targetCapsule = None
    minDistOppToCapsule = 9999
    for capsule in capsules:
      for opponent in opponents:
        distOppToCapsule = self.getMazeDistance(capsule, opponent.getPosition())
        if distOppToCapsule < minDistOppToCapsule:
          targetCapsule = capsule
          minDistOppToCapsule = distOppToCapsule
    if targetCapsule != None:
        distToTargetCapsule = self.getMazeDistance(myPos, targetCapsule)
        disDiff = distOppToCapsule - distToTargetCapsule
        if distOppToCapsule > 10:
          features['disDiffToCapsule'] = -distToTargetCapsule
        elif distOppToCapsule > 5:
          if distToTargetCapsule <  distOppToCapsule - 3:
            features['disDiffToCapsule'] = (-10) * distToTargetCapsule
          elif distToTargetCapsule > distOppToCapsule:
            features['disDiffToCapsule'] = -999
          else:
            features['disDiffToCapsule'] = (-100) * distToTargetCapsule
        else:
          if distToTargetCapsule > distOppToCapsule:
            features['disDiffToCapsule'] = -9999
          else:
            features['disDiffToCapsule'] = (-100) * (2 ** distToTargetCapsule)

    return features

  def getWeights(self, gameState, action):
    return {'numInvaders': -1000, 'onDefense': 100, 'distanceToInvader': -10, 'stop': -100, 'reverse': -2, 'disDiffToCapsule': 1}