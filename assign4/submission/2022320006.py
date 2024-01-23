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

# your_baseline3 은 your_baseline2 의 발전된 버전
# defensive agent의 코드는 your_baseline2와 완전히 같음
# 추가 또는 변경된 사항은 아래와 같음

# 1. 고스트까지의 거리 feature를 continuous 하도록 변경.
# 2. offensive agent가 scared 고스트를 먹도록 변경
# 3. offensive agent가 우리 진영에 있는 상황에서, 상대 팩맨의 바로 옆을 지나간다면 상대 팩맨을 먹도록 변경
# 4. offensive agent가 방어자 모드일 땐 캡슐을 지키지 않음 -> 방어자끼리 분산시켜서 더 효율적으로 방어하도록

# 해당하는 부분 위주로 주석을 달았습니다

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
  # 가능한 action들을 모두 구함
    actions = gameState.getLegalActions(self.index)

    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    # 액션들을 evaluation
    values = [self.evaluate(gameState, a) for a in actions]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

    # evaluation value가 가장 높은 액션들을 구함
    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    foodLeft = len(self.getFood(gameState).asList())

    # 남은 음식이 2개 이하일 경우
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

    # 베스트 액션들 중에서 랜덤으로 리턴, 그러나 실제로는 feature가 복잡하기때문에 같은 값을 가지는 액션이 존재할 확률은 거의 없음.
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

    # 공격자 모드일 경우 필요한 feature들 계산
    if self.isOffensive:
      # 남은 food들의 갯수 계산
      foodList = self.getFood(successor).asList()   
      features['successorScore'] = -len(foodList)#self.getScore(successor)

      # Compute distance to the nearest food
      if len(foodList) > 0: # This should always be True,  but better safe than sorry
        minDistToFood = min([self.getMazeDistance(myPos, food) for food in foodList])
        features['distanceToFood'] = minDistToFood
      
      #
      # ghost까지의 최단 거리를 계산하여 이를 feature에 반영함
      # 고스트들의 위치와 scared 시간을 구함
      ghosts = [i for i in opponents if not i.isPacman]
      ghostsPos = [ghost.getPosition() for ghost in ghosts]
      ghostsScaredTimer = [ghost.scaredTimer for ghost in ghosts]

      if ghostsScaredTimer and min(ghostsScaredTimer) > 5:
          features['distanceToGhost'] = min(ghostsScaredTimer) # 상대 ghost들이 모두 scared 상태라면 남은 scared 시간을 피쳐값으로 사용
      # 고스트까지의 거리를 feature 값으로 반영할 때, 함수 모양에 신경을 많이 씀
      # 거리가 20 이하일 때 ghost와의 거리를 고려함
      # 20~11일 때는 선형 함수 모양, 10~0일 때는 유리함수 모양, 거리가 작을수록 evaluation value를 안좋게 함
      # 구간의 경계선에서는 함수를 미분 가능하도록 하였음
      elif myState.isPacman:  # 팩맨일 때만 ghost까지의 거리를 고려하도록 변경함
        distToGhost = [self.getMazeDistance(myPos, gp) for gp in ghostsPos] # 고스트들까지의 거리 계산
        if distToGhost:
          minDistToGhost = min(distToGhost)   # 고스트까지의 최단 거리 계산
          if minDistToGhost <= 10:
              features['distanceToGhost'] = -9 / (minDistToGhost + 0.001)
          elif minDistToGhost <= 20:
              features['distanceToGhost'] = (9 / 100) * minDistToGhost - 5/9
          else:
              features['distanceToGhost'] = 0
      # feature의 절댓값이 감소한 만큼, 가중치를 10으로 증가시킴

      # opponents는 successor 기준 -> 그래서 액션을 취했을 때 상대 에이전트가 그 좌표에 있는지를 계산 못함. (이미 먹히고 난 뒤이기 때문)
      # 현재 state를 이용해 action을 취하기 전 정보로 현재 상대 agent들과 고스트들의 위치를 구함
      curOpponents = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
      curGhosts = [i for i in curOpponents if not i.isPacman]
      #
      # scared 고스트를 만나면 무조건 먹도록 함
      #
      for ghost in curGhosts:
        if myPos == ghost.getPosition() and ghost.scaredTimer > 0: 
          features['eatGhost'] = 1
          # 액션을 취했을 때 위치가 srared 고스트의 위치랑 같다면 고스트를 먹는 것이므로 1, 아니면 0 -> 가중치를 매우 크게
      
      #
      # 공격자 모드에서도, 우리 진영에서 상대 팩맨을 마주친다면 무조건 먹도록 함
      #
      invaders = [i for i in curOpponents if i.isPacman]
      for invader in invaders:
        if not curState.isPacman and myPos == invader.getPosition():
          features['eatInvader'] = 1
          # 내가 팩맨이 아니고, 액션을 취했을 때 포지션이 상대 팩맨과 같다면 상대 팩맨을 먹는 것이므로 1, 아니면 0 -> 가중치를 매우 크게
      
      #
      # 해당 action을 통해 캡슐을 먹을 수 있다면, 무조건 먹도록 함
      #
      # 만약 해당 action으로 캡슐을 먹는 상황이라면, action의 successor로 캡슐을 구했을 땐 캡슐은 이미 없어지고 난 상황임
      # 이미 팩맨이 캡슐을 먹은 상황에서 캡슐 위치를 불러오는 것이기 때문
      # 그래서 현재 state로 캡슐 위치를 구해야 함
      curCapsules = self.getCapsules(gameState)
      for capsule in curCapsules:
        if myPos == capsule:
          features['eatCapsule'] = 1    # 먹을 수 있는 상황이면 1, 아니라면 0 -> 가중치를 매우 높게 부여함

      #
      # capsule까지의 최단 거리 계산
      # 
      capsules = self.getCapsules(successor)
      if capsules:
        distToCapsule = [self.getMazeDistance(myPos, capsule) for capsule in capsules]
        minDistToCapsule = min(distToCapsule)
        features['distanceToCapsule'] = minDistToCapsule
      
      #
      # 충분한 food를 먹었다면 myside에 deposite 하도록
      #
      freeEat = (
                  (len([i for i in opponents if i.isPacman]) == 2 and curState.isPacman)
                    or (min(ghostsScaredTimer) > 5)
                )   # 상대 고스트들이 scared 타임인 경우도 추가로 free eat 상황이라고 판단
      homePos = gameState.getInitialAgentPosition(self.index)
      if freeEat:
        features['distanceToHome'] = self.getMazeDistance(myPos, homePos)
      else:
        features['distanceToHome'] = 10 * self.getMazeDistance(myPos, homePos) * myState.numCarrying
        # 가중치를 조금 변경함


    # 방어자 모드일 경우 필요한 feature들을 계산
    # 캡슐 주변을 지키지 않도록 함 -> 어짜피 캡슐 주변은 defensive agent가 지킬 것이므로, 방어자들을 분산시키기 위함
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
      if action == rev: features['reverse'] = 1

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
  # your_baseline2와 코드 같음

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

    # capsule 지키기
    # your_baselin2 코드랑 똑같아서 주석은 따로 달지 않음.
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
          features['disDiffToCapsule'] = distToTargetCapsule
        elif distOppToCapsule > 5:
          if distToTargetCapsule <  distOppToCapsule - 3:
            features['disDiffToCapsule'] = 10 * distToTargetCapsule
          elif distToTargetCapsule > distOppToCapsule:
            features['disDiffToCapsule'] = 999
          else:
            features['disDiffToCapsule'] = 100 * distToTargetCapsule
        else:
          if distToTargetCapsule > distOppToCapsule:
            features['disDiffToCapsule'] = 9999
          else:
            features['disDiffToCapsule'] = 100 * (2 ** distToTargetCapsule)

    return features

  def getWeights(self, gameState, action):
    return {'numInvaders': -1000, 'onDefense': 100, 'distanceToInvader': -10, 'stop': -100, 'reverse': -2, 'disDiffToCapsule': -1}