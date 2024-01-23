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

# your_baseline 1, 2, 3은 모두 baseline의 로직에 기반을 두고 있음

# baseline의 기본적인 action 선택 방식은 다음과 같음
# agent의 기본적인 뼈대가 되는 ReflexCaptureAgent -> OffensiveReflexAgent, defensiveReflexAgent를 정의
# agent는 게임의 모든 정보들을 볼 수 있음
# -> 가능한 action마다 successor를 구하고, successor들의 정보를 feature로 가져와 feature * weight의 linear sum을 evaluation value로 사용
# evaluation value가 가장 높은 action을 다음 action으로 선택.
# 결국 어떤 feature를 사용하느냐, 즉 getFeatures() 함수가 agent의 성능을 좌우함.

# 추가 또는 변경된 사항은 아래와 같음

# 1. offensive agent는 현재 점수가 충분하다면 방어자 모드로 전환 (offensive or defensive agent 클래스)
# 2. offensive agent와 고스트, 캡슐까지의 거리를 피쳐로 반영
# 3. offensive agent가 일정 수 이상의 food를 먹었다면, 자기 진영으로 돌아오도록 함
# 4. defensive agent가 캡슐 주변에서 방어하도록 함

# 해당사항 코드 위주로 주석을 달았습니다.

from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game

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
    self.isOffensive = None     # offensive agent가 공격을 할지 수비를 할지 결정하는 변수를 인스턴스로 추가함.
    
  def getFeatures(self, gameState, action):
    # 해당 action의 feature를 구하는데 필요한 successor와 state, position, 상대 agents를 구함
    successor = self.getSuccessor(gameState, action)
    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()
    opponents = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    features = util.Counter()   # feature들을 저장할 딕셔너리를 util.Counter()를 이용해 선언 -> 일반 딕셔너리보다 key, value 선언이나 linear sum 계산이 간편함

    # 현재 점수가 내가 지켜야 할 음식들의 1/2보다 적다면
    # 점수가 충분하지 않다는 뜻이므로 공격자 모드
    # 더 크다면 점수가 충분하므로 역전당하지 않도록 방어를 함
    if self.getScore(gameState) < len(self.getFoodYouAreDefending(successor).asList()) / 2:
      self.isOffensive = True
    else:
      self.isOffensive = False

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
      #
      ghosts = [i for i in opponents if not i.isPacman]
      ghostsPos = [ghost.getPosition() for ghost in ghosts]
      ghostsScaredTimer = [ghost.scaredTimer for ghost in ghosts]

      # 고스트들이 scared 상태라면 고스트까지의 거리는 충분한 상황과 비슷하므로 50의 값을 부여
      if ghostsScaredTimer and min(ghostsScaredTimer) > 5:
          features['distanceToGhost'] = 50
      else:
        distToGhost = [self.getMazeDistance(myPos, gp) for gp in ghostsPos] # 고스트까지의 실제 거리를 계산
        if distToGhost:
          minDistToGhost = min(distToGhost)   # 고스트가지의 최단 거리를 계산
          if minDistToGhost <= 5:
            features['distanceToGhost'] = -9999   # 고스트까지 거리가 5 이하로 가깝다면 최악의 상황이므로 매우 낮은 값 부여
          else:
            features['distanceToGhost'] = minDistToGhost    # 아니라면 거리가 멀 수록 좋은 것이므로 고스트까지의 최단 거리를 자체를 feature 값으로 부여.
      
      #
      # capsule까지의 최단 거리 계산
      # 
      capsules = self.getCapsules(successor)
      if capsules:         # 캡슐이 남아있다면
        distToCapsule = [self.getMazeDistance(myPos, capsule) for capsule in capsules]   
        minDistToCapsule = min(distToCapsule)   # 캡슐까지의 최단 거리를 계산
        features['distanceToCapsule'] = minDistToCapsule
        # 캡슐까지 최단 거리는 가까우면 좋은 것이므로, weight에 음수를 부여해 가까울수록 높은 값을 가지도록 함.
      
      #
      # 충분한 food를 먹었다면 myside에 deposite 하도록
      #
      freeEat = len([i for i in opponents if i.isPacman]) == 2  # 상대 pacman의 수가 2라면, 고스트 숫자는 0 -> 자유롭게 food를 먹을 수 있는 환경인지를 나타내는 freeEat변수 선언
      limit = len(foodList) if freeEat else len(foodList)/5     # 한도는 freeEat라면 없고, 아니라면 남은 food의 1/5로 함
      if myState.numCarrying > limit:
        homePos = gameState.getInitialAgentPosition(self.index)
        features['distanceToHome'] = self.getMazeDistance(myPos, homePos) # 현재 들고 있는 food 수가 한도보다 많다면 시작 위치로부터의 거리를 feature로 함. 
      else:                                                               # -> 가까울수록 좋은 것이므로 가중치를 음수로 부여해 가까울수록 높은 값을 가지도록 함.
        features['distanceToHome'] = 0

    # 방어자 모드일 경우 필요한 feature들을 계산
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

      #
      # capsule 지키기
      #
      capsules = self.getCapsulesYouAreDefending(gameState) # 지켜야 하는 캡슐을 구함
      targetCapsule = None
      minDistOppToCapsule = 9999
      for capsule in capsules:
        for opponent in opponents:
          distOppToCapsule = self.getMazeDistance(capsule, opponent.getPosition()) # 모든 (캡슐, 상대 agent) 쌍마다 거리 계산
          if distOppToCapsule < minDistOppToCapsule:
            targetCapsule = capsule
            minDistOppToCapsule = distOppToCapsule  # 캡슐과 상대 agent의 최단 거리, 그리고 타겟 캡슐을 구함
      if targetCapsule != None:
        distToTargetCapsule = self.getMazeDistance(myPos, targetCapsule)  # 타겟 캡슐과 자신의 거리를 계산
        disDiff = distOppToCapsule - distToTargetCapsule
        if distOppToCapsule > 5:
          if disDiff > 3:
            features['disDiffToCapsule'] = 100    # 만약 상대방과 캡슐간의 거리가 5보다 클 때, 내가 상대보다 캡슐까지 최소 3만큼 더 가깝도록 하면 100의 값 부여
          else:
            features['disDiffToCapsule'] = (-10) * distToTargetCapsule # 캡슐까지의 거리가 나와 상대방이 거의 비슷하거나 내가 더 멀면 거리에 비례해서 안좋아지도록 값 부여
        else:
          features['disDiffToCapsule'] = (-100) * distToTargetCapsule # 상대방이 캡슐까지 거리 5 이하로 가깝다면 나와 캡슐 사이의 거리가 매우 중요해지도록 값 부여

    return features

  def getWeights(self, gameState, action):
    offensive = {'successorScore': 1000, 'distanceToFood': -1, 'distanceToGhost': 1, 'distanceToCapsule': -10, 'distanceToHome': -10}
    defensive = {'numInvaders': -1000, 'onDefense': 100, 'distanceToInvader': -10, 'stop': -100, 'reverse': -2, 'disDiffToCapsule': 1}
    return offensive if self.isOffensive else defensive
    # 공격자, 방어자 모드 각각의 가중치들을 정의한 후 에이전트의 모드에 따라 알맞는 가중치 리턴

class DefensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """
  # 위에 있는 offensive or defensive agent의 defense mode랑 코드 같음

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
    # 바로 위에 있는 offensive or defensive agent 코드랑 똑같아서 주석은 따로 달지 않음.
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
      if distOppToCapsule > 5:
        if disDiff > 3:
          features['disDiffToCapsule'] = 100
        else:
          features['disDiffToCapsule'] = (-10) * distToTargetCapsule
      else:
        features['disDiffToCapsule'] = (-100) * distToTargetCapsule

    return features

  def getWeights(self, gameState, action):
    return {'numInvaders': -1000, 'onDefense': 100, 'distanceToInvader': -10, 'stop': -100, 'reverse': -2, 'disDiffToCapsule': 1}