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

# your_baseline2 는 your_baseline1 의 발전된 버전
# 추가 또는 변경된 사항은 아래와 같음

# 1. distanceToHome의 경우, 집까지의 거리 * 먹은 음식 수 를 피쳐로 변경
# 2. offensive agent가 우리 진영일 때 상대 공격자가 2명이 되면 방어자 모드로 전환하도록 함
# 3. offensive agent가 방어자 모드로 전환하도록 하는 점수 기준을 낮춤 
# (실제로 게임을 해보니 큰 점수차가 나는 경우가 드물었음)
# 4. 해당 action을 통해 캡슐을 먹을 수 있다면, 무조건 이 action을 취하도록 함
# 5. 캡슐 주변을 지키는 피쳐를 구체화함

# 해당하는 부분 위주로 주석을 달았습니다

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
    self.isOffensive = None

  def getFeatures(self, gameState, action):
    curState = gameState.getAgentState(self.index)
    successor = self.getSuccessor(gameState, action)
    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()
    opponents = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    features = util.Counter()

    # 우리 진영에 남은 food의 1/5만 넘어도 방어자 모드로 전환하도록 함
    if self.getScore(gameState) > len(self.getFoodYouAreDefending(successor).asList()) / 5:
      self.isOffensive = False
    elif not curState.isPacman and len([i for i in opponents if i.isPacman]) == 2:
      self.isOffensive = False    # agent가 우리 진영이고, 상대 팩맨 수가 2명이면 일단 방어를 하도록 함
    else:
      self.isOffensive = True     # 다른 경우는 공격자 모드

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

      if ghostsScaredTimer and min(ghostsScaredTimer) > 5:
          features['distanceToGhost'] = 50
      else:
        distToGhost = [self.getMazeDistance(myPos, gp) for gp in ghostsPos]
        if distToGhost:
          minDistToGhost = min(distToGhost)
          if minDistToGhost <= 5:
            features['distanceToGhost'] = -9999
          else:
            features['distanceToGhost'] = minDistToGhost
        
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
      # limit을 설정하는 기존 방식은 현재 먹어둔 음식 수가 중요하지 않았음
      # home까지의 거리에다가 numCarrying을 곱한 것을 feature로 변경
      freeEat = len([i for i in opponents if i.isPacman]) == 2
      homePos = gameState.getInitialAgentPosition(self.index)
      if freeEat:
        features['distanceToHome'] = self.getMazeDistance(myPos, homePos) # free eat 상황이면 거리만 중요함 -> 너무 멀리 나가진 않도록
      else:
        features['distanceToHome'] = self.getMazeDistance(myPos, homePos) * myState.numCarrying
        # 거리 * 먹은 음식 수 -> 더 많이 먹을수록 다시 돌아오려는 성향을 가지도록 함


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
      # 피쳐를 보다 구체화 하였음
      capsules = self.getCapsulesYouAreDefending(gameState)  # 지켜야 하는 캡슐을 구함
      targetCapsule = None
      minDistOppToCapsule = 9999
      for capsule in capsules:
        for opponent in opponents:
          distOppToCapsule = self.getMazeDistance(capsule, opponent.getPosition())  # 모든 (캡슐, 상대 agent) 쌍마다 거리 계산
          if distOppToCapsule < minDistOppToCapsule:
            targetCapsule = capsule
            minDistOppToCapsule = distOppToCapsule  # 캡슐과 상대 agent의 최단 거리, 그리고 타겟 캡슐을 구함

      if targetCapsule != None:
        distToTargetCapsule = self.getMazeDistance(myPos, targetCapsule)  # 타겟 캡슐과 자신의 거리를 계산
        if distOppToCapsule > 10:       
          features['disDiffToCapsule'] = distToTargetCapsule  # 상대방이 캡슐과 거리 10 이상으로 충분히 먼 상황이라면 단순히 나와 캡슐까지의 거리를 피쳐값으로 함
        elif distOppToCapsule > 5:
          if distToTargetCapsule <  distOppToCapsule - 3:
            features['disDiffToCapsule'] = 10 * distToTargetCapsule  # 상대와 캡슐의 거리가 10~6 사이이고, 내가 어느정도 더 가까울 때는 가중치 10 부여 -> 거리가 조금 더 중요해지도록
          elif distToTargetCapsule > distOppToCapsule:
            features['disDiffToCapsule'] = 999    # 상대가 더 가까우면 최악의 상황으로 가정
          else:
            features['disDiffToCapsule'] = 100 * distToTargetCapsule  # 내가 더 가깝긴 하지만 얼마 차이가 안 날때 -> 가중치 100 부여 -> 더더욱 중요하도록
        else:   # 상대와 캡슐의 거리가 5 이하로 매우 가깝다면
          if distToTargetCapsule > distOppToCapsule:  # 상대가 더 가깝다면 매우 최악의 상황으로 가정
            features['disDiffToCapsule'] = 9999
          else:
            features['disDiffToCapsule'] = 100 * (2 ** distToTargetCapsule) # 내가 가까워도 100의 가중치 및 거리의 제곱을 피쳐값으로 함 -> 거리가 매우매우 중요함
            # 가중치는 음수 -> 거리가 가까우면 좋은 evaluation

    return features

  def getWeights(self, gameState, action):
    offensive = {'successorScore': 1000, 'distanceToFood': -1, 'distanceToGhost': 1, 'eatCapsule': 9999, 'distanceToCapsule': -10, 'distanceToHome': -10}
    defensive = {'numInvaders': -1000, 'onDefense': 100, 'distanceToInvader': -10, 'stop': -100, 'reverse': -2, 'disDiffToCapsule': -1}
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