# mlLearningAgents.py
# parsons/27-mar-2017
#
# A stub for a reinforcement learning agent to work with the Pacman
# piece of the Berkeley AI project:
#
# http://ai.berkeley.edu/reinforcement.html
#
# As required by the licensing agreement for the PacMan AI we have:
#
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

# The agent here was written by Simon Parsons, based on the code in
# pacmanAgents.py
# learningAgents.py

from pacman import Directions
from game import Agent
import random
import game
import util


# QLearnAgent
#
class QLearnAgent(Agent):

    # Constructor, called when we start running the
    def __init__(self, alpha=0.1, epsilon=0.08, gamma=0.7, numTraining=10):
        # alpha       - learning rate
        # epsilon     - exploration rate
        # gamma       - discount factor
        # numTraining - number of training episodes
        #
        # These values are either passed from the command line or are
        # set to the default values above. We need to create and set
        # variables for them
        #
        self.alpha = float(alpha)
        # @Param alpha value
        # @Return floating point alpha value

        self.epsilon = float(epsilon)
        # @Param epsilon value
        # @Return floating point epsilon value

        self.gamma = float(gamma)
        # @Param gamma value
        # @Return floating point gamma value

        self.numTraining = int(numTraining)
        # @Param training runs
        # @Return training number in integer format

        self.episodesSoFar = 0
        # @Return number of training episodes

        self.current_score = 0
        # @Return current score value

        self.prior_state = None
        # @Return prior state

        self.prior_action = None
        # @Return prior action

        self.qVal = util.Counter()
        # @Return Q Value

    # Accessor functions for the variable episodesSoFars controlling learning
    # @return number of episodes
    def incrementEpisodesSoFar(self):
        self.episodesSoFar += 1

    def getEpisodesSoFar(self):
        # @Return the episodes undergone so far
        return self.episodesSoFar

    def getNumTraining(self):
        return self.numTraining

    # Accessor functions for parameters
    def setEpsilon(self, value):
        # @Return epsilon value
        self.epsilon = value

    def setAlpha(self, value):
        # @Return alpha value
        self.alpha = value
        return self.alpha

    def getGamma(self):
        # @Return gamma value
        return self.gamma

    def getMaxAttempts(self):
        return self.maxAttempts

    # The function will be used to acquire the initial Q values of the
    # state and action.

    def QVal(self, state, action):
        # @param state and action
        # @return state and action of Q_value
        return self.qVal[(state, action)]

    # The function is used to acquire the action that will generate
    # the maximum Q value.  Our goal is to maximise the value of the
    # Q function.

    def QValMax(self, state, legal):
        # @param state and legal
        # @return max value of the Q_state
        # Appends value into empty list, max value of list is returned
        q_state = []
        for action in legal:
            q_amount = self.QVal(state, action)
            q_state.append(q_amount)
        return max(q_state)

    # The Q-learning update rule is vital to Q-learning, it is employed
    # every time an action is executed in the state and moves the agent
    # to s'.  The Q-learning update rule is also known as the bellman
    # equation.

    def QValUpdate(self, state, action, reward, q_val_max):
        # @param state, action, reward, Max_Q_Value
        # @return state, action
        # Q-learning update rule applied every time an action is
        # executed in a state.
        self.qVal[(state, action)] = self.QVal(state, action) + self.alpha * \
                                       (reward + self.gamma * q_val_max -
                                        self.QVal(state, action))

    # getAction
    #
    # The main method required by the game. Called every time that
    # PAC-MAN is expected to move
    #
    # Epsilon greedy implemented
    # Used to encourage PAC-MAN to explore rather than exploit the best
    # path.
    # Ensures that PAC-MAN doesn't get stuck
    #
    # Checks if a random value generated is greater than epsilon
    # Returns a random action if true.
    # If false it will return the legal action that generates the
    # largest Q value.
    #
    # The action according to the if statement is returned.

    def getAction(self, state):
        # @Param state.getScore
        # @Param current score
        # @Param random value
        # @Param epsilon value
        # @Param legal actions
        # @Return Max Q Value action
        # @Return best action to take

        # Reward Calculation
        reward = state.getScore() - self.current_score

        # The data we have about the state of the game
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)
        print "Legal moves: ", legal
        print "Pacman position: ", state.getPacmanPosition()
        print "Ghost positions:", state.getGhostPositions()
        print "Food locations: "
        print state.getFood()
        print "Score: ", state.getScore()

        # Max Q Value calculation
        q_val_max = self.QValMax(state, legal)

        # Pick either random action or action that generates the
        # largest Q Value.
        if random.random() < self.epsilon:
            action = random.choice(legal)
        else:
            q_value_action = util.Counter()
            for action in legal:
                q_value_action[action] = self.QVal(state, action)
            action = q_value_action.argMax()

            self.QValUpdate(self.prior_state, self.prior_action, reward,
                            q_val_max)

        # Update the states and current score
        self.prior_state = state
        self.prior_action = action
        self.current_score = state.getScore()

        # Epsilon is decreased overtime as we do not want the PAC-MAN to always
        # explore.
        self.epsilon = self.epsilon * 0.99

        # Return the ideal action for PAC-MAN to take.
        return action

    # Handle the end of episodes
    #
    # This is called by the game after a win or a loss.
    def final(self, state):

        # Reward value is calculated
        reward = state.getScore() - self.current_score

        print "A game just ended!"

        # Keep track of the number of games played, and set learning
        # parameters to zero when we are done with the pre-set number
        # of training episodes
        self.incrementEpisodesSoFar()

        # Q-learning update rule applied
        self.QValUpdate(self.prior_state, self.prior_action, reward, 0)

        # Epsilon value slowly decreases overtime
        self.epsilon = self.epsilon * 0.99

        if self.getEpisodesSoFar() == self.getNumTraining():
            msg = 'Training Done test1(turning off epsilon and alpha)'
            print '%s\n%s' % (msg, '-' * len(msg))
            self.setAlpha(0)
            self.setEpsilon(0)

# References
# https://www.freecodecamp.org/news/an-introduction-to-q-learning-reinforcement-
# learning-14ac0b4493cc/
# https://keats.kcl.ac.uk/pluginfile.php/6784507/mod_resource/content/11/rl2.pdf



