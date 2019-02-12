import numpy as np
import sys
from gym.envs.toy_text import discrete
from graphicsLib import GridDisp
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

class GridworldEnv(discrete.DiscreteEnv):
    """
    Grid World environment from Sutton's Reinforcement Learning book chapter 4.
    You are an agent on an MxN grid and your goal is to reach the terminal
    state at the top left or the bottom right corner.
    For example, a 4x4 grid looks as follows:
    T  o  o  o
    o  x  o  o
    o  o  o  o
    o  o  o  T
    x is your position and T are the two terminal states.
    You can take actions in each direction (UP=0, RIGHT=1, DOWN=2, LEFT=3).
    Actions going off the edge leave you in your current state.
    You receive a reward of -1 at each step until you reach a terminal state.
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, shape=[4,4], obstacles = [], goals=[(0,0)], statePen=-0.1, transProb=1,  title = "std"):
        if not isinstance(shape, (list, tuple)) or not len(shape) == 2:
            raise ValueError('shape argument must be a list/tuple of length 2')

        self.shape = shape
        self.obstacles = obstacles
        self.goals = goals
        
        nS = np.prod(shape)
        nA = 4

        MAX_Y = shape[0]
        MAX_X = shape[1]
        P = {}
        grid = np.arange(nS).reshape(shape)
        it = np.nditer(grid, flags=['multi_index'])
        self.states = []
        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index

            P[s] = {a : [] for a in range(nA)}

            is_done = lambda t: (t%self.shape[0], t//self.shape[0]) in self.goals
            reward = 0.0 if is_done(s) else statePen

            # We're stuck in a terminal state
            if is_done(s):
                P[s][UP] = [(transProb, s, reward, True)]
                P[s][RIGHT] = [(transProb, s, reward, True)]
                P[s][DOWN] = [(transProb, s, reward, True)]
                P[s][LEFT] = [(transProb, s, reward, True)]
            # Not a terminal state
            else:
                ns_up = s if (y == 0 or (x,y-1) in obstacles) else s - MAX_X
                ns_right = s if (x == (MAX_X - 1) or (x+1,y) in obstacles) else s + 1
                ns_down = s if (y == (MAX_Y - 1) or (x,y+1) in obstacles)else s + MAX_X
                ns_left = s if (x == 0 or (x-1,y) in obstacles) else s - 1
                P[s][UP] = [(transProb, ns_up, reward, is_done(ns_up))]
                P[s][RIGHT] = [(transProb, ns_right, reward, is_done(ns_right))]
                P[s][DOWN] = [(transProb, ns_down, reward, is_done(ns_down))]
                P[s][LEFT] = [(transProb, ns_left, reward, is_done(ns_left))]
            if (x,y) not in obstacles:
                self.states.append(s)
            it.iternext()

        # Initial state distribution is uniform
        isd = np.zeros(nS)
        isd[nS-1]=1

        # We expose the model of the environment for educational purposes
        # This should not be used in any model-free learning algorithm
        self.P = P
        self.GD = GridDisp(shape,title)
        super(GridworldEnv, self).__init__(nS, nA, P, isd)

    def _render(self, mode='human', close=False, values = [], valIt = False):
        if close:
            return

        outfile = StringIO() if mode == 'ansi' else sys.stdout

        grid = np.arange(self.nS).reshape(self.shape)
        it = np.nditer(grid, flags=['multi_index'])
        vals = {}
        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index
            if self.s == s:
                output = " x "
            elif s == 0 or s == self.nS - 1:
                output = " T "
            else:
                output = " o "

            if x == 0:
                output = output.lstrip()
            if x == self.shape[1] - 1:
                output = output.rstrip()

            #outfile.write(output)

            # if x == self.shape[1] - 1:
            #     outfile.write("\n")
            #print(values)
            if values!=[]:
                vals[(x,y)] = values[y,x]
            it.iternext()
        if(valIt):
            self.GD.updateGrid(self.shape, self.obstacles, self.goals, (-1,-1), vals)
        else:
            self.GD.updateGrid(self.shape, self.obstacles, self.goals, (self.s%self.shape[0], self.s//self.shape[0]), vals)
        
    def stepQ(self, action):
        a = action
        #print(a)
        #You can take actions in each direction (UP=0, RIGHT=1, DOWN=2, LEFT=3).
        hindered = False
        obstacles = self.obstacles
        x,y = self.s%self.shape[0], self.s//self.shape[0]
        if (a==0 and (x,y-1) in obstacles):
            hindered = True
        if (a==1 and (x+1,y) in obstacles): 
            hindered = True
        if (a==2 and  (x,y+1) in obstacles):
            hindered = True
        if (a==3 and (x-1,y) in obstacles): 
            hindered = True
        if not hindered:
           return self.step(action)
        else:
            return (self.s, -10.0, False, "bumdep")
    
    def close(self):
        self.GD.destroy()
