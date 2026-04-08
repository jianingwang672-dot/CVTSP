import os
import re
import time
import math
import numpy as np
import matplotlib.pyplot as plt

from gurobipy import *
from collections import namedtuple


Node = namedtuple('Node', ['index', 'x', 'y', 'r'])


def gurobi_socp(loc, radius, threads=None, time_limit=None, test=False):

    m = Model("CETSP-2D")
    m.setParam('OutputFlag', 0)
    if time_limit:
        m.setParam('TimeLimit', time_limit)
    m.setParam('threads', threads)


    n = loc.shape[0]
    NN = []
    for i in range(0, n):
        NN.append(Node(i, float(loc[i, 0]), float(loc[i, 1]), float(radius[i])))

    # variable (head of cone)
    length = {}
    for i in NN:
        length[i.index] = m.addVar(lb=0, ub=GRB.INFINITY, name="length" + str(i.index))

    # add variables corresponding to coordinates x, y and z
    x = {}
    for i in NN:
        x[i.index] = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="x" + str(i.index))
    y = {}
    for i in NN:
        y[i.index] = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="y" + str(i.index))

    # add auxiliary variables w, u, v, s, t and q
    w = {}
    for i in NN:
        w[i.index] = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="w" + str(i.index))
    u = {}
    for i in NN:
        u[i.index] = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="u" + str(i.index))
    s = {}
    for i in NN:
        s[i.index] = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="s" + str(i.index))
    t = {}
    for i in NN:
        t[i.index] = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="t" + str(i.index))

    # set objective function
    m.update()
    obj = quicksum(length[i.index] for i in NN)
    m.setObjective(obj)

    # set SOC constraints
    for i in NN:
        m.addConstr(-length[i.index]*length[i.index] + w[i.index]*w[i.index] + u[i.index]*u[i.index] <= 0)
    for i in NN:
        m.addConstr(-i.r*i.r + s[i.index] * s[i.index] + t[i.index] * t[i.index] <= 0)

    # radius constraints
    for i in NN:
        m.addConstr(-i.x + s[i.index] + x[i.index] == 0)
    for i in NN:
        m.addConstr(-i.y + t[i.index] + y[i.index] == 0)

    # separating first constraint
    m.addConstr(w[0] - x[n-1] + x[0] == 0)
    for i in NN[1:]:
        m.addConstr(w[i.index] - x[i.index-1] + x[i.index] == 0)
    m.addConstr(u[0] - y[n - 1] + y[0] == 0)
    for i in NN[1:]:
        m.addConstr(u[i.index] - y[i.index - 1] + y[i.index] == 0)

    m.optimize()
    # print(2)
    # print(m.objVal)
    if test:
        x = m.getAttr('x', x)
        y = m.getAttr('x', y)
        return m.objVal, x, y

    return m.objVal
