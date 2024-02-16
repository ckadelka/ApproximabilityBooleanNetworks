# Implementation of the boolion network

import torch
from torch import tensor
from boolion import boolion
import numpy as np
from itertools import product

# Possible performance improvements:
# 1) Automatic updates of input combinations from just input updates
# 2) Batch implementation of update()

# This class provides methods for implementing a Boolean function using the Taylor series of the nodes and
# using the series to update the state of the network

class boolionnet():

    def __init__(self,inputsList=[],functionsList=[],derivativesFile=''):
        if inputsList == []:
            raise Exception("Please supply a list of inputs")
        else:
            self.inputsList = inputsList
        # if a derivatives file is supplied load it into a list of dictionaries
        if (derivativesFile != '') and (functionsList != []):
            self.derivativesList = torch.load(derivativesFile)
            self.functionsList = functionsList
            self.n = len(self.derivativesList)  # the number of nodes in the network
            # compute a list of boolion objects, one for each node
            self.boolionFuncs = [boolion() for node in range(self.n)]
            for node in range(self.n):
                self.boolionFuncs[node].loadDerivatives(self.derivativesList[node], self.functionsList[node])
                self.boolionFuncs[node].generateInputCombinations()
                outputs = self.functionsList[node]
                bias = outputs.sum() / len(outputs)
                self.boolionFuncs[node].OutputBias = bias
        elif (derivativesFile == '') and (functionsList != []):  # compute derivatives if a file is not supplied
            self.functionsList = functionsList
            self.n = len(self.functionsList)  # the number of nodes in the network
            # compute a list of boolion objects, one for each node
            self.boolionFuncs = [boolion() for node in range(self.n)]
            for node in range(self.n):
                NumInputs = len(self.inputsList[node])
                l = [[0, 1] for i in range(NumInputs)]
                fullLUT = np.array(list(product(*l)))
                indicesToOne = np.where(self.functionsList[node] == 1)[0]
                LUTtoOne = torch.tensor(fullLUT[indicesToOne, :])
                self.boolionFuncs[node].loadLUT(LUTtoOne)
                self.boolionFuncs[node].decomposeFunction()
                outputs = self.functionsList[node]
                bias = outputs.sum() / len(outputs)
                self.boolionFuncs[node].OutputBias = bias
        else:
            raise Exception("Please either supply a list of functions or a corresponding file containing the Taylor derivatives")
        # set a default current state
        self.NumSamples = 1
        self.currentState = torch.zeros(self.NumSamples,self.n)

    def set(self,state):
        if len(state.shape) == 1:
            self.NumSamples = 1
        elif len(state.shape) == 2:
            NumSamples = state.shape[0]
            NumNodes = state.shape[1]
            if (NumNodes!=self.n) or (NumSamples==0):
                raise Exception("Number of input states must be at least one and length of the state vector must equal the number of nodes")
            else:
                self.NumSamples = NumSamples
                self.currentState = state.view(self.NumSamples, self.n)

    # Updates the current state of the network using the Taylor expansions of the nodes.
    # If an approximation is used, then it's possible that the updated states spill outside of the interval [0,1],
    # in which case they are rounded to the nearest boundary.
    def update(self,iters=1,approxUpto='default',saveIntermediateOrders=False):
        if (approxUpto != 'default'):
            if approxUpto < 0:
                raise Exception('Invalid approximation order; valid values are either \'default\' or a positive integer')
        if saveIntermediateOrders:
            nextStateAllOrders = torch.ones(self.NumSamples,approxUpto,self.n)*-1
        for iter in range(iters):
            nextState = torch.FloatTensor([-99]*self.NumSamples*self.n).view(self.NumSamples,self.n)
            for node in range(self.n):
                inputVariables = self.inputsList[node]
                PrInputState = self.currentState[:,inputVariables]
                self.boolionFuncs[node].computeFunctionValueTaylor(PrInputState,approxUpto=approxUpto,saveIntermediateOrders=saveIntermediateOrders)
                nextState[:,node] = self.boolionFuncs[node].functionValue
                if saveIntermediateOrders and (iter==(iters-1)):
                    functionValueAllOrders = self.boolionFuncs[node].functionValueAllOrders
                    MaxOrder = functionValueAllOrders.shape[1]  # max order for a node = its number of inputs
                    nextStateAllOrders[:,0:MaxOrder,node] = functionValueAllOrders
                    if approxUpto > MaxOrder:  # function values for orders > MaxOrder is simply the exact value (equal to the value at MaxOrder)
                        nextStateAllOrders[:,MaxOrder:,node] = nextStateAllOrders[:,MaxOrder-1,node].view(self.NumSamples,1)
            self.currentState = nextState
            self.currentState[self.currentState < 0] = 0
            self.currentState[self.currentState > 1] = 1
        if saveIntermediateOrders:
            nextStateAllOrders[nextStateAllOrders < 0] = 0
            nextStateAllOrders[nextStateAllOrders > 1] = 1
            self.currentStateAllOrders = nextStateAllOrders  # shape = (NumSamples,approxUpto,n)

