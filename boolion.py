# Implementation of the boolion function

import torch
from itertools import product
from itertools import combinations as comb

# Possible performance improvements:
# 1) Rather than treat output bias separately make it a 0th order term in the derivatives dictionary
# 2) Batch implementation of computeFunctionValueTaylor() (try to eliminate 'for' loops even for one input sample)
#    and computeFunctionValueProbabilistic()

# This class provides methods for decomposing a Boolean function using the Taylor series and
# using the series to compute the value of the function for a given input that can assume any value in [0,1]^n

class boolion():

    def __init__(self):
        pass

    # Loads LUT
    def loadLUT(self,LUTtoOne=torch.tensor([])):
        # if len(LUTtoOne) == 0:
        #     raise Exception('LUT cannot be empty')
        # else:
        self.LUTtoOne = LUTtoOne  # the subset of LUT rows that go to 1
        self.NumInputs = LUTtoOne.shape[1]  # number of columns of the LUT
        self.zerosx, self.zerosy = torch.where(LUTtoOne == 0)  # the indices of the LUT rows containing a 0
        self.OutputBias = LUTtoOne.shape[0] / pow(2,self.NumInputs)

    # Loads externally computed derivatives
    def loadDerivatives(self,derivatives=dict(),outputs=[]):
        if (derivatives == {}) or (outputs == []):
            raise Exception('Derivatives dictionary and outputs list cannot be empty')
        else:
            self.derivatives = derivatives  # must be a dictionary
            #self.NumInputs = len(derivatives) #fixed by Claus Kadelka, 2023/10/27 - wrong if not all derivatives are computed because only a lower order approximation is desired
            self.NumInputs = len(self.derivatives[1])

    # finite difference method for computing the differential of f with respect to the variables
    def computeDerivative(self,variables):
        order = len(variables)
        NumSamples = 2**order
        PrInputs = torch.ones(NumSamples,1,self.NumInputs) * 0.5
        l = [[0, 1] for i in range(order)]
        finiteDiffEvalPoints = torch.tensor(list(product(*l)),dtype=torch.float32)  # the point-pairs at which the function values in the finite difference are computed
        PrInputs[:,0,variables] = finiteDiffEvalPoints
        functionValues = self.computeFunctionValueProbabilistic(PrInputs)  # the function-value-pairs comprising the finite difference
        for ord in torch.arange(order,0,-1):
            finiteDifferences = torch.zeros(len(self.PartitionIndices[ord-1]))
            for i in range(len(self.PartitionIndices[ord-1])):
                indexPair = self.PartitionIndices[ord-1][i]
                finiteDifference = functionValues[indexPair[1]] - functionValues[indexPair[0]]
                finiteDifferences[i] = finiteDifference
            functionValues = finiteDifferences.detach()
        derivative = torch.tensor([functionValues[0]])
        return(derivative)

    # Creates all possible input combinations of orders 1 through NumInputs
    def generateInputCombinations(self):
        inputVariablesList = list(range(self.NumInputs))
        self.inputIndexCombinations = dict()
        for order in range(1, self.NumInputs + 1):
            self.inputIndexCombinations[order] = torch.tensor(list(comb(inputVariablesList, order)))

    # Computes the Taylor decomposition
    def decomposeFunction(self,upto=-1):
        if upto==-1:
            upto=self.NumInputs
        else:
            upto = min(self.NumInputs,upto)
        self.generateInputCombinations()
        # generate the indices at which finite differences will be computed
        self.PartitionIndices = []
        for k in torch.arange(1,upto+1,1):  # the indices of the point-pairs at which the function values in the finite difference are computed
            indices = torch.arange(2**k)
            self.PartitionIndices.append(torch.split(indices,2))
        self.derivatives = dict()
        # Computes all the derivatives of f w.r.t all combinations of input variables up to order n
        for order in range(1,upto+1):
            self.derivatives[order] = torch.tensor([])
            for variables in self.inputIndexCombinations[order]:
                derivative = self.computeDerivative(variables)
                self.derivatives[order] = torch.cat((self.derivatives[order], derivative))

    # Computes the Boolean function value using the Taylor expansion.
    # Puts together the Taylor terms (product of input offsets from the hypercube center and the derivatives) and
    # sums them up to compute the final output of the function.
    # This method accepts batch inputs with shape (NumSamples,NumInputs).
    def computeFunctionValueTaylor(self, inputs, approxUpto='default', saveIntermediateOrders=False):
        if len(inputs.shape) == 1:
            inputs = inputs.view(1,-1)
            NumSamples = 1
        if len(inputs.shape) > 1:
            NumSamples = inputs.shape[0]
        elif len(inputs.shape) == 0:
            raise Exception("Length of input vector should be > 1")
        self.inputVariableValues = inputs - 0.5
        self.inputVariableOffsets = dict()  # all possible combinations of input offsets from the center of the unit hypercube
        if (approxUpto == 'default') or (approxUpto >= self.NumInputs):
            MaxOrder = self.NumInputs
        elif (approxUpto >= 1):
            MaxOrder = approxUpto
        else:
            raise Exception("ApproxUpto must be a +ve integer")
        for order in range(1, MaxOrder + 1):
            self.inputVariableOffsets[order] = torch.prod(self.inputVariableValues[:, self.inputIndexCombinations[order]], 2)
        self.functionValue = self.OutputBias  # 0th order Taylor term
        if saveIntermediateOrders:  # save functionValues for all intermediate orders through approxUpto
            self.functionValueAllOrders = torch.ones(NumSamples,MaxOrder)*-1  # shape = (NumSamples,MaxOrder)
        for order in range(1, MaxOrder + 1):  # higher order Taylor terms
            self.functionValue += (self.derivatives[order] * self.inputVariableOffsets[order]).sum(1)  # shape = (NumSamples)
            if saveIntermediateOrders:
                self.functionValueAllOrders[:,order-1] = self.functionValue

    # Computes the Boolean function value using the full (unapproximated) probabilistic method.
    # This method works with batch inputs (NumSamples >= 1).
    def computeFunctionValueProbabilistic(self, PrInputs):  # required shape of PrInputs = (NumSamples,1,NumInputs)
        # Compute the probabilistic LUT: replace 1s and 0s in the original LUT
        # with Pr_x or Pr_y (depending on the column) and 1-Pr_x or 1-Pr_y (depending on the column) respectively
        # NOTE: the PrLUTtoOne defined below is local, whereas self.PrLUTtoOne is the one used by the Taylor calculations,
        # which is defined with a fixed PrInputs representing uniform distribution of inputs.
        PrLUTtoOne = torch.FloatTensor(PrInputs).repeat((1, self.LUTtoOne.shape[0], 1))
        PrLUTtoOne[:, self.zerosx, self.zerosy] = 1 - PrLUTtoOne[:, self.zerosx, self.zerosy]
        # Compute the net probability that the output is 1, by multiplying the probabilities in each row
        # and summing them up
        PrOne = torch.prod(PrLUTtoOne, 2).sum(1)  # value of Pr[f(x,y)=1]
        return(PrOne)  # shape = (NumSamples)
