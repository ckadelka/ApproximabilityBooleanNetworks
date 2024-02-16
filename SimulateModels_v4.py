# Simulate both the bio models and the associated random ensembles, and store the simulation in a single file
# for each Bio model

import os
from boolionnet import boolionnet
import torch
import numpy as np
import time
import sys

try:
    filename = sys.argv[0]
    SLURM_ID = int(sys.argv[1])
except:
    filename = ''
    SLURM_ID = 1#random.randint(0,100)

# Analysis parameters
NumSamples = 2  # 1000 is the desired number, iterate here to get 2*50
NumIters = 50  # 500 is the desired number
NumRandomModels = 100  # 100 is the desired number
max_order_approx = 10

# Data source location
BioInputsFuncsFolder = "inputs_funcs_bio_models/"
BioInputsFuncsFiles = sorted(os.listdir(BioInputsFuncsFolder))
BioDerivativesFolder = 'derivatives_bio_models/'
RandomInputsFuncsFolder = "inputs_funcs_random_models/"
RandomDerivativesFolder = 'derivatives_random_models/'

for FileNumber in range(72,len(BioInputsFuncsFiles)):
    now = time.time()

    # Load network
    BioModelFile = BioInputsFuncsFiles[FileNumber]
    BioModel = np.load(BioInputsFuncsFolder + BioModelFile,allow_pickle=True)
    I = BioModel['arr_0']
    F = BioModel['arr_1']
    BioDerivativesFile = BioDerivativesFolder + BioModelFile.split('_Bio')[0] + '_Bio_derivatives.dat'
    bn = boolionnet(inputsList=I, functionsList=F, derivativesFile=BioDerivativesFile)
    NumNodes = bn.n

    # Analyze network
    indegrees = list(map(len, I))
    MaxOrder = min(max(indegrees),max_order_approx)#max(indegrees)  # max order of the derivatives to be used for analyzing this network
    initState = torch.tensor(np.random.randint(2, size=(NumSamples,NumNodes)))  # shape = (NumSamples,NumNodes)
    SimulationData = dict()

    # Simulate Bio model
    bn.set(initState)  # same initial state for the Bio and all the Random models
    bn.update(iters=NumIters,approxUpto=MaxOrder,saveIntermediateOrders=True)
    SimulationData['Bio'] = dict()
    SimulationData['Bio']['initial'] = initState  # shape = (NumSamples,NumNodes)
    SimulationData['Bio']['final'] = bn.currentStateAllOrders  # shape = (NumSamples,MaxOrder,NumNodes)
    
    # Simulate Random models associated with the above Bio model
    SimulationData['Random'] = dict()
    SimulationData['Random']['initial'] = torch.zeros(NumRandomModels,3,NumSamples,NumNodes)  # shape = (NumRandomModels,NumSamples,NumNodes)
    SimulationData['Random']['final'] = torch.zeros(NumRandomModels,3,NumSamples,MaxOrder,NumNodes)  # shape = (NumRandomModels,NumSamples,MaxOrder,NumNodes)
    
    
    for run in range(NumRandomModels):
        for null_model_id in range(1,4):
            RandomModelFile = RandomInputsFuncsFolder + BioModelFile.split('_Bio')[0] + '_Random_inputs_funcs_nullmodel'+str(null_model_id)+'_run'+ str(run) + '.npz'
            RandomModel = np.load(RandomModelFile, allow_pickle=True)
            I = RandomModel['arr_0']
            F = RandomModel['arr_1']
            RandomDerivativesFile = RandomDerivativesFolder + BioModelFile.split('_Bio')[0] + '_Random_derivatives_nullmodel'+str(null_model_id)+'_run'+ str(run) + '.dat'
            bn = boolionnet(inputsList=I, functionsList=F, derivativesFile=RandomDerivativesFile)
            bn.set(initState)  # same initial state for the Bio and all the Random models
            bn.update(iters=NumIters, approxUpto=MaxOrder, saveIntermediateOrders=True)
            SimulationData['Random']['initial'][run][null_model_id-1] = initState  # shape = (NumSamples,NumNodes)
            SimulationData['Random']['final'][run][null_model_id-1] = bn.currentStateAllOrders  # shape = (NumSamples,MaxOrder,NumNodes)

    # Save simulation data file
    SimulationFile = './Data/' + BioModelFile.split('_Bio')[0] + '_SimulationData_SLURMID'+str(SLURM_ID)+'.dat'
    torch.save(SimulationData,SimulationFile)
    print(FileNumber, BioInputsFuncsFiles[FileNumber],'simulated: ',round(time.time()-now,2),'seconds')

## To read the simulation file, run the following:
# filename = './Data/' + BioInputsFuncsFiles[FileNumber].split('_Bio')[0] + '_SimulationData.dat'
# SimulationData = torch.load(filename)