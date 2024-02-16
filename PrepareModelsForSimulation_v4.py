# Generate random ensembles for every Bio model and generate the _inputs_funcs_ files for both

import sys
sys.path.append('../')
import numpy as np
import torch
from itertools import product
from boolion import boolion
import time

import canalizing_function_toolbox_v13 as can
import itertools
import load_database13 as db

models_to_keep = ['T-Cell Signaling 2006_16464248.txt',
                  '27765040_tabular.txt',
                  'ErbB (1-4) Receptor Signaling_23637902.txt',
                  'HCC1954 Breast Cell Line Long-term ErbB Network_24970389.txt',
                  'T-LGL Survival Network 2011_22102804.txt',
                  'Predicting Variabilities in Cardiac Gene_26207376.txt',
                  'Lymphopoiesis Regulatory Network_26408858.txt',
                  'Lac Operon_21563979.txt',
                  'MAPK Cancer Cell Fate Network_24250280.txt',
                  'Septation Initiation Network_26244885.txt',
                  '29632237.txt',
                  '25063553_OR_OR.txt',
                  '19622164_TGF_beta1.txt',
                  '23658556_model_10.txt',
                  '23169817_high_dna_damage.txt',
                  '28426669_ARF10_greater_ARF5.txt',
                  '21450717_model_5_2.txt',
                  'Guard Cell Abscisic Acid Signaling_16968132.txt',
                  'FGF pathway of Drosophila Signaling Pathways_23868318.txt',
                  'Death Receptor Signaling_20221256.txt'
                  ]

models_to_exclude_manually_because_similar_from_same_PID = ['Trichostrongylus retortaeformis_22253585.txt',
                                                            'Bordetella bronchiseptica_22253585.txt']

try:
    filename = sys.argv[0]
    SLURM_ID = int(sys.argv[1])
except:
    filename = ''
    SLURM_ID = 1#random.randint(0,100)

def load_models_included_in_meta_analysis(max_degree=12,max_N=1000,similarity_threshold=0.9,folders=['../update_rules_cell_collective/', '../update_rules_models_in_literature_we_randomly_come_across/'],models_to_keep=[],models_to_exclude_manually_because_similar_from_same_PID=[]):
## load the database, choose low max_n for quick results and to only look at small models
    [Fs,Is,degrees,degrees_essential,variabless,constantss,models_loaded,models_not_loaded] = db.load_database(folders,max_degree=max_degree,max_N=max_N)
    #similar_sets_jaccard = db.exclude_similar_models(Fs,Is,degrees,degrees_essential,variabless,constantss,models_loaded,similarity_threshold=similarity_threshold,USE_JACCARD = True,models_to_keep=models_to_keep)[-1]
    #similar_sets_overlap = db.exclude_similar_models(Fs,Is,degrees,degrees_essential,variabless,constantss,models_loaded,similarity_threshold=0.9,USE_JACCARD = False,models_to_keep=models_to_keep)[-1]
    #Fs,Is,degrees,degrees_essential,variabless,constantss,models_loaded,models_excluded,similar_sets_jaccard = db.exclude_similar_models(Fs,Is,degrees,degrees_essential,variabless,constantss,models_loaded,similarity_threshold=similarity_threshold,USE_JACCARD = True,models_to_keep=models_to_keep)
    Fs,Is,degrees,degrees_essential,variabless,constantss,models_loaded,models_excluded,similar_sets = db.exclude_similar_models(Fs,Is,degrees,degrees_essential,variabless,constantss,models_loaded,similarity_threshold=similarity_threshold,USE_JACCARD = False,models_to_keep=models_to_keep,models_to_exclude_manually_because_similar_from_same_PID=models_to_exclude_manually_because_similar_from_same_PID)
    n_variables = np.array(list(map(len,variabless)))
    n_constants = np.array(list(map(len,constantss)))
    return Fs,Is,degrees,degrees_essential,variabless,constantss,models_loaded,models_excluded,models_not_loaded,similar_sets,n_variables,n_constants,max_degree

NumRandomModels = 2  # 100 is the desired number
preserveBias = True
preserveCanalizingDepth = True
max_degree = 10 # load only rules that have at most this many regulators
max_order_approx = 10
#rulesFolder = "rules_bio_models/"
#rulesFiles = sorted(os.listdir(rulesFolder))

if preserveCanalizingDepth:
    left_side_of_truth_table = [np.array([[0],[1]])]
    left_side_of_truth_table.extend([np.array(list(itertools.product([0, 1], repeat = nn))) for nn in range(2, max_degree+1)])


Fs,Is,degrees,degrees_essential,variabless,constantss,models_loaded,models_excluded,models_not_loaded,similar_sets_of_models,n_variables,n_constants,max_degree = load_models_included_in_meta_analysis(max_degree=max_degree,models_to_keep=models_to_keep,models_to_exclude_manually_because_similar_from_same_PID=models_to_exclude_manually_because_similar_from_same_PID)
N = len(models_loaded)

Fs_essential = []
Is_essential = []
for i in range(N):
    dummy = can.get_essential_network(Fs[i],Is[i])
    Fs_essential.append(dummy[0])
    Is_essential.append(dummy[1])

models_analyzed = []
indices_models_analyzed = []
for FileNumber in range(N):
    now = time.time()
    F,I,deg = Fs_essential[FileNumber],Is_essential[FileNumber],list(map(len,Is_essential[FileNumber]))
    n_variables = len(variabless[FileNumber])
    if max(deg)>max_degree:
        print(FileNumber, models_loaded[FileNumber],'max degree (%i) too high' % (max(deg)))
        continue
    if min(deg)==0:
        print(FileNumber, models_loaded[FileNumber],'min degree == 0, issue with model')
        continue
    if SLURM_ID==0:
        models_analyzed.append( models_loaded[FileNumber])
        indices_models_analyzed.append(FileNumber)
        bioInputsFuncsFolder = 'inputs_funcs_bio_models/'
        bioInputsFuncsFile = models_loaded[FileNumber].split('.txt')[0] + '_Bio_inputs_funcs.npz'
        np.savez(bioInputsFuncsFolder + bioInputsFuncsFile, I, F)  # saves the inputs and the new rules
        derivativesList = []
        for i in range(len(F)):
            numInputs = int(np.log2(len(F[i])))  # this is the number of inputs for each output F[i]
            myarray = [[0, 1] for i in range(numInputs)]
            fullLUT = np.array(list(product(*myarray)))  # get the truthtable input
            allIndices = list(range(fullLUT.shape[0]))
            outputArray = np.array(F[i])
            indicesToOne = np.where(outputArray == 1)[0]
            LUTtoOne = torch.tensor(fullLUT[indicesToOne, :])
            bool = boolion()  # creates the boolion object
            bool.loadLUT(LUTtoOne)
            bool.decomposeFunction(max_order_approx)  # computes the output for the given input
            derivatives = bool.derivatives
            derivativesList.append(derivatives)
        bioderivativesFolder = 'derivatives_bio_models/'
        biodderivativesFile = models_loaded[FileNumber].split('.txt')[0]+ '_Bio_derivatives.dat' #name of file
        torch.save(derivativesList,bioderivativesFolder + biodderivativesFile)
        
    for run in range(SLURM_ID*NumRandomModels,(SLURM_ID+1)*NumRandomModels):
        for null_model_id in range(1,4):
            preserveBias = int(null_model_id % 2)
            preserveCanalizingDepth = int(null_model_id//2)
            # generate new rules
            randRules = []  # new rules list
            for i in range(len(F)):
                if i>=n_variables: #constants don't change
                    randRules.append(np.array([0,1]))
                    continue
                n_states = len(F[i])
                if preserveBias:
                    if preserveCanalizingDepth:
                        depth,n_layers,can_inputs,can_outputs,core_polynomial,can_order = can.find_layers(F[i])
                        can_inputs = np.random.choice(2,depth,replace=True)
                        can_order = np.random.choice(deg[i],depth,replace=False)
                        assert len(core_polynomial)!=2,"core polynomial cannot depend on a single variable"
                        if len(core_polynomial)==1:
                            core_function = np.array([1 - can_outputs[-1]],dtype=int)
                        elif len(core_polynomial)==4:
                            core_function = [np.array([0,1,1,0],dtype=int),np.array([1,0,0,1],dtype=int)][np.random.random()>0.5]
                        else:
                            while True:
                                oneIndices = np.random.choice(len(core_polynomial),sum(core_polynomial),replace=False)
                                core_function = np.zeros(len(core_polynomial),dtype=int)
                                core_function[oneIndices] = 1
                                if not can.is_canalizing(core_function):
                                    if not can.is_degenerated(core_function):
                                        break
                        newRule = -np.ones(n_states,dtype=int)
                        for j in range(depth):
                            newRule[np.where(np.bitwise_and(newRule==-1,left_side_of_truth_table[deg[i]-1][:,can_order[j]]==can_inputs[j]))[0]] = can_outputs[j]
                        newRule[np.where(newRule==-1)[0]] = core_function
                    else:
                        numones = sum(F[i])
                        oneIndices = np.random.choice(n_states,numones,replace=False)
                        newRule = np.zeros(n_states,dtype=int)
                        newRule[oneIndices] = 1
                else:
                    if preserveCanalizingDepth:
                        depth = can.find_layers(F[i])[0]
                        newRule = can.random_k_canalizing(n=deg[i],k=depth,EXACT_DEPTH_K=True,left_side_of_truth_table=left_side_of_truth_table[deg[i]-1])
                    else:
                        is_all_zeros = True
                        is_all_ones = True
                        while (is_all_zeros == True) or (is_all_ones == True):  # don't allow all ones or all zeros in new rules
                            newRule = np.random.choice(2,n_states,replace=True)
                            is_all_zeros = sum(newRule)==0
                            is_all_ones = sum(newRule)==n_states
                randRules.append(newRule)
            #calculate new rule derivatives
            randderivativesFolder = 'derivatives_random_models/'
            randderivativesFile = models_loaded[FileNumber].split('.txt')[0]+ '_Random_derivatives_nullmodel'+str(null_model_id)+'_run'+ str(run)+'.dat' #name of file
            derivativesList = []
            for i in range(len(randRules)):
                fullLUT = left_side_of_truth_table[deg[i]-1]
                allIndices = list(range(fullLUT.shape[0]))
                outputArray = randRules[i]
                indicesToOne = np.where(outputArray == 1)[0]
                LUTtoOne = torch.tensor(fullLUT[indicesToOne, :])
                bool = boolion()  # creates the boolion object
                bool.loadLUT(LUTtoOne)
                bool.decomposeFunction(max_order_approx)  # computes the output for the given input
                derivatives = bool.derivatives
                derivativesList.append(derivatives)
            torch.save(derivativesList,randderivativesFolder + randderivativesFile)
            randInputsFuncsFolder = 'inputs_funcs_random_models/'
            randInputsFuncsFile = models_loaded[FileNumber].split('.txt')[0] + '_Random_inputs_funcs_nullmodel'+str(null_model_id)+'_run'+ str(run) + '.npz'
            np.savez(randInputsFuncsFolder + randInputsFuncsFile, I, randRules)  # saves the inputs and the new rules
    print(FileNumber, models_loaded[FileNumber],'simulated: ',round(time.time()-now,2),'seconds')

        