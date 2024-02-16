#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 14:47:22 2022

@author: ckadelka
"""



import numpy as np
import itertools
import matplotlib.pyplot as plt
import sys
sys.path.append('../')

import canalizing_function_toolbox_v13 as can
from boolionnet import boolionnet
import torch


#parameter choices for network generation
N = 15 #network size
n_max = 5 #degree parameter: average degree if constant of Poisson
STRONGLY_CONNECTED = True
indegree_distribution = 'constant'
kis = None
EXACT_DEPTH=False
NO_SELF_REGULATION=True
ALLOW_DEGENERATED_FUNCTIONS=False

#parameter choices for boolionnet nonlinearity approximation
NumSamples = 100
NumIters = 25
approxUpto = 1

#parameter choices for the simulation
nsim = 2

#initialize other helper variables for network generation
if indegree_distribution == 'constant':
    list_x = [[[0],[1]]]
    list_x.extend([list(itertools.product([0, 1], repeat = nn)) for nn in range(2, n_max+1)])
else:
    list_x = None
    
approxUptos = list(range(1,n_max+1))
ps = np.linspace(0.1,0.5,5)
ns = list(range(2,n_max+1))

# ks = [0]
# n = [4] * 7 + [2] * 8

nsim  = 100

bool_list = np.array(list(map(np.array,list(itertools.product([0, 1], repeat = N)))))

MEASURE_ATTRACTORS = 0
MEASURE_AVG_SENS = 0

RANDOM_N_K_KAUFFMAN = False

res = []
for n in ns:
    res.append([])
    for bias in ps:
        res[-1].append([])
        for iii in range(nsim):    
            print(n,bias,iii)
            ## generate random Boolean network
            if RANDOM_N_K_KAUFFMAN:
                F = [np.random.choice(2,2**n) for _ in range(N)]
                I = [np.random.choice(N,n,False) for _ in range(N)]
                degrees = n*np.ones(N,dtype=int)
            else:
                F,I,degrees = can.random_BN(N, n = n, k = 0, STRONGLY_CONNECTED = STRONGLY_CONNECTED, indegree_distribution = indegree_distribution, list_x=list_x, kis = kis, EXACT_DEPTH=EXACT_DEPTH,NO_SELF_REGULATION=NO_SELF_REGULATION,bias=bias,ALLOW_DEGENERATED_FUNCTIONS=ALLOW_DEGENERATED_FUNCTIONS)
            
            
            
            if MEASURE_ATTRACTORS:
                attractors, len_attractors, basin_sizes, attractor_dict = can.num_of_attractors_exact_fast(F, I, N,left_side_of_truth_table = None)
                res[-1][-1].extend([len_attractors,can.entropy(basin_sizes),np.median(list(map(len,attractors))),np.mean(list(map(len,attractors))),np.dot(np.array(list(map(len,attractors))) == 1, basin_sizes)/2**N])
            else:
                res[-1][-1].extend([0,0,0,0,0])
                
            if MEASURE_AVG_SENS:
                res[-1][-1].append(sum([can.average_sensitivity(f,EXACT=True,NORMALIZED=False) for f in F])/len(F))
            else:
                res[-1][-1].append(0)
                
                
            ## compute nonlinearity of dynamics
            bn = boolionnet(inputsList=I, functionsList=F, derivativesFile='')
            initState = torch.tensor(np.random.randint(2, size=(NumSamples, bn.n)))
            for approxUpto in approxUptos:
                if approxUpto>=n:
                    res[-1][-1].append(0)
                    continue
                bn.set(initState)
                bn.update(iters=NumIters,approxUpto=approxUpto)  # approxUpto is the order of approximation (>=1 and <=n); 'default' means no approximation
                exactFinalState = torch.zeros(NumSamples, bn.n)
                for sample in range(NumSamples):
                    currentState = initState[sample]
                    for iter in range(NumIters):
                        nextState = can.update(F, I, bn.n, currentState)
                        currentState = nextState
                    exactFinalState[sample] = torch.tensor(nextState)
                Difference = torch.pow((bn.currentState - exactFinalState), 2).mean()
                res[-1][-1].append(float(Difference))
        print(n,bias)
res = np.array(res)

labels = ['number attractors','entropy basin sizes','median length attractors','mean length attractors','proportion steady states','mean average sensitivity','first-order MAE','second-order MAE','third-order MAE','fourth-order MAE','fifth-order MAE']
n_labels = len(labels)
assert n_labels == res.shape[2]/nsim
for i in range(n_labels):
    f,ax = plt.subplots()
    im = ax.imshow(res[:,:,i::n_labels].mean(2))
    ax.set_xticks(range(len(ps)))
    ax.set_xticklabels([str(round(el,1)) for el in ps])
    ax.set_yticks(range(len(ns)))
    ax.set_yticklabels(list(map(str,ns)))
    ax.set_xlabel('bias')
    ax.set_ylabel('constant in-degree')
    cbar = plt.colorbar(im)
    cbar.set_label(labels[i])
    

f, axes = plt.subplots(nrows=2, ncols=2, sharex='col',sharey='row',figsize=(6.5, 5))
for i,ax in enumerate(axes.flat):
    im = ax.imshow(res[:,:,(6+i)::n_labels].mean(2), cmap='inferno_r',
                   vmin=0, vmax=0.25)
    ax.set_title('order '+str(i+1))
    ax.set_xticks(range(len(ps)))
    ax.set_xticklabels([str(round(el,1)) for el in ps])
    ax.set_yticks(range(len(ns)))
    ax.set_yticklabels(list(map(str,ns)))
    if i%2==0:
        ax.set_ylabel('constant in-degree')
    if i//2==1:
        ax.set_xlabel('bias')
f.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
                    wspace=0.08, hspace=0.05)
# add an axes, lower left corner in [0.83, 0.1] measured in figure coordinate with axes width 0.02 and height 0.8
cb_ax = f.add_axes([0.83, 0.12, 0.02, 0.76])
cbar = f.colorbar(im, cax=cb_ax)
cbar.set_label('mean approximation error')
plt.savefig('k_vs_bias_N%i_nsim%i_nsamples%i_niters%i_nondegallowed%i.pdf' % (N,nsim,NumSamples,NumIters,int(ALLOW_DEGENERATED_FUNCTIONS)),bbox_inches = "tight")


import scipy.stats as stats

for ii in range(4):
    for jj in range(5):
        
        res_flat = res.ravel()
        f,ax = plt.subplots(nrows=5,ncols=5,figsize=(10,10),sharex='col',sharey='row')
        for i in range(5):
            for j in range(i+1,6):
                x = res_flat[i::n_labels]
                y = res_flat[j::n_labels]
                
                #x = res[ii,jj,i::n_labels]
                #y = res[ii,jj,j::n_labels]
                
                ax[j-1,i].plot(x,y,'o')
                [x1,x2] = ax[j-1,i].get_xlim()
                out = stats.linregress(x,y)
                ax[j-1,i].plot([x1,x2],[out.intercept+x1*out.slope,out.intercept+x2*out.slope],'r-')
                ax[j-1,i].set_xlim([x1,x2])
        for i in range(5):
            ax[i,0].set_ylabel(labels[i+1])
            ax[0,i].set_title(labels[i])
            ax[4,i].set_xlabel(labels[i])
        f.suptitle('k=%i, p=%s' % (ns[ii],str(ps[jj])))

res_flat = res.ravel()
for c in range(6):
    from matplotlib import cm
    f,ax = plt.subplots()
    i = labels.index('Derrida 1')
    j = labels.index('first-order MAE')
    x = res_flat[i::n_labels]
    y = res_flat[j::n_labels]
    ax.scatter(x,y,c=res_flat[c::n_labels],cmap=cm.inferno_r)
    [x1,x2] = ax.get_xlim()
    out = stats.linregress(x,y)
    ax.plot([x1,x2],[out.intercept+x1*out.slope,out.intercept+x2*out.slope],'r-')
    ax.set_xlim([x1,x2])
    ax.set_title('colored by '+labels[c])
    ax.set_xlabel(labels[i])
    ax.set_ylabel(labels[j])




from sklearn import linear_model
indices_x = [labels.index(el) for el in ['number attractors','entropy basin sizes','median length attractors','mean length attractors','proportion fixed point basin','Derrida 1']]
index_y = labels.index('first-order MAE')
X = []
for index in indices_x:
    X.append(res_flat[index::n_labels])
X = np.array(X)
y = res_flat[index_y::n_labels]
alpha_lasso = 0.1
clf = linear_model.Lasso(alpha=alpha_lasso)
clf.fit(X.T,y)



from sklearn.linear_model import LinearRegression
which_labels = ['number attractors','entropy basin sizes','mean length attractors','proportion steady states','mean average sensitivity']
indices_x = [labels.index(el) for el in which_labels]
index_y = labels.index('first-order MAE')
X = []
for index in indices_x:
    X.append(res_flat[index::n_labels])
X = np.array(X)
y = res_flat[index_y::n_labels]
reg = LinearRegression().fit(X.T, y)


import statsmodels.api as sm
import pandas as pd
data = pd.DataFrame(X.T,columns = which_labels)
data = sm.add_constant(data, prepend=False)
mod = sm.OLS(y, data)
result = mod.fit()
print(result.summary())


index_y = labels.index('third-order MAE')
y = res_flat[index_y::n_labels]
for i in range(5):
    print(which_labels[i],round(stats.spearmanr(X.T[:,i],y)[0],3))

data_mat = np.zeros((5,3))
for i in range(5):
    for j,label_y in enumerate(['first-order MAE','second-order MAE','third-order MAE']):
        index_y  =labels.index(label_y)
        y = res_flat[index_y::n_labels]
        data_mat[i,j] = stats.spearmanr(X.T[:,i],y)[0]

def arg_closest(lst, K):
     lst = np.asarray(lst)
     idx = (np.abs(lst - K)).argmin()
     return idx

#indices_of_interest = np.where(['MAE' in el for el in labels])[0]
#other_indices = np.where(['MAE' not in el for el in labels])[0]
sorted_order = np.argsort(np.mean(np.abs(data_mat),1))[::-1]
f,ax = plt.subplots(figsize=(3.5,3))
markers = ['x','o','d']
for i in range(3):
    ax.plot(data_mat[sorted_order,i],marker=markers[i],ls=':',label=str(i+1))
[x1,x2] = ax.get_xlim()
[y1,y2] = ax.get_ylim()
ax.plot([x1,x2],[0,0],'k--')
dof  = len(y) - 2
dist = stats.t(df=dof)
t_vals = np.linspace(-5, 5, 10000)
rs = np.linspace(0,y2,10000)
transformed = np.abs(rs) * np.sqrt(dof / ((np.abs(rs)+1.0)*(1.0-np.abs(rs))))
pvalue = dist.cdf(-transformed) + dist.sf(transformed)
list_ps = [0.05,0.001,1e-10,1e-20]
r_to_ps = [rs[arg_closest(pvalue,p)] for p in list_ps]
ax.set_xlim([x1,x2])
ax.set_xticks(range(len(which_labels)))
ax.set_xticklabels(np.array(which_labels)[sorted_order],rotation=90)
ax.legend(title='MAE of order',loc='best',frameon=False)
ax.set_ylabel('Spearman correlation')
plt.savefig('Spearman'+'_simple_N%i_nsim%i_nsamples%i_niters%i_nondegallowed%i.pdf' % (N,nsim,NumSamples,NumIters,int(ALLOW_DEGENERATED_FUNCTIONS)),bbox_inches = "tight")    


















from boolion import boolion
fullLUT = np.array(list(itertools.product([0, 1], repeat = 3)))
f = can.f_from_expression('(x1 * x2 + x3) % 2')[0]
indicesToOne = np.where(np.array(f))[0]
LUTtoOne = torch.tensor(fullLUT[indicesToOne, :])
a = boolion()
a.loadLUT(LUTtoOne)
a.decomposeFunction()
print(a.derivatives)


n=3
out_fs = []
out_der = []
out_contr = []
all_Boolean_functions = list(itertools.product([0, 1], repeat = 2**n))
for f in all_Boolean_functions:
    f = list(f)
    indicesToOne = np.where(np.array(f))[0]
    LUTtoOne = torch.tensor(fullLUT[indicesToOne, :])
    a = boolion()
    try:
        a.loadLUT(LUTtoOne)
        a.decomposeFunction()
    except:
        continue
    #print(f,a.derivatives)
    
    out_fs.append(f)
    out_der.append([])
    out_contr.append([])
    for i in range(1,n+1):
        out_der[-1].append(np.array(a.derivatives[i]))
        out_contr[-1].append(np.mean(np.abs(np.array(a.derivatives[i]))))

import pandas as pd
A = pd.DataFrame(np.c_[out_fs,out_contr])
A.to_excel('out.xlsx')  
    

























import scipy.optimize

def sample_zero_truncated_poisson(rate):
        u = np.random.uniform(np.exp(-rate), 1)
        t = -np.log(u)
        return 1 + np.random.poisson(rate - t)

n_max=3
approxUptos = list(range(1,n_max))
ps = np.linspace(0.1,0.5,5)
ns = list(range(2,n_max+1))

# ks = [0]
# n = [4] * 7 + [2] * 8

nsim  = 10

bool_list = np.array(list(map(np.array,list(itertools.product([0, 1], repeat = N)))))

MEASURE_ATTRACTORS = False

res = []
for n in ns:
    res.append([])
    for bias in ps:
        res[-1].append([])
        for indegree_distribution in ['constant','ZTP']:
            if indegree_distribution=='ZTP':
                degree_parameter = scipy.optimize.fsolve(lambda x: x*np.exp(x)/(np.exp(x)-1)-n, n)[0]
            else:
                degree_parameter = n
            res[-1][-1].append([])
            for iii in range(nsim):    
                print(n,bias,indegree_distribution,iii)
                ## generate random Boolean network
                degrees = [sample_zero_truncated_poisson(degree_parameter) for _ in range(N)]
                F,I,degrees = can.random_BN(N, n = degree_parameter if indegree_distribution=='constant' else degrees, k = 0, STRONGLY_CONNECTED = STRONGLY_CONNECTED, indegree_distribution = 'constant', list_x=list_x, kis = kis, EXACT_DEPTH=EXACT_DEPTH,NO_SELF_REGULATION=NO_SELF_REGULATION,bias=bias)#,ALLOW_DEGENERATED_FUNCTIONS=ALLOW_DEGENERATED_FUNCTIONS)
                
                if MEASURE_ATTRACTORS:
                    attractors, len_attractors, basin_sizes, attractor_dict = can.num_of_attractors_exact_fast(F, I, N,bool_list = None)
                    res[-1][-1][-1].extend([len_attractors,can.entropy(basin_sizes),np.median(list(map(len,attractors))),np.mean(list(map(len,attractors)))])
                else:
                    res[-1][-1][-1].extend([0,0,0,0])
                    
                ## compute nonlinearity of dynamics
                bn = boolionnet(inputsList=I, functionsList=F, derivativesFile='')
                initState = torch.tensor(np.random.randint(2, size=(NumSamples, bn.n)))
                for approxUpto in approxUptos:
                    if approxUpto>=n:
                        res[-1][-1][-1].append(0)
                        continue
                    bn.set(initState)
                    bn.update(iters=NumIters,approxUpto=approxUpto)  # approxUpto is the order of approximation (>=1 and <=n); 'default' means no approximation
                    exactFinalState = torch.zeros(NumSamples, bn.n)
                    for sample in range(NumSamples):
                        currentState = initState[sample]
                        for iter in range(NumIters):
                            nextState = can.update(F, I, bn.n, currentState)
                            currentState = nextState
                        exactFinalState[sample] = torch.tensor(nextState)
                    Difference = torch.pow((bn.currentState - exactFinalState), 2).mean()
                    res[-1][-1][-1].append(float(Difference))
res = np.array(res)



for jj,n in enumerate(ns):
    SAVEFIG= True
    import matplotlib.cm
    f,ax = plt.subplots()
    bplots = []
    for ii,indegree_distribution in enumerate(['constant','poisson']):
        bplots.append( ax.boxplot(res[jj,:,ii,4::6].T,widths=0.8,showmeans=False,positions=3*np.arange(5)-0.5+ii,patch_artist=True,medianprops={'color':'k'},boxprops={'facecolor':matplotlib.cm.Pastel1(ii)}) )
    ax.set_ylabel('mean first-order approximation error')
    ax.set_xlabel('bias')
    ax.set_xticks(3*np.arange(5)) 
    ax.set_xticklabels([str(round(el,1)) for el in ps])
    ax.spines[['right','top']].set_visible(False)
    [x1,x2] = ax.get_xlim()
    [y1,y2] = ax.get_ylim()
    ax.set_xlim([x1-0.01*(x2-x1),x2+0.01*(x2-x1)])
    ax.set_ylim([y1,y2+0.06])
    ax.legend([bplots[i]['boxes'][0] for i in range(2)],['constant','ZTP'],title='in-degree distribution',ncol=2,loc=9,frameon=False)
    if SAVEFIG:
        plt.savefig('indegree_N%i_n%i_kall_orderall_nsim%i_nsamples%i_niters%i.pdf' % (N,n,nsim,NumSamples,NumIters),bbox_inches = "tight")























