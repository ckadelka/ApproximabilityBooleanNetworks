#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 14:47:22 2022

@author: ckadelka
"""



import numpy as np
import itertools
import matplotlib.pyplot as plt

import canalizing_function_toolbox_v13 as can
from boolionnet import boolionnet
import torch


#parameter choices for network generation
N = 15 #network size
n = 4 #degree parameter: average degree if constant of Poisson
k = 3 #canalizing depth
STRONGLY_CONNECTED = True
indegree_distribution = 'constant'
kis = None
EXACT_DEPTH=True
NO_SELF_REGULATION=True

#parameter choices for boolionnet nonlinearity approximation
NumSamples = 100
NumIters = 25
approxUpto = 1

#parameter choices for the simulation
nsim = 50

#initialize other helper variables for network generation
if indegree_distribution == 'constant':
    list_x = [[[0],[1]]]
    list_x.extend([list(itertools.product([0, 1], repeat = nn)) for nn in range(2, k+1)])
else:
    list_x = None

ks = list(range(n-1)) + [n]
approxUptos = list(range(1,n+1))


# ks = [0]
# n = [4] * 7 + [2] * 8


differences_overall = []
for k in ks:
    differences = []
    for iii in range(nsim):
        print(k,iii)
        differences.append([])
        
        ## generate random Boolean network
        F,I,degrees = can.random_BN(N, n = n, k = k, STRONGLY_CONNECTED = STRONGLY_CONNECTED, indegree_distribution = indegree_distribution, list_x=list_x, kis = kis, EXACT_DEPTH=EXACT_DEPTH,NO_SELF_REGULATION=NO_SELF_REGULATION)
           
        ## compute nonlinearity of dynamics
        bn = boolionnet(inputsList=I, functionsList=F, derivativesFile='')
        initState = torch.tensor(np.random.randint(2, size=(NumSamples, bn.n)))
        for approxUpto in approxUptos:
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
            differences[-1].append(float(Difference))
    differences_overall.append(differences)

differences_overall = np.array(differences_overall)
min_overall,max_overall = np.min(differences_overall), np.max(differences_overall)


#analyze results
SAVEFIG= False

for ii,approxUpto in enumerate(approxUptos):
    f,ax = plt.subplots()
    ax.boxplot(differences_overall[:,:,ii].T,showmeans=True)
    ax.set_ylabel('mean approximation error\n(using Taylor polyomials of order %i)' % approxUpto)
    ax.set_xlabel('minimal canalizing depth')
    ax.set_xticklabels(list(map(str,ks)))
    ax.set_ylim([min_overall-0.02*(max_overall-min_overall),max_overall+0.02*(max_overall-min_overall)])
    if type(n) == int:
        ax.set_title('Boolean networks of size %i with constant in-degree %i' % (N,n))
    if SAVEFIG:
        plt.savefig('linearity_vs_canalization_N%i_n%i_order%i_nsim%i_nsamples%i_niters%i.pdf' % (N,n,approxUpto,nsim,NumSamples,NumIters),bbox_inches = "tight")
    
for ii,k in enumerate(ks):
    f,ax = plt.subplots()
    ax.boxplot(differences_overall[ii,:,:],showmeans=True)
    ax.set_ylabel('mean approximation error')
    ax.set_xlabel('using Taylor polyomials of order')
    ax.set_xticklabels(list(map(str,approxUptos)))
    ax.set_ylim([min_overall-0.02*(max_overall-min_overall),max_overall+0.02*(max_overall-min_overall)])
    if type(n) == int:
        ax.set_title('Boolean networks of size %i with constant in-degree %i\n and canalizing depth %i' % (N,n,k))
    if SAVEFIG:
        plt.savefig('linearity_vs_canalization_N%i_n%i_k%i_nsim%i_nsamples%i_niters%i.pdf' % (N,n,k,nsim,NumSamples,NumIters),bbox_inches = "tight")
    
SAVEFIG= True
import matplotlib.cm
f,ax = plt.subplots()
bplots = []
for ii,approxUpto in enumerate(approxUptos):
    bplots.append( ax.boxplot(differences_overall[:,:,ii].T,widths=0.8,showmeans=False,positions=5*np.arange(n)-1.5+ii,patch_artist=True,medianprops={'color':'k'},boxprops={'facecolor':matplotlib.cm.Pastel1(ii)}) )
ax.set_ylabel('mean approximation error')
ax.set_xlabel('minimal canalizing depth')
ax.set_xticks(5*np.arange(n)) 
ax.set_xticklabels(list(map(str,ks)))
ax.spines[['right','top']].set_visible(False)
[x1,x2] = ax.get_xlim()
ax.set_xlim([x1-0.01*(x2-x1),x2+0.01*(x2-x1)])
ax.legend([bplots[i]['boxes'][0] for i in range(4)],list(map(str,approxUptos)),title='order of Taylor polynomial',ncol=4,loc=9,frameon=False)
if SAVEFIG:
    plt.savefig('linearity_vs_canalization_N%i_n%i_kall_orderall_nsim%i_nsamples%i_niters%i.pdf' % (N,n,nsim,NumSamples,NumIters),bbox_inches = "tight")



    

nsim=50
differences_overall = []
for w in range(1,2**(n-1),2):
    differences = []
    for iii in range(nsim):
        print(w,iii)
        differences.append([])
        
        ## generate random Boolean network
        kis = can.kindoflayer(n,w)[1]
        F,I,degrees = can.random_BN(N, n = n, k = n, STRONGLY_CONNECTED = STRONGLY_CONNECTED, indegree_distribution = indegree_distribution, list_x=list_x, kis = kis, EXACT_DEPTH=EXACT_DEPTH,NO_SELF_REGULATION=NO_SELF_REGULATION)
           
        ## compute nonlinearity of dynamics
        bn = boolionnet(inputsList=I, functionsList=F, derivativesFile='')
        initState = torch.tensor(np.random.randint(2, size=(NumSamples, bn.n)))
        for approxUpto in approxUptos:
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
            differences[-1].append(float(Difference))
    differences_overall.append(differences)

differences_overall = np.array(differences_overall)
min_overall,max_overall = np.min(differences_overall), np.max(differences_overall)

    

SAVEFIG= True
import matplotlib.cm
f,ax = plt.subplots()
bplots = []
for ii,approxUpto in enumerate(approxUptos):
    bplots.append( ax.boxplot(differences_overall[:,:,ii].T,widths=0.8,showmeans=False,positions=5*np.arange(n)-1.5+ii,patch_artist=True,medianprops={'color':'k'},boxprops={'facecolor':matplotlib.cm.Pastel1(ii)}) )
ax.set_ylabel('mean approximation error')
ax.set_xlabel('layer structure of nested canalizing functions')
ax.set_xticks(5*np.arange(n)) 
ax.set_xticklabels(['\n'.join([r'$k_%i = %i$' % (ii+1,el) for ii,el in enumerate(can.kindoflayer(n,w)[1])]) for w in range(1,2**(n-1),2)])#list(map(str,ks)))
ax.spines[['right','top']].set_visible(False)
[x1,x2] = ax.get_xlim()
[y1,y2] = ax.get_ylim()
ax.set_xlim([x1-0.01*(x2-x1),x2+0.01*(x2-x1)])
ax.set_ylim([y1,y2+0.06])
ax.legend([bplots[i]['boxes'][0] for i in range(4)],list(map(str,approxUptos)),title='order of Taylor polynomial',ncol=4,loc=9,frameon=False)
if SAVEFIG:
    plt.savefig('linearity_vs_layerstructure_N%i_n%i_kisall_orderall_nsim%i_nsamples%i_niters%i.pdf' % (N,n,nsim,NumSamples,NumIters),bbox_inches = "tight")

import pandas as pd
A = np.c_[list(map(str,[can.kindoflayer(n,w)[1] for w in range(1,2**(n-1),2)])),
          np.arange(1,2**(n-1),2)/2**n,
          differences_overall.mean(1),
          np.array(list(map(lambda x: list(map(np.median,differences_overall[:,:,x])),range(n)))).T]
A = pd.DataFrame(A,columns = ['layer structure','bias']+['mean MAE, order='+str(i+1) for i in range(n)]+['median MAE, order='+str(i+1) for i in range(n)])
A.to_excel('linearity_vs_layerstructure_N%i_n%i_kisall_orderall_nsim%i_nsamples%i_niters%i.xlsx' % (N,n,nsim,NumSamples,NumIters))

    
