# Compute approximation errors for both the Bio models and the associated random ensembles, and
# store the error data in a single for each Bio model

import torch
import os
import numpy as np
import sys
sys.path.append('../')

# Data source location
BioInputsFuncsFolder = "inputs_funcs_bio_models/"
BioInputsFuncsFiles = sorted(os.listdir(BioInputsFuncsFolder))

data_folder =  'Data/DataFull/Data/'
data_folder =  'DataFinal/Data/'

DataFiles = sorted(os.listdir(data_folder))

all_data_bio = []
all_data_random = []
max_orders = []

N_SLURM_IDS = 50

for FileNumber in range(    len(BioInputsFuncsFiles)):
    prefix = BioInputsFuncsFiles[FileNumber].split('_Bio')[0] + '_SimulationData'
    print(FileNumber, sum([prefix in el for el in DataFiles]), BioInputsFuncsFiles[FileNumber])

    if sum([prefix in el for el in DataFiles]) != N_SLURM_IDS:
        continue
    
    dummy_all_data_bio = []
    dummy_all_data_random = []
        
    for SLURM_ID in range(N_SLURM_IDS):
        suffix = '_SLURMID'+str(SLURM_ID)
        SimulationFile = './' + data_folder + prefix + suffix +'.dat'
        SimulationData = torch.load(SimulationFile)
    
    
        MaxOrder = SimulationData['Bio']['final'].shape[1]
        if SLURM_ID==0:
            max_orders.append(MaxOrder)
        n_iter = SimulationData['Bio']['final'].shape[0]
    
        BioFinalExact = SimulationData['Bio']['final'][:,-1,:]  # shape = (NumSamples,MaxOrder,NumNodes)
        RandomFinalExact = SimulationData['Random']['final'][:,:,:,-1,:]  # shape = (NumRandomModels,NumNullModels,NumSamples,MaxOrder,NumNodes)
    
        ErrorData = dict()
        ErrorData['Bio'] = dict()
        ErrorData['Random'] = [dict(),dict(),dict()]
        for order in range(MaxOrder):
            BioFinalApprox = SimulationData['Bio']['final'][:,order,:]  # 0 = linear; 1 = quadratic, etc.
            BioApproxError = torch.pow(BioFinalExact - BioFinalApprox, 2).mean()
            ErrorData['Bio'][order] = BioApproxError.item()
    
            for null_model_id in range(1,4):
                RandomFinalApprox = SimulationData['Random']['final'][:,null_model_id-1,:,order,:]  # 0 = linear; 1 = quadratic, etc.
                RandomApproxError = torch.pow(RandomFinalExact[:,null_model_id-1,:,:] - RandomFinalApprox, 2).mean()
                
                ErrorData['Random'][null_model_id-1][order] = RandomApproxError.item()
        
        dummy_all_data_bio.append(ErrorData['Bio'])
        dummy_all_data_random.append(ErrorData['Random'])
        
    all_data_bio.append([np.mean([dictionary[order] for dictionary in dummy_all_data_bio]) for order in range(MaxOrder)])
    all_data_random.append([[np.mean([dummy_all_data_random[ii][null_model_id][order] for ii in range(N_SLURM_IDS)]) for order in range(MaxOrder)] for null_model_id in range(3)])

    #AnalysisFile = './Data/' + BioInputsFuncsFiles[FileNumber].split('_Bio')[0] + '_AnalysisData.dat'
    #torch.save(ErrorData,AnalysisFile)

import matplotlib.pyplot as plt
import scipy.stats as stats

def format_p(p):
    if p>0.1:
        return str(round(p,2))
    elif p>0.01:
        return str(round(p,3))
    elif p>=0.001:
        return str(round(p,4))
    else:
        return '{:.0e}'.format(p)
    
def format_p_latex(p):
    if p>0.1:
        return str(round(p,2))
    elif p>0.01:
        return str(round(p,3))
    elif p>=0.001:
        return str(round(p,4))
    else:
        return r'$10^{%i}$' % round(np.log(p)/np.log(10))

for null_model_id in range(1,4):
    for order in range(3):
        f,ax = plt.subplots(figsize=(3.3,3.3))
        x = [el[order] if len(el)>order else 0 for el in all_data_bio]
        y = [el[null_model_id-1][order] if len(el[null_model_id-1])>order else 0 for el in all_data_random]
        ax.plot(x,y,'o')
        [x1,x2] = ax.get_xlim()
        [y1,y2] = ax.get_ylim()
        [xmin,xmax] = min(x1,y1),max(x2,y2)
        ax.set_xlim([xmin,xmax])
        ax.set_ylim([xmin,xmax])
        ax.text(xmin+0.75*(xmax-xmin),xmin+0.92*(xmax-xmin),r'$\rho = $'+str(round(stats.spearmanr(x,y)[0],2)),va='center',ha='left')
        ax.text(xmin+0.75*(xmax-xmin),xmin+0.84*(xmax-xmin),r'$p = $'+format_p(stats.spearmanr(x,y)[1]),va='center',ha='left')
        ax.set_xlabel('bio models' )
        ax.set_ylabel('random null models (type %i)'  % null_model_id)
        ax.set_title('Mean order %i approximation error' % (order+1))
        
        print(order+1,stats.wilcoxon(x,y,alternative='two-sided')[1],stats.ttest_rel(x,y)[1],stats.ttest_ind(x,y,equal_var=False)[1])


f,ax = plt.subplots(nrows=3,ncols=3,figsize=(8,8))
for ii,null_model_id in enumerate(range(1,4)):
    for order in range(3):
        x = [el[order] if len(el)>order else 0 for el in all_data_bio]
        y = [el[null_model_id-1][order] if len(el[null_model_id-1])>order else 0 for el in all_data_random]
        ax[ii,order].plot(x,y,'o')
        [x1,x2] = ax[ii,order].get_xlim()
        [y1,y2] = ax[ii,order].get_ylim()
        [xmin,xmax] = min(x1,y1),max(x2,y2)
        ax[ii,order].plot([xmin,xmax],[xmin,xmax],'k--')
        ax[ii,order].set_xlim([xmin,xmax])
        ax[ii,order].set_ylim([xmin,xmax])
        ax[ii,order].text(xmin+0.5*(xmax-xmin),xmin+0.92*(xmax-xmin),r'$\rho = $'+str(round(stats.spearmanr(x,y)[0],2)),va='center',ha='left')
        #ax[ii,order].text(xmin+0.5*(xmax-xmin),xmin+0.84*(xmax-xmin),r'$p = $'+format_p(stats.spearmanr(x,y)[1]),va='center',ha='left')
        if ii==2:
            ax[ii,order].set_xlabel('biological networks' )
        if order==0:
            ax[ii,order].set_ylabel('random null models (type %i)'  % null_model_id)
        if ii==0:
            ax[ii,order].set_title('Order %i MAE' % (order+1))
        print(order+1,stats.wilcoxon(x,y,alternative='two-sided')[1],stats.ttest_rel(x,y)[1],stats.ttest_ind(x,y,equal_var=False)[1])
plt.subplots_adjust(hspace=0.3,wspace=0.3)
plt.savefig('order_vs_MAE_N%i_detailed.pdf' % (len(all_data_bio)),bbox_inches = "tight")


import pandas as pd
A = pd.read_excel('../derrida_and_other_parameters_per_network_N122.xlsx')
inputfuncfiles = list(map(lambda x: x.split('_Bio_inputs')[0]+'.txt',BioInputsFuncsFiles))
dict_A = dict(zip(list(A['models_loaded']),range(len(A))))
derrida_values = np.array(A['derrida'])[np.array([dict_A[el] for el in inputfuncfiles])]
mean_EC = np.array(A['mean_EC'])[np.array([dict_A[el] for el in inputfuncfiles])]
mean_IR = np.array(A['mean_IR'])[np.array([dict_A[el] for el in inputfuncfiles])]
mean_bias = np.array(A['mean_bias'])[np.array([dict_A[el] for el in inputfuncfiles])]
prop_NCF = np.array(A['prop NCF'])[np.array([dict_A[el] for el in inputfuncfiles])]
mean_deg_essential = np.array(A['mean deg essential'])[np.array([dict_A[el] for el in inputfuncfiles])]

import pickle
with open('../deg_essential_N122', 'rb') as f:
    deg_essential_N122 = pickle.load(f)
with open('../bias_N122', 'rb') as f:
    bias_N122 = pickle.load(f)
    
bias = [bias_N122[index][:] for index in [dict_A[el] for el in inputfuncfiles]]
deg_essential = [deg_essential_N122[index][:] for index in [dict_A[el] for el in inputfuncfiles]]

with open('../attractor_info_N122', 'rb') as f:
    dummy_N122 = pickle.load(f)
dummy_N122 = np.array(dummy_N122).T
dummy = np.array([dummy_N122[index].copy() for index in [dict_A[el] for el in inputfuncfiles]]).T
[lower_bound_number_attractors,mean_length_attractors,mean_length_attractors_weighted_by_basinsize,entropy_basin_sizes,prop_ss,prop_ss_weighted_by_basinsize] = dummy





import matplotlib

n_variables = list(map(len,bias))
cov = [np.cov(k,p)[0,1] for k,p in zip(deg_essential,bias)]
correl = [stats.pearsonr(k,p)[0] for k,p in zip(deg_essential,bias)]
order_1_MAE = [el[0] if len(el)>0 else 0 for el in all_data_bio]
order_2_MAE = [el[1] if len(el)>1 else 0 for el in all_data_bio]
order_3_MAE = [el[2] if len(el)>2 else 0 for el in all_data_bio]
PEARSON = False
KENDALL = False
func = stats.pearsonr if PEARSON else (stats.kendalltau if KENDALL else stats.spearmanr)
notnan = np.isnan(derrida_values)==False
ys =[n_variables,prop_NCF,mean_bias,mean_deg_essential,mean_EC,cov,
     np.array(mean_bias)*np.array(mean_deg_essential),
     np.array(mean_bias)*np.array(mean_deg_essential)+cov,
     np.array(mean_bias)*np.array(mean_EC),
     derrida_values,lower_bound_number_attractors,mean_length_attractors,entropy_basin_sizes,prop_ss,order_1_MAE,order_2_MAE,order_3_MAE]
labels = 'network size,proportion NCFs,<p(1-p)>,<$K$>,<$K_e$>,Cov(p(1-p);K),<K><p(1-p)>,<K><p(1-p)>+Cov,<Ke><p(1-p)>,mean average sensitivity,minimal number attractors,mean length attractors,entropy basin sizes,proportion steady states,order 1 MAE,order 2 MAE,order 3 MAE'.replace('_','\n').split(',')
labels[2] = r'<$p(1-p)$>'
labels[3] = r'<$K$>'
labels[4] = r'<$K_e$>'
labels[5] = r'Cov($p(1-p),K$)'
labels[-3-3-4] = r'<$K$><$p(1-p)$> + Cov'
labels[-2-3-4] = r'<$K_e$><$p(1-p)$>'
# ys =[n_variables,prop_NCF,mean_bias,mean_deg_essential,mean_EC,cov,
#      np.array(mean_bias)*np.array(mean_deg_essential),
#      np.array(mean_bias)*np.array(mean_deg_essential)+cov,
#      np.array(mean_bias)*np.array(mean_EC),
#      derrida_values,lower_bound_number_attractors,mean_length_attractors,mean_length_attractors_weighted_by_basinsize,entropy_basin_sizes,prop_ss,prop_ss_weighted_by_basinsize,order_1_MAE,order_2_MAE,order_3_MAE]
# labels = 'network size,proportion NCF,<p(1-p)>,<$K$>,<$K_e$>,Cov(p(1-p);K),<K><p(1-p)>,<K><p(1-p)>+Cov,<Ke><p(1-p)>,mean average sensitivity,minimal number attractors,mean length attractors,mean length attractors weighted,entropy basin sizes,proportion steady states,weigthed proportion steady states,order 1 MAE,order 2 MAE,order 3 MAE'.replace('_','\n').split(',')
# labels[2] = r'<$p(1-p)$>'
# labels[3] = r'<$K$>'
# labels[4] = r'<$K_e$>'
# labels[5] = r'Cov($p(1-p),K$)'
# labels[-3-3-6] = r'<$K$><$p(1-p)$> + Cov'
# labels[-2-3-6] = r'<$K_e$><$p(1-p)$>'
n_ys = len(ys)
correl_mat = np.ones((n_ys,n_ys))
for i in range(n_ys):
    for j in range(i+1,n_ys):
        correl_mat[i,j] = func(np.array(ys[i])[notnan],np.array(ys[j])[notnan])[0]
        correl_mat[j,i] = correl_mat[i,j] 
f,ax = plt.subplots(figsize=(4.4,3.5))   
im = ax.imshow(correl_mat,cmap=matplotlib.cm.RdBu,vmin=-1,vmax=1)
cbar = plt.colorbar(im)
cbar.set_label(('Pearson' if PEARSON else ('Kendall' if KENDALL else 'Spearman')) + ' correlation')
ax.set_xticks(range(n_ys))
ax.set_xticklabels(labels,rotation=90)    
ax.set_yticks(range(n_ys))
ax.set_yticklabels(labels)     
plt.savefig(('Pearson' if PEARSON else ('Kendall' if KENDALL else 'Spearman'))+'_derrida_N%i.pdf' % len(all_data_bio),bbox_inches = "tight")    

import hierarchial_clustering as hc
filename = ('Pearson' if PEARSON else ('Kendall' if KENDALL else 'Spearman'))+'_derrida_N%i_hc.pdf' % len(all_data_bio)
hc.heatmap(correl_mat,row_header=labels,column_header = labels,row_method = 'average', column_method = 'average', row_metric='euclidean', column_metric='euclidean', color_gradient=matplotlib.cm.RdBu, filename=filename, title_colormap=('Pearson' if PEARSON else ('Kendall' if KENDALL else 'Spearman')) + ' correlation', CENTERED_AT_0 = True,ticks=[],ticklabels=[],rel_width_panel=0.3,rel_height_panel=0.3,fontproperties=None,ref_values=[])

def arg_closest(lst, K):
     lst = np.asarray(lst)
     idx = (np.abs(lst - K)).argmin()
     return idx

indices_of_interest = np.where(['MAE' in el for el in labels])[0]
other_indices = np.where(['MAE' not in el for el in labels])[0]
dat = correl_mat[other_indices,:][:,indices_of_interest]
sorted_order = np.argsort(np.mean(dat,1))[::-1]
f,ax = plt.subplots(figsize=(3.5,3))
markers = ['x','o','d']
for i in range(3):
    ax.plot(dat[sorted_order,i],marker=markers[i],ls=':',label=str(i+1))
[x1,x2] = ax.get_xlim()
[y1,y2] = ax.get_ylim()
ax.plot([x1,x2],[0,0],'k--')
dof  = len(order_1_MAE) - 2
dist = stats.t(df=dof)
t_vals = np.linspace(-5, 5, 10000)
rs = np.linspace(0,y2,10000)
transformed = np.abs(rs) * np.sqrt(dof / ((np.abs(rs)+1.0)*(1.0-np.abs(rs))))
pvalue = dist.cdf(-transformed) + dist.sf(transformed)
list_ps = [0.05,0.001,1e-10,1e-20]
r_to_ps = [rs[arg_closest(pvalue,p)] for p in list_ps]
ax2 = ax.twinx()
ax2.set_yticks([-el for el in r_to_ps]+[0]+r_to_ps)
ax2.set_yticklabels(list(map(format_p_latex,list_ps+[1]+list_ps)))
ax2.set_ylabel('p-value')
ax.set_xlim([x1,x2])
ax.set_xticks(range(len(other_indices)))
ax.set_xticklabels(np.array(labels)[other_indices][sorted_order],rotation=90)
ax.legend(title='MAE of order',loc='best',frameon=False)
ax.set_ylabel(('Pearson' if PEARSON else ('Kendall' if KENDALL else 'Spearman')) + ' correlation')
plt.savefig(('Pearson' if PEARSON else ('Kendall' if KENDALL else 'Spearman'))+'_simple_N%i.pdf' % len(all_data_bio),bbox_inches = "tight")    
























from matplotlib import cm
width=0.7
max_order_to_plot=3
f,ax = plt.subplots(figsize=(7,4))
for null_model_id in range(1,4):
    ys = [[el[null_model_id-1][order] if len(el[null_model_id-1])>order else 0 for el in all_data_random] for order in range(max_order_to_plot)]
    ax.boxplot(ys,widths=width,positions = [4*order+null_model_id-1 for order in range(max_order_to_plot)])#,color=cm.tab20(null_model_id))
xs = [[el[order] if len(el)>order else 0 for el in all_data_bio] for order in range(max_order_to_plot)]
ax.boxplot(xs,widths=width,positions = [4*order+3 for order in range(max_order_to_plot)])#,color='b')
ax.set_xlabel('order of approximation' )
ax.set_ylabel('mean approximation error')
    
from matplotlib import cm
width=0.7
max_order_to_plot=3
f,ax = plt.subplots(figsize=(7,4))
bplots = []
positions_x = 1.5+4*np.arange(max_order_to_plot)
ys_null_models = []
for null_model_id in range(1,4):
    ys = [[el[null_model_id-1][order] if len(el[null_model_id-1])>order else 0 for el in all_data_random] for order in range(max_order_to_plot)]
    bplots.append( ax.boxplot(ys,widths=width,positions = positions_x - 0.9*1.5+0.9*(null_model_id-1),patch_artist=True,medianprops={'color':'k'},boxprops={'facecolor':cm.tab20c(null_model_id)}) )
    ys_null_models.append(ys)
xs = [[el[order] if len(el)>order else 0 for el in all_data_bio] for order in range(max_order_to_plot)]
bplots.append( ax.boxplot(xs,widths=width,positions = positions_x - 0.9*1.5+0.9*3,patch_artist=True,medianprops={'color':'k'},boxprops={'facecolor':cm.tab20c(4)}) ) 
ax.spines[['right','top']].set_visible(False)
ax.set_xlabel('order of approximation' )
ax.set_ylabel('mean approximation error')
ax.set_xticks(positions_x) 
ax.set_xticklabels(list(map(str,range(1,max_order_to_plot+1))))
[y1,y2] = ax.get_ylim()
for null_model_id in range(1,4):
    for order in range(max_order_to_plot):
        #p = stats.ttest_rel(xs[order],ys_null_models[null_model_id-1][order])[1]
        p = stats.wilcoxon(xs[order],ys_null_models[null_model_id-1][order])[1]
        x_dummy = [positions_x[order] - 0.9*1.5+0.9*(null_model_id-1),positions_x[order] - 0.9*1.5+0.9*3]
        y_dummy = [y2+0.1*(y2-y1)*(null_model_id-1)]*2
        ax.plot(x_dummy,y_dummy,'k',lw=0.5)
        ax.text(np.mean(x_dummy),y2+0.1*(y2-y1)*(null_model_id-1+0.4),'p = '+format_p(p),va='center',ha='center',fontsize=9)

ax.legend([bplots[i]['boxes'][0] for i in range(4)], ['null model '+str(null_model_id) for null_model_id in range(1,4)] + ['biological networks'],loc='best',frameon=False)
plt.savefig('order_vs_MAE_N%i.pdf' % (len(all_data_bio)),bbox_inches = "tight")

