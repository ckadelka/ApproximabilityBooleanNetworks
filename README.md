# DesignPrinciplesGeneNetworks
A pipeline to compute the nonlinearity in regulation of Boolean biological networks and compare it to a variety of random network models

This repository contains all code to perform the analyses described in the paper "Canalization reduces the nonlinearity of regulation in biological networks", available at [https://arxiv.org/abs/2402.09703](https://arxiv.org/abs/2402.09703).

Three programs need to be run sequentially:
1. PrepareModelsForSimulation_v4.py, 
2. SimulateModels_v4.py, and
3. AnalyzeSimulationData_v4b.py.

Additional key programs generate small random N-K Kauffman networks and analyze them. This includes:
1. generateKauffmanNetworks_v03.py
2. generateCanalizingNetworks.py

These two programs use the Python library canalizing_function_toolbox_v13, available from [https://github.com/ckadelka/DesignPrinciplesGeneNetworks](https://github.com/ckadelka/DesignPrinciplesGeneNetworks).

# PrepareModelsForSimulation_v4.py
This program loads all published expert-curated Boolean network models from a recent [meta-analysis](https://www.science.org/doi/full/10.1126/sciadv.adj0822), whose nodes have a predefined maximal in-degree (10 used in the paper; to avoid an exponential increase in run time) The models are stored in a list of folders, as text files in a standardized format:
```text
A = B OR C
B = A OR (C AND D)
C = NOT A
```
This little example represents a model with three genes, A, B and C, and one external parameter D (which only appears on the right side of the equations).

For each biological network, this program generates three different sets of random null models. All null models preserve the dependency graph (wiring diagram) of the original network. In addition,
1. Null model 1 preserves the bias of each Boolean update rule, 
2. Null model 2 preserves the canalizing depth of each Boolean update rule,
3. Null model 3 preserves both the bias and canalizing depth  of each Boolean update rule.

Ideally, this program is run in parallel on some HPC infrastructure. We used the shell script bash_v4_unix.sh to run it on the Iowa State HPC cluster Pronto.

# SimulateModels_v4.py
This program, using the Python libraries torch and [boolion](https://gitlab.com/smanicka/boolion), simulates the biological networks as well as the corresponding random null models, both the Boolean versions and continuous extensions of the Boolean networks of various degree. 

Ideally, this program is run in parallel on some HPC infrastructure. We used the shell script bash_v4_step2_unix.sh to run it on the Iowa State HPC cluster Pronto.

# AnalyzeSimulationData_v4b.py
This program computes the approximability of each analyzed network. It generates the plots in the paper "Canalization reduces the nonlinearity of regulation in biological networks", available at [https://arxiv.org/abs/2402.09703](https://arxiv.org/abs/2402.09703). For some of these plots, additional model-specific data from the meta-analysis is required. This includes:
1. derrida_and_other_parameters_per_network_N122.xlsx
2. bias_N122
3. deg_essential_N122
4. attractor_info_N122


 

