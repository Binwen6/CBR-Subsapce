# CBR-Subsapce
This repository contains the codes and datasets for our paper **"Cell-Based Representation of Relational Binding in Language Models"**

We analyze how LLMs encode discourse and identify a structured subspace, which we term the CBR subspace. Moreover, causal interventions on the CBR subspace can systematically alter LLM behavior.

#### Visualization of CBR Subspace: 
~~~
./cbr_subspace_visualization.png
~~~
<!--  <img src="/cbr_subspace_visualization.png" width="500"> -->

#### Overview of CBR Subspace based mechanism:
~~~
./mechanism.png
~~~
<!-- <img src="/mechanism.png" width="500"> -->

#### Method
The implementation in this paper is based on [baukit](https://github.com/davidbau/baukit/tree/main)  and its application in the [paper](https://github.com/Nix07/finetuning/tree/main).

## Setup
Install dependencies (creating a virtual environment called "finetuning") :
~~~~
conda env create -f environment.yml
conda activate finetuning
~~~~

## Dataset
The main dataset used in this paper is available in `./data`, or you can create a new dataset by running the following script.
~~~
python script/data_creation.py --nb_sample 1000
~~~

## Activation Extraction
Run the following script for extracting attribute activations for CBR subspace analysis.
~~~
python script/activation_extraction.py
~~~

## Sampling CBR Subspace
Run the following script for sampling CBR subspace.
#### Step1: sampling points from CBR subspace
~~~
python script/activation_sampling.py --llm_tp llama/qwen --learn_proj_ma
~~~
#### Step2: activation patching via the sampled points
~~~
python script/activation_sampling.py --llm_tp llama/qwen
~~~

## Perturbing CBR Subspace
Run the following script for perturbing CBR subspace.
~~~
python script/activation_perturbing.py --llm_tp llama/qwen --use_rand_proj_ma 0/1
~~~

## Steering CBR Subspace
Run the following script for activation steering on CBR subspace.
#### Step1: learning steering vector
~~~
python script/steer_v_learning.py --llm_tp llama/qwen
~~~
#### Step2: activation patching via the learned steering vector
~~~
python script/activation_steering.py --llm_tp llama/qwen --steer_tp steer_a_1_2/steer_a_2_3/steer_a_3_4/steer_e_2_3
~~~

## Results Visualization
~~~
notebook/jupyter_script_for_visualization.ipynb
~~~
Please run the Jupyter Notebook script above for:
1. CBR subspace visualization
2. CBR subspace prediction via PLS from the activations of LLMs
3. Analyzing the generality of CBR subspace across contexts
4. Result visualization for Perturbing CBR subspace
5. Result visualization for activation steering on CBR subspace
6. Result visualization for CBR subspace sampling

