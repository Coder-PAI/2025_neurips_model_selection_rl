This is an official repository for our paper [**Model Selection for Off-policy Evaluation: New Algorithms and Experimental Protocol (NeurIPS 2025 Poster)**](https://arxiv.org/pdf/2502.08021).

## Installation
Below is our recommendation with regard to experiment setup & configuration.
+ **Python Environment**
    + See `environment.yml`. We strongly recommend readers to run our code using `python 3.10` to avoid potential incompatibility.
    + For Windows users, we recommend performing the installation through `WSL`, since `Mujoco 2.3.3` is required to be set up in a Linux environment.
 
## Table: Details of Experiment Settings

| ID | Gravity $g$ | Noise Level $\sigma$ | Groundtruth Model $M^\star$ and Behavior Policy $\pi_b$ | Target Policies $\Pi$ |
|----|------------------|----------------------|------------------------------------------------------------|-----------------------|
| **MF.G** | $\text{LIN}(-51, -9, 15)$ | 100 | $\{(M_i, \pi_i^{\epsilon}), i\in \{0,7,14\}\}$ | $\{\pi_{0:9}\}$ |
| **MF.N** | -30 | $\text{LIN}(10,100,15)$ | $\{(M_i, \pi_i^{\epsilon}), i\in \{0,7,14\}\}$ | $\{\pi_{0:9}\}$ |
| **MB.G** | $\text{LIN}(-36,-24,5)$ | 100 | $\{(M_i, \pi_i^{\epsilon}), i\in \{0,2,4\}\}$ | $\{\pi_{0:5}\}$ |
| **MB.N** | -30 | $\text{LIN}(10,100,5)$ | $\{(M_i, \pi_i^{\epsilon}), i\in \{0,2,4\}\}$ | $\{\pi_{0:5}\}$ |
| **MF.OFF.G** | $\text{LIN}(-51, -9, 15)$ | 100 | $\{(M_i, \pi_i^{\textrm{poor}}), i\in \{0,7,14\}\}$ | $\{\pi_{0:9}\}$ |
| **MF.OFF.N** | -30 | $\text{LIN}(10,100,15)$ | $\{(M_i, \pi_i^{\textrm{poor}}), i\in \{0,7,14\}\}$ | $\{\pi_{0:9}\}$ |
| **MF.T.G** | $\text{LIN}(-51, -9, 15)$ | 100 | $\{(M_i, \pi_8 \text{~\&~} \pi_i^{\textrm{poor}}), i\in \{0,7,14\}\}$ | $\{\pi_8\}$ |

> **Note:** $\text{LIN}(a,b,n)$ (per NumPy convention) denotes the arithmetic sequence  
> with $n$ elements, starting from $a$ and ending in $b$, e.g.  
> $\text{LIN}(0,1,6)=\{0, 0.2, 0.4, 0.6, 0.8, 1.0\}$.



## Main Usage
This part serves as a general guide for reproducing the primary results of **MF.G, MF.N, MF.OFF.G, MF.OFF.N, MB.G, and MB.N**, including analyses of **sample efficiency, misspecification, gap, convergence, and sanity checks**.
### Notes on MF.X Reproducibility
We exemplify the code usage on MF.G.
+ Create a separate folder for your full experiment.
+ Copy `mf_on_policy_gravity.py` from `env_setup` (the folder that gathered experimental configurations), and rename it as `parser.py`.
+ Copy all `.py` files from `common` folder.

After completing the above steps, the code and data should be structured in parallel as shown below. Generating policies, datasets, or function estimates from scratch will automatically trigger re-caching of the relevant parts in offline_data; note that this process can take up to 2–3 days, according to our experience.
+ `your_folder`
    + `offline_data`
        + `policies` - 0, 1, ..., 14; the evaluation policy class (which also induces a behavior policy class for **MF.G, MF.N** using epsilon-greedy mapping)
        + `datasets` - 0, 1, ..., 14; 
        + `q_functions` - 0, 1, ..., 14
        + `v_functions` - 0, 1, ..., 14
    + `ablation_observer.py`
    + ......
    + `validator.py`

You can then execute `python3 main.py` for a full running.

+ (Optional) If you would like to leverage existing data:
    + Reuse our running data from `experiments_data`.
    + For example, if you are running **MF.OFF.G** experiments, copy `offline_data` from `experiments_data/mf-off-g` and adhere to the aforementioned file structure. 

### Notes on MF.OFF.X Reproducibility
In off-policy experiments, we trained a set of behavior policies that approximately enhanced the distributional shift. While you are recommended to follow the same pipeline to retrieve code & data from our **MF.OFF.G** or **MF.OFF.N** experiments, you still need to:
+ Replace current `offline_dataset_collector.py, policy_trainer.py, main.py` in your folder with the ones in `additional_code/off_policy`.

### Notes on MB.X Reproducibility
In model-based experiments, we necessitate the offline cache for bellman operators. While you are recommended to follow the same pipeline to retrieve code & data from our **MB.G** or **MB.N** experiments, you still need to:
+ Drag bellman operators folder from `data/mf_on_policy_gravity/datasets`;

### TODO......

****
**It is essential to verify that the target policies, behavior policies, datasets, and function estimates match the correct version for your intended experimental setup. For example, using the MF.OFF.G data cache in an MF.G experiment may result in entirely incorrect outcomes.**
****
