This repository contains data for masters thesis "Enhancing Causal Inference with Network Information". 

### Project goals and results
Accurate estimation of causal effects is crucial for making scientific conclusions. However, when network interference is present, network settings relying on methods that ignore it can lead to biased results. Although various design- and inference-based methods have emerged to address this, a gap remains in understanding how network interference biases causal estimates and how well existing methods perform. This project addresses that gap by evaluating several adjustment strategies using a simulation framework that varies network types, interference levels, and observational conditions. Focusing on exposure mapping, centrality measures, community detection, and Node2Vec embeddings, the study demonstrates that incorporating network information significantly reduces bias and false positives compared to naive estimators. Node2Vec and exposure-based methods show the most robust performance, especially in observational settings, though their effectiveness decreases under corrupted networks. These findings emphasise the need for causal inference techniques that integrate network structures and remain reliable under practical data limitations.

### Getting started
1. Clone this repository ``git clone https://github.com/pr9vina/causal_interference_study.git``
2. Navigate to the main folder: ``cd causal_interference_study``
3. (optional) Create a virtual environment: 
   ```
   python3 -m venv venv
   source venv/bin/activate
   ```

### Repository structure 

```
├── LICENSE
├── README.md
└── src                 
    ├── constants.py       # constants for generations 
    ├── data_generation    # data generation functions
    ├── data_results       # .csv with aggregated data
    ├── figs               # resulted plots for simulations
    ├── methods            # methods estimation and vizualisation functions
    ├── notebooks          # notebooks with DGP and bias estimation
    ├── poetry.lock        # requirements
    ├── pyproject.toml     # requirements
    └── utils              # utils functions
```

### Requirements 
The steps in this repository assumes that you have Python (3.12) is installed. If you don't have Python installed, you can follow this [installation guide](https://docs.python.org/3/using/index.html). In this project [poetry](https://python-poetry.org/docs/) is used for managing dependencies. You can look into `src/pyproject.toml` to see the whole list. 

To install poetry use the following:
1. ``python3 -m pip install poetry``
2. ``poetry --version  # -- verify installation``
3. ``poetry install --no-root  # -- loading dependencies``

To use poetry with jupyter notebooks:
1. ``poetry add ipykernel``
2. ``poetry run python -m ipykernel install --user --name causal_network``

### How to replicate results

##### Disclaimer 
Simulations produces files that are very large in size, the total amount of storage needed is 30 GB, GitHub is not allowing files of this size. To obtain raw data, please follow the steps as described bellow. Please also note that data generation is computationally costly, it took me a week to generate it. 

##### Study 1 

This study shows bias under Naive estimator and effectiveness of adjustments strategies for random networks under RCT settings. Every corresponding data and results figure for this part starts with 01. 

1. To obtain simulation results, run [``src/notebooks/01_01_random_dgp.py``](./src/notebooks/01_01_random_dgp.py). It calls function ``run_data_simulations()`` with set parameters for simulations. To see their description, run ``help(run_data_simulations)``.
1.2 After running the script, the resulting file will be saved as ``"sim_{network_structure_type}_random_{network_type}_influence{influence}_.pkl"``
2. To obtain aggregated results, which estimate bias and its CI, run [``src/01_02_random_results.ipynb``](./src/01_02_random_results.ipynb). You can also find them in [``01_01_all_results_.csv``](./src/data/01_01_all_results_.csv). All figures can be found in ``src/figs/``.

To simulate more realistic settings, their performance under partial observability is also simulated.

1. To obtain simulation results, run [``src/notebooks/02_01_observational_dgp.ipynb``](./src/notebooks/02_01_observational_dgp.ipynb).
1.2 After running the script, the resulting file will be saved as ``"sim_{network_structure_type}_random_{network_type}_influence{influence}_corrupt_.pkl"``
2. To obtain aggregated results, which estimate bias and its CI, run [``src/02_02_partial_obs_results.ipynb``](./src/02_02_partial_obs_results.ipynb). Please change the path to the .pkl file. You can also find final data in [``02_01_corrupt_df.csv``](./src/data_results/02_01_corrupt_df.csv). All figures can be found in [``src/figs/``](./src/figs/).

##### Study 2 
This study estimated bias under observational settings and Stochastic Block Model network.

1. To obtain simulation results, run [``src/notebooks/03_01_observationa_dgp.ipynb``](./src/notebooks/03_01_observationa_dgp.ipynb)
1.2 After running the script, the resulting file will be saved as ``"sim_{network_structure_type}_individ_and_neigbhour_sbm_influence{influence}_.pkl"``
2. To obtain aggregated results, which estimate bias and its CI, run [``src/03_02_observational_results.ipynb``](./src/03_02_observational_results.ipynb). You can also find them in [``03_01_observational_df.csv``](./src/data_results/03_01_observational_df.csv). All figures can be found in [``src/figs/``](./src/figs/).

## Permissions

This research archive is openly published on GitHub, https://github.com/pr9vina/causal_interference_study.git under MIT license.

## Ethical approval 
The study is approved by the Ethical Review Board of the Faculty of Social and Behavioural Sciences of Utrecht University. The approval is based on the documents sent by the researchers as requested in the form of the Ethics committee and filed under number 24-2019. The approval is valid through 01 June 2025. The approval of the Ethical Review Board concerns ethical aspects, as well as data management and privacy issues (including the GDPR).

## Questions?
For additional information you can reach me out by pauline826 (at) gmail (dot) com.
