This repository contains data for masters thesis "Enhancing Causal Inference with Network Information". 

### Project goals and results
Accurate estimation of causal effects is crucial for making scientific conclusions. However, when network interference is present, network settings relying on methods that ignore it can lead to biased results. Although various design- and inference-based methods have emerged to address this, a gap remains in understanding how network interference biases causal estimates and how well existing methods perform. This project addresses that gap by evaluating several adjustment strategies using a simulation framework that varies network types, interference levels, and observational conditions. Focusing on exposure mapping, centrality measures, community detection, and Node2Vec embeddings, the study demonstrates that incorporating network information significantly reduces bias and false positives compared to naive estimators. Node2Vec and exposure-based methods show the most robust performance, especially in observational settings, though their effectiveness decreases under corrupted networks. These findings emphasise the need for causal inference techniques that integrate network structures and remain reliable under practical data limitations.

### Requirements 

I use [poetry](https://python-poetry.org/docs/) for managing dependencies. You can look into `src/pyproject.toml` to see the whole list. To install poetry use the following:

1. ``python3 -m pip install poetry``
2. ``poetry --version  # -- verify installation``
3. ``poetry install --no-root  # -- loading dependencies``

To use poetry with jupyter notebooks:
1. ``poetry add ipykernel``
2. ``poetry run python -m ipykernel install --user --name causal_network``


### How to replicate results

<!-- Something about that data is simulation-based -->
Note! Raw files from simulations are extremly large, it unfeasible to store them on github, in this repository I store aggregated data. You can reach out to me for obtaining raw data.

All constants for simulations could be found in ``src/constants.py`` and can be changed if needed.

##### Study 1 

This study shows bias under Naive estimator and effectiveness of adjustments strategies for random networks under RCT settings. Every corresponding data and results figure for this part starts with 01. 

1. To obtain simulation results, run ``src/notebooks/01_01_random_dgp.py``. 
   <!-- тут добавить, что здесь конкретно происходит на этом этапе -->
2. To obtain aggregated results, which estimate bias and its CI, run ``src/01_02_random_results.ipynb``. You can also find them in ``01_01_all_results_.csv``. All figures can be found in ``src/figs/``.

To simulate more realistic settings, their performance under partial observability is also simulated.

1. To obtain simulation results, run ``src/notebooks/02_01_observational_dgp.ipynb``. 
2. To obtain aggregated results, which estimate bias and its CI, run ``src/02_02_partial_obs_results.ipynb``. You can also find them in ``02_01_corrupt_df.csv``. All figures can be found in ``src/figs/``.


##### Study 2 
This study estimated bias under observational settings and Stochastic Block Model network.

1. To obtain simulation results, run [``src/notebooks/03_01_observationa_dgp.ipynb``](./src/notebooks/03_01_observationa_dgp.ipynb)
2. To obtain aggregated results, which estimate bias and its CI, run ``src/03_02_observational_results.ipynb``. You can also find them in ``03_01_observational_df.csv``. All figures can be found in ``src/figs/``.

## Licens and terms of use

This repository uses dual licensing for code and data:
- **Code** is licensed under the [GNU General Public License v3.0 (GPLv3)](https://www.gnu.org/licenses/gpl-3.0.html).
- **Data** is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/).

See the [LICENSE](./LICENSE) file for full details.

## Ethical approval 
The study is approved by the Ethical Review Board of the Faculty of Social and Behavioural Sciences of Utrecht University. The approval is based on the documents sent by the researchers as requested in the form of the Ethics committee and filed under number 24-2019. The approval is valid through 01 June 2025. The approval of the Ethical Review Board concerns ethical aspects, as well as data management and privacy issues (including the GDPR).

## Questions?
For additional information you can reach me out by pauline826 (at) gmail (dot) com.