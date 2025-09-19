# `README.md`

# A High-Resolution Framework for Analyzing Transatlantic Monetary Spillovers

<!-- PROJECT SHIELDS -->
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![Type Checking: mypy](https://img.shields.io/badge/type_checking-mypy-blue)](http://mypy-lang.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-%23F37626.svg?style=flat&logo=Jupyter&logoColor=white)](https://jupyter.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2509.13578-b31b1b.svg)](https://arxiv.org/abs/2509.13578)
[![Year](https://img.shields.io/badge/Year-2025-purple)](https://github.com/chirindaopensource/between_transatlantic_monetary_disturbances)
[![Discipline](https://img.shields.io/badge/Discipline-International%20Macroeconomics-blue)](https://github.com/chirindaopensource/between_transatlantic_monetary_disturbances)
[![Methodology](https://img.shields.io/badge/Methodology-BVAR%20%7C%20Local%20Projection%20%7C%20HF%20Identification-orange)](https://github.com/chirindaopensource/between_transatlantic_monetary_disturbances)
[![Data Source](https://img.shields.io/badge/Data-High--Frequency%20Futures%20%26%20Macro-lightgrey)](https://github.com/chirindaopensource/between_transatlantic_monetary_disturbances)
[![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=flat&logo=numpy&logoColor=white)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-%23025596?style=flat&logo=scipy&logoColor=white)](https://scipy.org/)
[![Statsmodels](https://img.shields.io/badge/statsmodels-1A568C.svg?style=flat)](https://www.statsmodels.org/stable/index.html)
[![PyYAML](https://img.shields.io/badge/PyYAML-4B5F6E.svg?style=flat)](https://pyyaml.org/)
[![Joblib](https://img.shields.io/badge/joblib-2F72A4.svg?style=flat)](https://joblib.readthedocs.io/en/latest/)
[![TQDM](https://img.shields.io/badge/tqdm-FFC107.svg?style=flat)](https://tqdm.github.io/)
[![Analysis](https://img.shields.io/badge/Analysis-Monetary%20Policy-brightgreen)](https://github.com/chirindaopensource/between_transatlantic_monetary_disturbances)
[![Framework](https://img.shields.io/badge/Framework-Bayesian-blueviolet)](https://github.com/chirindaopensource/between_transatlantic_monetary_disturbances)
[![Identification](https://img.shields.io/badge/Identification-Sign%20Restrictions-red)](https://github.com/chirindaopensource/between_transatlantic_monetary_disturbances)
[![Validation](https://img.shields.io/badge/Validation-Out--of--Sample-yellow)](https://github.com/chirindaopensource/between_transatlantic_monetary_disturbances)
[![Robustness](https://img.shields.io/badge/Robustness-Sensitivity%20Analysis-lightgrey)](https://github.com/chirindaopensource/between_transatlantic_monetary_disturbances)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

--

**Repository:** `https://github.com/chirindaopensource/between_transatlantic_monetary_disturbances`

**Owner:** 2025 Craig Chirinda (Open Source Projects)

This repository contains an **independent**, professional-grade Python implementation of the research methodology from the 2025 paper entitled **"In-between Transatlantic (Monetary) Disturbances"** by:

*   Santiago Camara
*   Jeanne Aublin

The project provides a complete, end-to-end computational framework for identifying source-dependent monetary policy shocks and analyzing their international spillovers. It delivers a modular, auditable, and extensible pipeline that replicates the paper's entire workflow: from rigorous high-frequency data processing and validation, through the sophisticated rotational decomposition for shock identification, to the estimation of BVAR and Local Projection models and a comprehensive suite of robustness checks.

## Table of Contents

- [Introduction](#introduction)
- [Theoretical Background](#theoretical-background)
- [Features](#features)
- [Methodology Implemented](#methodology-implemented)
- [Core Components (Notebook Structure)](#core-components-notebook-structure)
- [Key Callable: execute_full_study_pipeline](#key-callable-execute_full_study_pipeline)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Input Data Structure](#input-data-structure)
- [Usage](#usage)
- [Output Structure](#output-structure)
- [Project Structure](#project-structure)
- [Customization](#customization)
- [Contributing](#contributing)
- [Recommended Extensions](#recommended-extensions)
- [License](#license)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

## Introduction

This project provides a Python implementation of the methodologies presented in the 2025 paper "In-between Transatlantic (Monetary) Disturbances." The core of this repository is the iPython Notebook `between_transatlantic_monetary_disturbances_draft.ipynb`, which contains a comprehensive suite of functions to replicate the paper's findings, from initial data validation to the final generation and analysis of impulse response functions and robustness tests.

The paper addresses a key question in international macroeconomics: How do monetary policy shocks from major economic blocs (the U.S. and the Euro Area) propagate to a smaller, open economy (Canada), and do the transmission channels differ? This codebase operationalizes the paper's advanced approach, allowing users to:
-   Rigorously validate and cleanse high-frequency financial data and low-frequency macroeconomic data.
-   Identify "pure" monetary policy shocks, purged of central bank information effects, using a high-frequency identification strategy with sign restrictions.
-   Estimate the dynamic effects of these shocks using both Bayesian Vector Autoregressions (BVAR) and Local Projections (LP).
-   Conduct a full suite of robustness checks to validate the stability of the findings across different identification schemes, sample periods, and model specifications.
-   Systematically investigate specific transmission channels (e.g., trade, financial) by running augmented models.

## Theoretical Background

The implemented methods are grounded in modern time-series econometrics and international finance.

**1. High-Frequency Identification with Sign Restrictions:**
Standard event studies can be confounded by the "information effect," where a central bank's policy action reveals private information about the economic outlook. To solve this, the paper uses the methodology of Jarociński & Karadi (2020). Raw surprises in interest rates ($s^{rate}$) and equity prices ($s^{equity}$) are assumed to be linear combinations of two structural shocks: a pure monetary policy shock ($\varepsilon^{MP}$) and an information shock ($\varepsilon^{INFO}$).
$$ \begin{bmatrix} s^{rate}_t \\ s^{equity}_t \end{bmatrix} = A \begin{bmatrix} \varepsilon^{MP}_t \\ \varepsilon^{INFO}_t \end{bmatrix} $$
The identification of the mixing matrix `A` is achieved by finding all rotations of an initial Cholesky decomposition that satisfy a set of theoretical sign restrictions on the impulse responses (e.g., a contractionary MP shock must raise rates and lower equity prices).

**2. Bayesian Vector Autoregression (BVAR):**
The primary workhorse model is a VAR-X, where the identified shocks are treated as exogenous variables. The model for a vector of endogenous variables $Y_t$ is:
$$ Y_t = c + \sum_{i=1}^{p} B_i Y_{t-i} + \Gamma_1 s_t^{ECB} + \Gamma_2 s_t^{Fed} + D_t + e_t $$
The model is estimated using Bayesian methods with a Normal-Wishart prior and a Minnesota-style specification for the prior hyperparameters. Inference is conducted by drawing from the posterior distribution using a Gibbs sampler.

**3. Local Projections (LP):**
As a robustness check, the impulse responses are also estimated using the Local Projection method of Jordà (2005). This involves running a separate regression for each forecast horizon `h`:
$$ y_{k, t+h} = \beta_h^{Shock} s_t^{Shock} + \text{controls}_t + \epsilon_{t+h} $$
The sequence of estimated coefficients $\{\hat{\beta}_h^{Shock}\}_{h=0}^H$ forms the impulse response function. This method is robust to misspecification but less efficient than a VAR. Inference requires HAC (Newey-West) standard errors.

## Features

The provided iPython Notebook (`between_transatlantic_monetary_disturbances_draft.ipynb`) implements the full research pipeline, including:

-   **Modular, Multi-Phase Architecture:** The entire pipeline is broken down into 17 distinct, modular tasks, from data validation to final robustness checks.
-   **Configuration-Driven Design:** All methodological and computational parameters are managed in an external `config.yaml` file, allowing for easy customization without code changes.
-   **Professional-Grade Data Pipeline:** A comprehensive validation, quality assessment, and cleansing suite for both high-frequency and low-frequency data, including robust handling of timezones and DST.
-   **High-Fidelity Shock Identification:** A precise, vectorized implementation of the rotational-angle decomposition method.
-   **Robust BVAR Estimation:** A complete Gibbs sampler for a BVAR with a Normal-Wishart prior, including intra-run convergence diagnostics.
-   **Complete Local Projections Estimator:** A full implementation of the LP method with HAC-robust standard errors.
-   **Advanced Robustness Toolkit:**
    -   A framework for testing alternative identification schemes (Poor Man's Sign Restriction).
    -   A parallelized framework for quantifying identification uncertainty by integrating over all admissible rotations.
    -   A framework for testing sensitivity to alternative sample periods (pre-GFC, pre-COVID).
    -   A framework for testing sensitivity to estimation choices (prior hyperparameters, lag length).

## Methodology Implemented

The core analytical steps directly implement the methodology from the paper:

1.  **Data Validation & Preprocessing (Tasks 1-3):** Ingests and rigorously validates all raw data and the `config.yaml` file, performs a deep data quality audit, and produces clean, analysis-ready data streams.
2.  **Shock Identification (Tasks 4-6):** Defines event windows, extracts high-frequency prices, calculates raw surprises, and performs the rotational decomposition to identify structural shocks.
3.  **Model Preparation (Tasks 7-8):** Aggregates the identified shocks to a monthly frequency and assembles the final, transformed dataset for econometric modeling.
4.  **Estimation (Tasks 9-11):** Sets up and estimates the baseline BVAR via Gibbs sampling and the Local Projections model via OLS with HAC errors.
5.  **Results & Validation (Tasks 12-14):** Calculates impulse response functions from the BVAR posterior and runs a full suite of in-sample and out-of-sample validation tests.
6.  **Robustness Analysis (Tasks 16-17):** Orchestrates the entire suite of robustness checks on the identification and estimation methods.

## Core Components (Notebook Structure)

The `between_transatlantic_monetary_disturbances_draft.ipynb` notebook is structured as a logical pipeline with modular orchestrator functions for each of the major tasks. All functions are self-contained, fully documented with type hints and docstrings, and designed for professional-grade execution.

## Key Callable: execute_full_study_pipeline

The central function in this project is `execute_full_study_pipeline`. It orchestrates the entire analytical workflow, providing a single entry point for running the baseline study and all associated robustness checks.

```python
def execute_full_study_pipeline(
    equity_tick_df: pd.DataFrame,
    rate_tick_df: pd.DataFrame,
    macro_df: pd.DataFrame,
    announcement_df: pd.DataFrame,
    target_market: str,
    study_config: Dict[str, Any],
    run_identification_robustness: bool = True,
    run_estimation_robustness: bool = True
) -> Dict[str, Any]:
    """
    Executes the entire research study, including the main analysis and all robustness checks.
    """
    # ... (implementation is in the notebook)
```

## Prerequisites

-   Python 3.9+
-   Core dependencies: `pandas`, `numpy`, `scipy`, `statsmodels`, `pyyaml`, `tqdm`, `joblib`, `pandas_market_calendars`.

## Installation

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/chirindaopensource/between_transatlantic_monetary_disturbances.git
    cd between_transatlantic_monetary_disturbances
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Python dependencies:**
    ```sh
    pip install pandas numpy scipy statsmodels pyyaml tqdm joblib pandas_market_calendars
    ```

## Input Data Structure

The pipeline requires four `pandas.DataFrame`s and a configuration file as input. Mock data generation functions are provided in the main notebook to create valid examples for testing.
1.  **`equity_tick_df` / `rate_tick_df`:** Must contain columns `['timestamp_micros_utc', 'price', 'volume', 'type']`.
2.  **`macro_df`:** A long-format DataFrame with columns `['date', 'source_series_id', 'country', 'variable_name', 'value_raw']`.
3.  **`announcement_df`:** Must contain columns `['event_id', 'central_bank', 'announcement_date_local', 'announcement_time_local', 'local_timezone']`.

## Usage

The `between_transatlantic_monetary_disturbances_draft.ipynb` notebook provides a complete, step-by-step guide. The core workflow is:

1.  **Prepare Inputs:** Load your four raw `pandas.DataFrame`s. Ensure the `config.yaml` file is present in the same directory.
2.  **Execute Pipeline:** Call the grand orchestrator function.

    ```python
    # This single call runs the entire project.
    final_results = execute_full_study_pipeline(
        equity_tick_df=my_equity_data,
        rate_tick_df=my_rate_data,
        macro_df=my_macro_data,
        announcement_df=my_announcement_data,
        target_market='CAN',
        study_config=my_config_dict,
        run_identification_robustness=False,  # Set to True for the full analysis
        run_estimation_robustness=False
    )
    ```
3.  **Inspect Outputs:** The returned `final_results` dictionary contains all generated artifacts, including intermediate data, final IRFs, and validation reports.

## Output Structure

The `execute_full_study_pipeline` function returns a single, comprehensive dictionary containing all generated artifacts, structured by analytical phase. Key outputs include:
-   `benchmark_run`: The results of the main analysis.
    -   `phase_2_identification['structural_shocks']`: The identified monthly shock series.
    -   `phase_3_model_prep['analysis_ready_df']`: The final dataset used for estimation.
    -   `phase_5_results['bvar_irfs']`: The final impulse response functions from the BVAR.
    -   `phase_5_results['model_validation_reports']`: The full suite of diagnostic reports.
-   `identification_robustness_suite`: (If run) Contains the results of the PMSR, rotational uncertainty, and sub-sample analyses.
-   `estimation_robustness_suite`: (If run) Contains the results of the prior, lag, and specification sensitivity analyses.

## Project Structure

```
between_transatlantic_monetary_disturbances/
│
├── between_transatlantic_monetary_disturbances_draft.ipynb   # Main implementation notebook
├── config.yaml                                               # Master configuration file
├── requirements.txt                                          # Python package dependencies
├── LICENSE                                                   # MIT license file
└── README.md                                                 # This documentation file
```

## Customization

The pipeline is highly customizable via the `config.yaml` file. Users can easily modify all methodological parameters, such as BVAR lags, prior hyperparameters, MCMC settings, and window definitions, without altering the core Python code.

## Contributing

Contributions are welcome. Please fork the repository, create a feature branch, and submit a pull request with a clear description of your changes. Adherence to PEP 8, type hinting, and comprehensive docstrings is required.

## Recommended Extensions

Future extensions could include:
-   **Visualization Module:** Creating a function that takes the final IRF results and automatically generates publication-quality plots that replicate the figures in the paper.
-   **Automated Reporting:** Building a module that uses the generated results and validation reports to automatically create a full PDF or HTML summary report of the findings.
-   **Alternative Priors:** Implementing other BVAR prior structures, such as the Independent Normal-Wishart prior or stochastic volatility.
-   **Structural VAR Identification:** Adding modules for other SVAR identification schemes, such as Cholesky or long-run restrictions, for comparison.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Citation

If you use this code or the methodology in your research, please cite the original paper:

```bibtex
@article{camara2025inbetween,
  title={{In-between Transatlantic (Monetary) Disturbances}},
  author={Camara, Santiago and Aublin, Jeanne},
  journal={arXiv preprint arXiv:2509.13578},
  year={2025}
}
```

For the implementation itself, you may cite this repository:
```
Chirinda, C. (2025). A High-Resolution Framework for Analyzing Transatlantic Monetary Spillovers.
GitHub repository: https://github.com/chirindaopensource/between_transatlantic_monetary_disturbances
```

## Acknowledgments

-   Credit to **Santiago Camara and Jeanne Aublin** for their foundational research, which forms the entire basis for this computational replication.
-   This project is built upon the exceptional tools provided by the open-source community. Sincere thanks to the developers of the scientific Python ecosystem, including **Pandas, NumPy, SciPy, Statsmodels, and Joblib**, whose work makes complex computational analysis accessible and robust.

--

*This README was generated based on the structure and content of `between_transatlantic_monetary_disturbances_draft.ipynb` and follows best practices for research software documentation.*
