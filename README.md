# EDCA-PR-DLF

This repository provides a compact, representative single-well implementation of the EDCA-PR-DLF workflow used in the manuscript. It was extracted and refactored from the research code for method review and reproduction.

## Overview

The workflow is:

```text
production data
-> active positive-rate observations
-> production-peak cropping
-> chronological 80% training / 20% validation split
-> train-only Spline smoothing and hyperbolic DCA fitting
-> residual construction and EEMD feature extraction
-> GRU residual prediction
-> production reconstruction and evaluation
```

The residual sign convention is:

```text
residual = q_DCA - q_actual
q_pred   = q_DCA - residual_pred
```

The current protocol has no independent test split. Validation uses observed historical windows for one-step diagnostic prediction rather than strict recursive forecasting.

## Repository Structure

```text
.
|-- README.md
|-- LICENSE
|-- requirements.txt
|-- config.yaml
|-- run_example.py
|-- .gitignore
|-- data/
|   `-- example_well.csv
`-- src/
    |-- __init__.py
    |-- config.py
    |-- data_processing.py
    |-- dca.py
    |-- decomposition.py
    |-- features.py
    |-- dataset.py
    |-- model.py
    |-- training.py
    |-- evaluation.py
    `-- utils.py
```

`run_example.py` is the end-to-end entry point. The `src/` directory contains data processing, DCA, EEMD, feature construction, GRU training, prediction, and evaluation modules.

## Data and Model

`data/example_well.csv` contains a partial A04 dataset with 135 records. Review the data before public release and replace it with data authorized for sharing when necessary.

The model uses a 36-step historical window with 15 features: residual, first and second residual differences, DCA trend, time, liquid rate, water rate, gas rate, bottom-hole pressure, drawdown, operation time, and four EEMD components.

```text
X: [batch_size, 36, 15]
y: [batch_size]
```

Spline smoothing, DCA fitting, feature scaling, and EEMD fitting use training data only. The GRU predicts the residual, and production is reconstructed using the formula shown above.

## Usage

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the A04 example:

```bash
python run_example.py
```

For a quick environment check:

```bash
python run_example.py --epochs 2 --eemd-trials 2
```

The data path and model parameters can be changed in `config.yaml`.

## Output

Running the example creates:

```text
results/prediction_results.csv
results/metrics.json
results/training_loss.csv
results/resolved_config.json
checkpoints/best_model.pt
```

The reported validation metrics are RMSE, MAE, MAPE, and R2. Generated results and model weights are excluded from version control by `.gitignore`.

## Citation and License

Please replace this paragraph with the final manuscript citation before publication.

This project is released under the MIT License. See `LICENSE`.
