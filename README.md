# Linear Regression: Optimization and Feature Engineering

A multivariate linear regression project comparing optimization methods and applying feature engineering to taxi fare prediction.

## What’s here

- `src/lr_sgd.py` — mini-batch stochastic gradient descent.
- `src/lr_firstorder.py` — Adam, a first-order adaptive optimization method.
- `src/lr_secondorder.py` — a Newton-style update that solves a Hessian system for the weights.
- `src/feature_engineering.py` — a taxi feature pipeline with geographic, temporal, airport, and distance features. Scaling statistics are learned from the training data and reused for validation/test data.
- `tests/` — checks for optimizer behavior, feature engineering, and comparison on the taxi dataset.
- `results/validation_loss_comparison.json` — saved validation metrics from an earlier run.
- `data/` — local taxi fare CSV files used by the data tests.

The first-order implementation uses bias-corrected first and second gradient moments, as in Adam. The second-order implementation computes a Newton-style update by solving a Hessian system for the weights.

## Run

Run the test suite from the repository root:

```bash
python -m unittest discover -s tests -v
```

The data-based tests expect `data/taxifares.csv` to be present. Confirm the dataset’s source and terms before redistributing the CSV files in a public repository.

## Example

```python
from src.lr_firstorder import AdamRegression

model = AdamRegression(learning_rate=0.1, epochs=100, batch_size=64, random_state=42)
train_losses, validation_losses = model.fit(X_train, y_train, X_val, y_val)
predictions = model.predict(X_val)
```

The `FirstOrderMethodRegression` and `SecondOrderMethodRegression` names are also available as compatibility aliases.
