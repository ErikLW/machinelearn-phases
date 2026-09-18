Training with `eval/run_nn_optim.py` saves the final weights and biases to
`optimized_nn_params/model_params.npz`, overwriting that file on subsequent runs.
The archive contains `n_layers`, `W_0`, `b_0`, and corresponding arrays for
each remaining layer. It does not contain training data or loss history.

From the repository root, restore it with:

```python
import machinelearn_phases as mlp

params = mlp.load_model_params("data/model_params.npz")
probabilities = mlp.classifier(params, x)
```
