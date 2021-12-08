DNN-MYK
==============

This is a collection of Python scripts for performing the proposed method **DNN-MYK** for controlled response selection.
* demo_code.py contains code for implementing the method on simulated data.

### Parameter Descriptions in demo_code.py. 

| Parameters | Description |
| --- | --- |
| n | Sample size |
| r | Total number of responses|
| p| Total number of features|
| m| Total number of important responses|
| rho| Correlation in X|
| betaValue| The absolute values of the coefficients|
| t| Sparsity level in the coefficients beta vector|

We vary the key parameter values with different levels below in both linear and nonlinear settings. 
| Parameters | Levels |
| --- | --- |
| Number of responses (r) | 1000, 1500, 2000, 2500, 3000 |
| Sample size (n) | 300, 400, 500, 600 |
| Correlation (rho) in X | 0.1, 0.3, 0.5, 0.7, 0.9 |
| Sparsity level (t) in coefficients beta | 0.1, 0.3, 0.5, 0.7, 0.9 |

By varying the key parameters and run demo.py in the linear setting, we show the results in the following figure under a linear setting. 
![Power and FDR in Linear Settings](https://github.com/flahertylab/deepYknockoff/blob/master/figs/linear_power_combine.pdf)

## Reference

Identification of Significant Gene Expression Changes in Multiple Perturbation Experiments using Knockoffs
