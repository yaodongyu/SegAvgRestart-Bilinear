# On the Convergence of Stochastic Extragradient for Bilinear Games using Restarted Iteration Averaging

This is the code for the [AISTATS-2022 paper](https://arxiv.org/pdf/2107.00464.pdf) "On the Convergence of Stochastic Extragradient for Bilinear Games using Restarted Iteration Averaging".


## Prerequisites
* Python
* numpy

### SEG, SEG-Avg, SEG-Avg-Restart
To reproduce the results, run ```seg_exp.py```.


```
parser = argparse.ArgumentParser()
parser.add_argument('--d', default=100, type=int, help='dimension')
parser.add_argument('--sigmaA', default=0.1, type=float, help='sigma A')
parser.add_argument('--sigmab', default=0.0, type=float, help='sigma b')
parser.add_argument('--ETA', default=0.01, type=float, help='step size')
parser.add_argument('--Iteration', default=10001, type=int, help='number of iteration')
```

## Reference
For more technical details, please check our [paper](https://arxiv.org/pdf/2107.00464.pdf). If you find this useful for your work, please consider citing
```
@article{li2021convergence,
  title={On the convergence of stochastic extragradient for bilinear games with restarted iteration averaging},
  author={Li, Chris Junchi and Yu, Yaodong and Loizou, Nicolas and Gidel, Gauthier and Ma, Yi and Roux, Nicolas Le and Jordan, Michael I},
  journal={arXiv preprint arXiv:2107.00464},
  year={2021}
}
```
