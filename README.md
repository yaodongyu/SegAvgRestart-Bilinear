# Accelerating Stochastic Extragradient for Bilinear Games using Restarted Iteration Averaging

This is the code for the "Accelerating Stochastic Extragradient for Bilinear Games using Restarted Iteration Averaging".


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