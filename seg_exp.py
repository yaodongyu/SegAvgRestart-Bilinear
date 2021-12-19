import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from matplotlib import rc
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--d', default=100, type=int, help='dimension')
    parser.add_argument('--sigmaA', default=0.1, type=float, help='sigma A')
    parser.add_argument('--sigmab', default=0.0, type=float, help='sigma b')
    parser.add_argument('--ETA', default=0.01, type=float, help='step size')
    parser.add_argument('--Iteration', default=10001, type=int, help='number of iteration')
    return parser.parse_args()

args = get_args()

def compute_norm(theta):
    return LA.norm(theta, ord='fro') ** 2

def grad_x(B, u, y):
    return B @ y + u

def grad_y(B, v, x):
    return B.transpose() @ x + v

### Set random seed.
seed = 0
np.random.seed(seed)

# Set number of iterations
N_iteration = args.Iteration
# Set the dimension
d = args.d
# Noise level
sigma_b = args.sigmab
sigma_A = args.sigmaA
# Set step size
ETA = args.ETA
# Set restart psarameter
RESTART = 100

print('iterations to run: ', N_iteration)
print('dimension: ', d)
print('step size: ', ETA)
print('sigma A: ', sigma_A)
print('sigma b: ', sigma_b)

# Generate constant vector
v_vec = np.random.rand(d) * d + 1.0
V_matrix = np.diag(v_vec)

# Generate x_star and y_star
x_star = np.random.randn(d, 1) * 0.0
y_star = np.random.randn(d, 1) * 0.0

# Init
x_init = (np.random.randn(d, 1)) * 0.1
y_init = (np.random.randn(d, 1)) * 0.1


#####################################################
################## SEG and SEG-Avg ##################
#####################################################
# Record for plot
for eta in [ETA]:
    x_1 = x_init.copy()
    y_1 = y_init.copy()
    x_avg = x_init.copy()
    y_avg = y_init.copy()

    s = 0
    for i in range(N_iteration):
        s += 1
        # sample data
        B_i = np.random.randn(1, d, d)[0] * sigma_A + V_matrix
        u_i = np.random.randn(1, d, 1)[0] * sigma_b
        v_i = np.random.randn(1, d, 1)[0] * sigma_b

        # extra gradient update
        x_psudo = x_1 - eta * grad_x(B_i, u_i, y_1)
        y_psudo = y_1 + eta * grad_y(B_i, v_i, x_1)
        x_1 = x_1 - eta * grad_x(B_i, u_i, y_psudo)
        y_1 = y_1 + eta * grad_y(B_i, v_i, x_psudo)

        # averaging
        x_avg = x_avg * ((s - 1.) / s) + x_1 * (1. / s)
        y_avg = y_avg * ((s - 1.) / s) + y_1 * (1. / s)

print('distance to optimal (SEG): ', compute_norm(x_1 - x_star) + compute_norm(y_1 - y_star))
print('distance to optimal (SEG-Avg): ', compute_norm(x_avg - x_star) + compute_norm(y_avg - y_star))


#####################################################
################## SEG-Avg-Restart ##################
#####################################################
distance_to_optimal_seg_prj_restart_list = []
plot_index_seg_restart_list = []

for restart_epoch in [RESTART]:
    print('restart_epoch: ', restart_epoch)
    x_1 = x_init.copy()
    y_1 = y_init.copy()
    x_avg = x_init.copy()
    y_avg = y_init.copy()
    eta = ETA * 1.0

    s = 0
    for i in range(1, N_iteration):
        s += 1
        # sample data
        B_i = np.random.randn(1, d, d)[0] * sigma_A + V_matrix
        u_i = np.random.randn(1, d, 1)[0] * sigma_b
        v_i = np.random.randn(1, d, 1)[0] * sigma_b

        # extra gradient update
        x_psudo = x_1 - eta * grad_x(B_i, u_i, y_1)
        y_psudo = y_1 + eta * grad_y(B_i, v_i, x_1)
        x_1 = x_1 - eta * grad_x(B_i, u_i, y_psudo)
        y_1 = y_1 + eta * grad_y(B_i, v_i, x_psudo)

        # averaging
        x_avg = x_avg * ((s - 1.) / s) + x_1 * (1. / s)
        y_avg = y_avg * ((s - 1.) / s) + y_1 * (1. / s)

        # restarting
        if s % restart_epoch == 0:
            s = 0
            x_1 = x_avg * 1.0
            y_1 = y_avg * 1.0

print('distance to optimal (SEG-Avg-Restart): ', compute_norm(x_avg - x_star) + compute_norm(y_avg - y_star))

