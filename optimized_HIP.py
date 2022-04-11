import heterocl as hcl
import numpy as np
import random
import time
import math

import scipy.io

#################### PROBLEM INITIALIZATION ####################
precomputed_info = scipy.io.loadmat('V.mat')
phi = precomputed_info['ttrValue_obs']
Grid = precomputed_info['xs_whole']
print(Grid.shape)

xPrecision = [0.2, 0.2, 2 * math.pi/15, 0.2]

uPrecision = [0.5, 0.5]
Ua = np.linspace(-1, 1, int(2/uPrecision[0]) + 1)
Uw = np.linspace(-1.5, 1.5, int(3/uPrecision[1]) + 1)

x1 = Grid[:, 0, 0, 0, 0]
#print(x1)
x2 = Grid[0, :, 0, 0, 1]
#print(x2)
x3 = Grid[0, 0, :, 0, 2]
#print(x3)
x4 = Grid[0, 0, 0, :, 3]
#print(x4)

# print(Ua)
# print(Uw)

data = scipy.io.loadmat('synthetic_trajectory.mat')

w = data['U_synth'][:, 0]
a = data['U_synth'][:, 1]
X0 =  data['X_synth'][0,:]

final_time = 20
#print(X0)

def dynamic(X, a, w, dt):
    x = X[0]
    y = X[1]
    theta = X[2]
    v = X[3]
    xo = x + v*np.cos(theta)*dt+0*a
    yo = y + v*np.sin(theta)*dt+0*a
    thetao = theta+ w * dt
    vo = v+ a*dt

    return np.array([xo, yo, thetao, vo])

delta_t = 0.2

# Initialize the observed state
X = np.zeros((final_time, 4))
X[0, :] = dynamic(X0, a[0], w[0], delta_t)
#print(X[0, :])

for i in range(final_time-1):
    X[i+1, :] = dynamic(X[i, :], a[i+1], w[i+1], delta_t)

#################### PARTICLES INITIALIZATION ####################
num_goal_particles = 20
num_state_particles = 200

xmin = np.floor(min(np.min(X[0, :]), np.min(X[1, :])))
xmax = np.ceil(min(np.max(X[0, :]), np.max(X[1, :])))
goal_particles = np.random.rand(num_goal_particles, 4)*(xmax - xmin) + xmin

# Initialize first particle
goal_particles[0, :] = data['G_synth']
# Goal particles' angle
goal_particles[:, 2] = np.random.rand(num_goal_particles)*2*math.pi - math.pi
goal_particles[:, 3] = 0

# State particles
state_particles = X0 * np.ones((num_state_particles, 4)) + 0.01 * np.random.rand(num_state_particles, 4)

# Beta, gamma inits
beta = np.array([0.1, 10])
gamma = np.array([0.09, 0.99])

# Weight of beta, gamma value
P_beta = (1/beta.shape[0]) * np.ones((beta.shape[0]))
#print(P_beta)
P_gamma = (1/gamma.shape[0]) * np.ones((gamma.shape[0]))
#print(P_gamma)

# Goal particles weight
P_goal = (1/goal_particles.shape[0]) * np.ones((goal_particles.shape[0]))
#print(P_goal)

len_u_comb = Ua.shape[0] * Uw.shape[0]

# P(u | x; beta, goal, gamma), assume that x is observed
Pu = np.zeros((len_u_comb, gamma.shape[0], beta.shape[0], goal_particles.shape[0]),
              dtype=np.float32) + 1.5
Pu = Pu* 10.2389
#print(Pu)
actions = []
for i in Ua:
    for j in Uw:
        actions.append((i,j))
u_comb = np.array(actions)

hcl.init()
hcl.config.init_dtype = hcl.Float(32)

#print(u_comb)
u_probDist = hcl.asarray(Pu)
#print(u_probDist)

# Hardcode these numbers for now
my_bounds     = np.array([[-8.0, 8.0],[-8.0, 8.0],[-math.pi, math.pi], [-0.2, 0.2]])
my_ptsEachDim = np.array([161, 161, 31, 25])


# HeteroCL code
# def optimizedHIP_graph(Pu, x_1, x_2, x_3, x_4, beta, gamma, V,
#                        goal_particles, U_comb, target=None):
def optimizedHIP_graph(Pu, goal_particles, U_comb, my_bounds, my_ptsEachDim, V, target=None):
    PU = hcl.placeholder(np.shape(Pu), name="PU", dtype=hcl.Float())
    # Beta = hcl.placeholder(np.shape(beta), name="Beta", dtype=hcl.Float())
    # Gamma = hcl.placeholder(np.shape(gamma), name="Gamma", dtype=hcl.Float())
    TTR = hcl.placeholder(np.shape(V), name="TTR", dtype=hcl.Float())
    G_particles = hcl.placeholder(np.shape(goal_particles), name="G_particles", dtype=hcl.Float())
    Ucomb = hcl.placeholder(np.shape(U_comb), name="Ucomb", dtype=hcl.Float())
    bounds = hcl.placeholder(np.shape(my_bounds), name="bounds", dtype=hcl.Float())
    ptsEachDim = hcl.placeholder(np.shape(my_ptsEachDim), name="ptsEachDim", dtype=hcl.Float())
    iVals      = hcl.placeholder(tuple([4]), name="iVals", dtype=hcl.Float())

    X = hcl.placeholder((4,), name="Ucomb", dtype=hcl.Float())

    # x1 = hcl.placeholder((x_1.shape[0],), name="x1", dtype=hcl.Float())
    # x2 = hcl.placeholder((x_2.shape[0],), name="x2", dtype=hcl.Float())
    # x3 = hcl.placeholder((x_3.shape[0],), name="x3", dtype=hcl.Float())
    # x4 = hcl.placeholder((x_4.shape[0],), name="x4", dtype=hcl.Float())
    
    # def graph_create(u_prob_dist, X, x1, x2, x3, x4, beta_bel, gamma_bel,
    #                  TTR, G_particles, Ucomb):
    def graph_create(u_prob_dist, X, G_particles, Ucomb, bounds, ptsEachDim, iVals, V):
        def evaluatePandQ(i,j,k,l):
            iVals1 = hcl.scalar(0, "iVals1")
            iVals2 = hcl.scalar(0, "iVals2")
            iVals3 = hcl.scalar(0, "iVals3")
            iVals4 = hcl.scalar(0, "iVals4")

            x_tp1 = hcl.scalar(0, "x_tp1")
            y_tp1 = hcl.scalar(0, "y_tp1")
            theta_tp1 = hcl.scalar(0, "theta_tp1")
            v_tp1 = hcl.scalar(0, "v_tp1")

            x_tp1[0] = X[0] + X[3] * hcl.cos(X[2]) * 0.2
            y_tp1[0] = X[1] + X[3] * hcl.sin(X[2]) * 0.2
            theta_tp1[0] = X[2] + Ucomb[i, 1] * 0.2
            v_tp1[0]    = X[3] + Ucomb[i, 0] * 0.2

            x_tp1[0] = (x_tp1[0] - G_particles[l, 0]) * hcl.cos(G_particles[l, 2]) + \
                       (x_tp1[1] - G_particles[l, 1]) * hcl.sin(G_particles[l, 2])
            y_tp1[0] = (x_tp1[0] - G_particles[l, 0]) * (-hcl.sin(G_particles[l, 2])) + \
                       (x_tp1[1] - G_particles[l, 1]) * hcl.cos(G_particles[l, 2])
            theta_tp1[0] = theta_tp1[0] - G_particles[l, 2]

            with hcl.if_(theta_tp1[0] < -3.14159):
                theta_tp1[0] = theta_tp1[0] + 2 * 3.14159
            with hcl.if_(theta_tp1[0] > 3.14159):
                theta_tp1[0] = theta_tp1[0] - 2 * 3.14159

            # Now convert the next state to grid index
            iVals[0] = ((x_tp1[0] - bounds[0,0]) / (bounds[0,1] - bounds[0,0])) * (ptsEachDim[0] - 1)
            iVals[1] = ((y_tp1[0] - bounds[1,0]) / (bounds[1,1] - bounds[1,0])) * (ptsEachDim[1] - 1)
            iVals[2] = ((theta_tp1[0] - bounds[2,0]) / (bounds[2,1] - bounds[2,0])) * (ptsEachDim[2] - 1)
            iVals[3] = ((v_tp1[0] - bounds[3,0]) / (bounds[3,1] - bounds[3,0])) * (ptsEachDim[3] - 1)
            # NOTE: add 0.5 to simulate rounding
            iVals[0] = hcl.cast(hcl.Int(), iVals[0] + 0.5)
            iVals[1] = hcl.cast(hcl.Int(), iVals[1] + 0.5)
            iVals[2] = hcl.cast(hcl.Int(), iVals[2] + 0.5)
            iVals[3] = hcl.cast(hcl.Int(), iVals[3] + 0.5)

            #stateToIndex()

            u_prob_dist[i,j,k,l] = V[iVals[0],iVals[1], iVals[2], iVals[3]] * 60 + 90.9201

        # def stateToIndex(sVals, iVals):
        #     iVals[0] = ((sVals[0] - bounds[0,0]) / (bounds[0,1] - bounds[0,0])) * (ptsEachDim[0] - 1)
        #     iVals[1] = ((sVals[1] - bounds[1,0]) / (bounds[1,1] - bounds[1,0])) * (ptsEachDim[1] - 1)
        #     iVals[2] = ((sVals[2] - bounds[2,0]) / (bounds[2,1] - bounds[2,0])) * (ptsEachDim[2] - 1)
        #     iVals[3] = ((sVals[3] - bounds[3,0]) / (bounds[3,1] - bounds[3,0])) * (ptsEachDim[3] - 1)
        #     # NOTE: add 0.5 to simulate rounding
        #     iVals[0] = hcl.cast(hcl.Int(), iVals[0] + 0.5)
        #     iVals[1] = hcl.cast(hcl.Int(), iVals[1] + 0.5)
        #     iVals[2] = hcl.cast(hcl.Int(), iVals[2] + 0.5)
        #     iVals[3] = hcl.cast(hcl.Int(), iVals[3] + 0.5)

        # i: Ucomb index
        # j: gamma idx, k: beta index, l: goal particle index,
        with hcl.Stage("updatePu"):
            with hcl.for_(0, u_prob_dist.shape[0], name="i") as i:
                with hcl.for_(0, u_prob_dist.shape[1], name="j") as j:
                    with hcl.for_(0, u_prob_dist.shape[2], name="k") as k:
                        with hcl.for_(0, u_prob_dist.shape[3], name="l") as l:
                            evaluatePandQ(i,j,k,l)
                            #result = hcl.update(u_prob_dist, lambda i, j, k, l: evaluatePandQ(i,j,k,l), "B")
    s = hcl.create_schedule([PU, X, G_particles, Ucomb, bounds, ptsEachDim, iVals, TTR], graph_create)
    s_H = graph_create.updatePu
    s[s_H].parallel(s_H.i)

    return (hcl.build(s))

f = optimizedHIP_graph(Pu, goal_particles, u_comb, my_bounds, my_ptsEachDim, phi)

x = np.zeros((4,))
#print(x)
X = hcl.asarray(x)
#print(X)
g_particles = hcl.asarray(goal_particles)
u_combs = hcl.asarray(u_comb)

bounds = hcl.asarray(my_bounds)
ptsEachDim = hcl.asarray(my_ptsEachDim)
V = hcl.asarray(phi)
idx = hcl.asarray(np.zeros(4))

import time

for i in range(20):
    t_s = time.time()
    f(u_probDist, X, g_particles, u_combs, bounds, ptsEachDim, idx, V)
    t_e = time.time()
    X = X.asnumpy() + np.random.rand(4) * 2
    X = hcl.asarray(X)
    print("Took        ", t_e - t_s, " seconds")



