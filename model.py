import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

def divnorm(stim, choices, input_param_names, input_params, dt=1/20, nunits=2):
    """divisive normalization model

    Parameters
    ----------
    stim : [type]
        array is (ntrials,ntimesteps)
    choices : [type]
        the number of kind equals to `nunits`
    input_param_names : [type]
        free parameter names
    input_params : [type]
        free parameters values
    dt : [type], optional
        time per step, by default 1/20
    nunits : int, optional
        number of neuron units, by default 2

    Returns
    -------
    [type]
        [description]
    """    
    # set default values to zero and replace with input values
    param_names = ['tauR','tauG','omega','sigma','mu','bias']
    params      = {param_names[i]:0 for i in range(len(param_names))}
    for i in range(len(input_param_names)):
        params[input_param_names[i]] = input_params[i]
        
    # initialize parameters
    ntrials, ntimesteps = stim.shape
    W = params['omega']*np.ones((nunits,nunits)) # initialize inhibition weight
    R = np.empty((nunits,ntrials,ntimesteps+1)) # initialize excitatory R and inhibitory G units
    G = np.empty((nunits,ntrials,ntimesteps+1))
    R[:,:,0] = 0
    G[:,:,0] = 0
    C = np.zeros((nunits,ntrials,ntimesteps)) # reshape stim
    for i in range(nunits):
        C[i,:,:] = stim == np.unique(stim)[i] # stim are just 0 and 1s
    C.astype(int)
    
    # computational implementation of model
    for t in range(ntimesteps):
        R[0,:,t+1] = R[0,:,t] + dt/params['tauR'] * ( - R[0,:,t] + C[0,:,t] / (1 + G[0,:,t]) ) # do all trials together
        R[1,:,t+1] = R[1,:,t] + dt/params['tauR'] * ( - R[1,:,t] + C[1,:,t] / (1 + G[1,:,t]) )
        R[R<0] = 0
        G[:,:,t+1] = G[:,:,t] + dt/params['tauG'] * ( - G[:,:,t] + np.matmul(W,R[:,:,t]) )
        G[G<0] = 0
        
    G = G[1,:,:].squeeze() # the product makes unit 0 and unit 1 same
    
    # compute kernel
    exponent = np.exp(-np.arange(ntimesteps,0,-1)*dt/params['tauR'])
    kernel = (exponent/(1+ G[:,1:])+params['mu'])*params['sigma']/params['tauR']# wired sigma
    
    # compute loglikelihood
    temp = np.sum(kernel*stim,1) + params['bias'] 
    loglikelihoods = loglik(temp,choices)
    
    # compute simulated choices from model
    prob = 1./(1+np.exp(-temp))
    sim_choices = np.random.binomial(1,prob)
    
    return -np.sum(loglikelihoods), sim_choices, kernel[0,:] # return negative loglikelihood for minimization function

def loaddata(sub_no):
    regs = loadmat('regs.mat')['regs'][0,sub_no-1] 
    stim, choices = regs[0], regs[1]
    return stim.T, choices

def loglik(temp, choices=np.empty(0)):
    # approximate log values to avoid numerical error due to numbers being too small
    approx = loglikelihoods = np.ones(temp.shape)
    ind = temp > 10
    approx[ind],approx[~ind] = np.exp(-temp[ind]), np.log(1+np.exp(-temp[~ind])) # a technique remove small numbers, the first one is the apporximation log(1+x)~x
    # np.exp here for it is a logit function , see equation(9)
    loglikelihoods[choices==1] = -approx[choices==1] # log(p)=-log(1+e^(-temp))
    loglikelihoods[choices==0] = -temp[choices==0] - approx[choices==0] # wired log(1-p)=-temp-log(1+e^(-temp))
    return loglikelihoods



sub_no = 1 # participant no. 1
stim, choices = loaddata(sub_no)

# set parameters and corresponding values
args = ['tauR','tauG','omega','sigma','mu','bias']
x = [1.423,  25.530,  99.045, 10.919, -.490, -.059]

# input into model to get negative log likelihood (nLL), simulated choices from model (sim_choices), and weights for each piece of information in the model (kernel)
nLL, sim_choices, kernel = divnorm(stim, choices[0,:], args, x)
