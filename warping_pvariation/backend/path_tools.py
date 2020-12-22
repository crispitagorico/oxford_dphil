import numpy as np
import torch
import math
import signatory
import iisignature


def white(steps, width, time=1.):
    mu, sigma = 0, math.sqrt(time / steps) 
    return np.random.normal(mu, sigma, (steps, width))


def brownian(steps, width, time=1.):
    path = np.zeros((steps + 1, width))
    np.cumsum(white(steps, width, time), axis=0, out=path[1:, :])
    return path


def sine_wave_path(n, N, time):
    """Returns the path (1/n cos(2n**2t), 1/n sin(2n**2t))"""
    a = 1./float(n)
    b = 2.*(float(n)**2)
    return torch.tensor([[a*math.cos(b*t), a*math.sin(b*t)] for t in time]).unsqueeze(0)#.cuda()


def sine_wave_rough_path(n, N, time):
    """Returns the rough path extension at level 2 of the path above (pure area)"""
    a = 1./float(n)
    b = 2.*(float(n)**2)
    c = lambda t: math.cos(2.*t*(float(n)**2))
    s = lambda t: math.sin(2.*t*(float(n)**2)) 
    T = torch.zeros((N, 6))
    for t, i in zip(time, range(N)):
        T[i,0] = a*math.cos(b*t)
        T[i,1] = a*math.sin(b*t)
        T[i,2] = (c(t)**2 - 2.*c(t) + 1.)/(2*(float(n)**2))
        T[i,3] = (2.*t*(float(n)**2) + c(t)*s(t) - 2.*s(t))/(2*(float(n)**2))
        T[i,4] = (-2.*t*(float(n)**2) + c(t)*s(t))/(2*(float(n)**2))
        T[i,5] = (-c(t)**2) /(2*(float(n)**2))
    return T.unsqueeze(0)#.cuda()


def delayed_roughBM(BM, BM_del, delay, depth, time):
    if delay==0:
        return delayedBM_limit(BM, depth, time)
    BM = np.concatenate((BM, np.array(delay*[BM[-1]])), 0)
    N = len(BM)
    roughBM = np.zeros((N, iisignature.siglength(2, depth)))
    DeltaBM = np.append(0., np.diff(BM, 0))
    for k in range(N):
        roughBM[k,0] = BM[k]
        roughBM[k,1] = BM_del[k]
        roughBM[k,2] = 0.
        levy_area = (BM_del[:k]*DeltaBM[:k]).sum() - 0.5*BM[k]*BM_del[k]
        roughBM[k,3] = levy_area
        roughBM[k,4] = -levy_area
        roughBM[k,5] = 0.
    return roughBM


def x_limit(N):
    """Returns the zeros rough path in T^2(R^2)"""
    return signatory.Path(torch.zeros((1, N, 2)), 1, basepoint=True)
    #return signatory.Path(torch.zeros((1, N, 2)).cuda(), 1, basepoint=True)


def X_limit(N, depth, time):
    """Returns limiting rough path in p-var top for the pure area path problem (with signatory)"""
    X_inf = torch.zeros((1, N, signatory.signature_channels(2, depth)))#.cuda()
    for k in range(N):
        X_inf[0,k,0] = 0.
        X_inf[0,k,1] = 0.
        X_inf[0,k,2] = 0.
        X_inf[0,k,3] = time[k]
        X_inf[0,k,4] = -time[k]
        X_inf[0,k,5] = 0.
    return signatory.Path(X_inf, 1, basepoint=True)


def delayedBM_limit(BM_1d, depth, time):
    """Returns limiting rough path in p-var top for the delayed BM problem (with iisignature and BnB)"""
    N = len(BM_1d)
    X_inf = np.zeros((N, iisignature.siglength(2, depth)))
    for k, t in enumerate(time):
        X_inf[k,0] = BM_1d[k]
        X_inf[k,1] = BM_1d[k]
        X_inf[k,2] = 0.
        X_inf[k,3] = t/2.
        X_inf[k,4] = -t/2.
        X_inf[k,5] = 0.
    return X_inf


def response_limit(N, time):
    """Returns the solution of a linear differential equation (with pre-specified linear vector field)
       for the pure area path problem (signatory)"""
    y = torch.zeros((1, N, 3))#.cuda()
    for k in range(N):
        y[0,k,0] = 0.
        y[0,k,1] = 0.
        y[0,k,2] = time[k]
    return signatory.Path(y, 1, basepoint=True)
 
    
def response_limit_BM(BM_1d, time):
    """Returns the solution of a linear differential equation (with pre-specified linear vector field)
       for the delayed BM problem (iisignature)"""
    N = len(BM_1d)
    y = np.zeros((N, 3))
    for k, t in enumerate(time):
        y[k,0] = BM_1d[k]
        y[k,1] = BM_1d[k]
        y[k,2] = t/2.
    return y


def Lq(gamma1, gamma2, q):
    g = np.abs(gamma1 - gamma2)
    norm = 0.
    for p in g:
        norm += np.sqrt(sum([i**2 for i in p]))**q
    return norm**(1./q)


def L_infinity(gamma1, gamma2):
    return np.abs(gamma1 - gamma2).max()

