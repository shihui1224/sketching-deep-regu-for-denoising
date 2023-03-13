import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from sklearn.preprocessing import normalize

def emp_sketch(X, W):
    # X: n*d
    # W: m*d
    n = X.shape[0]
    m = W.shape[0]
    z = np.zeros(2 * m)
    wtx = W@X.transpose()
    z[:m] = np.sum(np.cos(wtx), 1)
    z[m:] = np.sum(np.sin(wtx), 1)
    z /= n
    return z

def cost_fit_exp(ssig, y, r):
    e1 = np.exp(- 1 / 2 * r * ssig)
    d = e1 - y
    return np.linalg.norm(d) ** 2


def grad_cost(ssig, y, r):
    e1 = np.exp(- 1 / 2 * r * ssig)
    d = e1 - y
    grad = -np.sum(e1 * r * d)
    return grad


def update_spread(c, sk, r, mean_var):
    m = sk.shape[0]
    fr = int(np.floor(m / c))
    val = np.zeros(c)
    rr = np.zeros(c)
    for k in range(c):
        a = sk[k * fr: (k + 1) * fr]
        val[k], i = np.max(a), np.argmax(a)
        rr[k] = r[k * fr + i]  ##
    opt_result = minimize(cost_fit_exp, x0=mean_var, method="trust-constr", args=(val, rr),
                          jac=grad_cost, options={'disp': False})
    mean_var = opt_result.x
    return mean_var


def cdf_lookup(ax, pdf, n):
    pdf = pdf / np.sum(pdf)
    cdf = np.cumsum(pdf)
    cdf = np.hstack((cdf, 0))
    ind_list = np.nonzero((cdf[:-1] - cdf[1:]) >= 0)
    ind = ind_list[0][0]
    cdf_new = cdf[:ind]  ##
    ax_new = ax[:ind]
    y = np.random.rand(n,1)
    f = interp1d(cdf_new, ax_new, bounds_error=False, fill_value=0)
    Y = f(y)
    return Y


def draw_freq(d, m, mean_var):
    # draw angles
    wdir = np.random.randn(m, d)
    # wdir /= np.sqrt(np.sum(wdir ** 2, 1))
    wdir = normalize(wdir, axis = 1)
    # multiply by variances
    # Wdir = wdir@variances
    Wdir = wdir / np.sqrt(mean_var)
    ax = np.arange(0, 10.001, 0.001)
    # radius_density = 0
    radius_density = np.sqrt(ax ** 2 + ax ** 4 / 4) * np.exp(-ax ** 2 / 2)
    wnorm = cdf_lookup(ax, radius_density, m)
    w = wnorm * Wdir
    # # sort the frequencies by increasing radius ||w_j||2
    # """ind"""
    # ind = np.argsort(np.sum(w ** 2, 1))
    # w = w[ind, :]
    return w


def estimate_mean_var(sk_wrapper, X):
    d = sk_wrapper.d
    smallm = sk_wrapper.options.smallm
    nb_it = sk_wrapper.options.estim_Sigma_nb_iter
    c = sk_wrapper.options.estim_Sigma_sketch_division
    mean_var = 1
    for i in range(nb_it):
        W = draw_freq(d, smallm, mean_var)
        R = np.sum(W ** 2, axis=1)
        esk = emp_sketch(X, W)
        sk = esk[:smallm] - esk[smallm:] * 1j
        mean_var = update_spread(c, np.abs(sk), R, mean_var)
    return mean_var
