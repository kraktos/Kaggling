import scipy as sp


def llfun(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1 - epsilon, pred)
    # ll = sum(act*sp.log(pred + 1) + sp.subtract(1, act)*sp.log(sp.subtract(1, pred)))
    sum(pow((log(pred + 1) - log(act + 1)), 2))
    ll = sqrt(ll * 1.0 / len(act))
    return ll
