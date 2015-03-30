import scipy as sp


def llfun(act, pred):
    # print("diff = " + str(act) + "\t" + str(pred))

    ll = sum(pow((sp.log(pred + 1) - sp.log(act + 1)), 2))
    ll = sp.sqrt(ll * 1.0 / len(act))
    return ll
