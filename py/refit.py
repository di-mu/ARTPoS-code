'''

refit.py - Reformulation of Network Optimization code to allow for
PRR model refits, performed on violation of a threshold
(i.e PRR values being consistently under/overestimated ).

'''
import numpy as np
from scipy.optimize import leastsq


def insert_val(prr_model, dBm, measured_val, wifi_dBm):
    ''' Insert measured PRR value that violates constraints at poor/low/medium/high
    transmission setting into the appropriate PRR array (poor-high).
    Requires measured PRR dBm.
    '''
    if np.where(wifi_dBm == dBm) == 0:
        i = np.where(wifi_dBm == dBm - 1)
    else:
        i = np.where(wifi_dBm == dBm)
    wifi_dBm = np.insert(wifi_dBm, i[0], dBm)
    prr_model = np.insert(prr_model, i[0], measured_val)
    return prr_model, wifi_dBm


def logistic4(x, A, B, C, D):
    ''' 4PL Logistic function '''
    return ((A - D) / (1.0 + ((x / C)**B))) + D


def residuals(p, y, x):
    ''' Deviations of data from fitted curve '''
    A, B, C, D = p
    err = y - logistic4(x, A, B, C, D)
    return err


def peval(x, p):
    ''' Evaluate value at x with current parameters '''
    A, B, C, D = p
    return logistic4(x, A, B, C, D)


# ICDCS paper uses measured values at odd wifi dBm values
wifi_dBm = np.array([1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21])
p0 = [0, 1, 1, 1]  # initial guess
orig_prr = np.load('orig_prr.npy')
orig_prr = orig_prr / 100
orig_prr_poor = orig_prr[0]
orig_prr_low = orig_prr[1]
orig_prr_med = orig_prr[2]
orig_prr_high = orig_prr[3]  # load and store original PRR arrays


def reformulation_main(prr_model, dBm, measured_val, wifi_dBm):
    for i in range(0, np.size(dBm) - 1):
        prr_model, wifi_dBm = insert_val(
            prr_model, dBm[i], measured_val[i], wifi_dBm)

    # perform L4P optimization
    plsq = leastsq(residuals, p0, args=(prr_model, wifi_dBm))
    refit_data = peval(wifi_dBm, plsq[0])  # fit data
    return refit_data
