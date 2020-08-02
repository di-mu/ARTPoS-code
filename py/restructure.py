'''

Previous network code integrated w/ refit.py to refit curve.
Reformulation of Network Optimization code to allow for PRR model refits,
performed on violation of a threshold
(i.e PRR values being consistently under/overestimated).

'''

import numpy as np
import logging
import os.path
import os
from refit import reformulation_main

# Logging
LOG_FILENAME = 'searchtest.log'
logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG)


# generating consts
if os.path.exists('power_table.npy') and os.path.exists('prr_w.npy'):
    power_table = np.load('power_table.npy')
    prr_w = np.load('prr_w.npy')
    prr_z = np.load('prr_z.npy')  # load python-converted datasets
    """ looks in the working director for the following files:
    'PRR_4_cases.mat' - matlab 21x9 struct with inner matrices
    'power_table.csv' - the power table in a csv computer-readable format
    'power_table.npy' - power tables stored as numpy python-readable arrays
    'inner_matrices.npy' - inner matrices stored as numpy python-readable
    arrays

    If we find that the .npy files do not exist, they are generated
    using the genfile.py module (on the assumption that the .csv
    and .mat datasets are present in the working directory.
    """
else:
    from genfile import gen
    gen()

zigbee_dBm = np.array([-9, -6, -3, 0, 1, 2, 3, 4, 5])
wifi_dBm = np.array([1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21])
orig_prr_w = np.load('orig_prr_w.npy')
orig_prr_w = orig_prr_w / 100
orig_prr_z = np.load('orig_prr_z.npy')
orig_prr_z = orig_prr_z / 100
orig_prr_z = np.c_[np.zeros((4, 1)), orig_prr_z]
prr_limit = 0.5
col = [-9, -6, -3, 0, 1, 2, 3, 4, 5]
row = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
       12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
np.seterr(all='ignore')


def find_nearest_wifi(array, val, x):
    if np.abs(array - val).min() > prr_limit:
        prr_state = np.abs(array - val).argmin()
        refit_data = reformulation_main(
            orig_prr_w[prr_state], row.index(x), val, wifi_dBm)
        return refit_data[np.abs(refit_data - val).argmin()]
    else:
        return array[np.abs(array - val).argmin()]


def find_nearest_zigbee(array, val, y):
    if np.abs(array - val).min() > prr_limit:
        prr_state = np.abs(array - val).argmin()
        refit_data = reformulation_main(
            orig_prr_z[prr_state], col.index(y), val, zigbee_dBm)
        return refit_data[np.abs(refit_data - val).argmin()]
    else:
        return array[np.abs(array - val).argmin()]


def func(x, y, prrw, prrz, Dt, hw, hz):
    return Dt / ((hw * find_nearest_wifi(prr_w[row.index(x)], prrw, x)) +
                 (hz * find_nearest_zigbee(prr_z[col.index(y)], prrz, y)))


def search(x, y, prrw, prrz, Dt, hw, hz):
    """ search function with exclusion condition removed to test results """
    if x == 21:
        x = 20
    if x == 0:
        x = 11

    if os.path.exists('prev.npz'):
        prev_data = np.load('prev.npz')
        goodput_w = hw * find_nearest_wifi(prr_w[row.index(x)], prrw, x)
        goodput_z = hz * find_nearest_zigbee(prr_z[col.index(y)], prrz, y)
        old_goodput_w = prev_data['prev_hw'] * find_nearest_wifi(
            prr_w[row.index(prev_data['out'][0])], prev_data['prev_prrw'],
            prev_data['out'][0])
        old_goodput_z = prev_data['prev_hz'] * find_nearest_zigbee(
            prr_z[col.index(prev_data['out'][1])], prev_data['prev_prrz'],
            prev_data['out'][1])
                                                                       
        if prev_data['out'][0] == 0 or prev_data['out'][1] == 0:
            if prev_data['prev_Dt'] == Dt:
                if np.abs(goodput_z - old_goodput_z) < 0.1*old_goodput_z:
                    return prev_data['out'][0], prev_data['out'][1]
                else:
                    pass
            else:
                pass
        else:
            pass
    else:
        pass
    
    feasible = []
    power_val = []
    loc = []
    exclude = power_table[:row.index(x) + 1, :col.index(y) + 1]

    for (x, y), value in np.ndenumerate(power_table):
        if value in exclude.flatten():
            pass
        else:
            a = (np.where(power_table == value)[0])[0]
            b = (np.where(power_table == value)[1])[0]
            current_settings = func(row[a], col[b], prrw, prrz, Dt, hw, hz)
            if current_settings > 0.9 or np.isnan(
                    current_settings * value):
                pass
            else:
                feasible.append(value)
                power_val.append(current_settings*value)
                loc.append((x, y))
            np.savetxt('powerval', power_val)
            np.savetxt('loc', loc)
            np.savetxt('feasible', feasible)
            np.savetxt('exclude', exclude)
    if not power_val:
            # all values break > 0.9 constraint
        for (x, y), value in np.ndenumerate(power_table):
            if value in exclude.flatten():
                pass
            else:
                a = (np.where(power_table == value)[0])[0]
                b = (np.where(power_table == value)[1])[0]
                current_settings = func(row[a], col[b], prrw, prrz, Dt, hw, hz)
                if np.isnan(current_settings * value):
                    pass
                else:
                    feasible.append(value)
                    power_val.append(current_settings)
                    loc.append((x, y))
        idx = power_val.index(min(power_val))
        np.savez('prev.npz', out=[row[loc[idx][0]], col[loc[idx][1]]], prev_Dt=Dt,
                 prev_prrw=prrw, prev_prrz=prrz, prev_hw=hw, prev_hz=hz)
        return row[loc[idx][0]], col[loc[idx][1]]
    else:
        idx = power_val.index(min(power_val))
        np.savez('prev.npz', out=[row[loc[idx][0]], col[loc[idx][1]]], prev_Dt=Dt,
                 prev_prrw=prrw, prev_prrz=prrz, prev_hw=hw, prev_hz=hz)
        return row[loc[idx][0]], col[loc[idx][1]]
