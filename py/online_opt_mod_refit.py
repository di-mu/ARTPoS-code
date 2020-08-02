# online_opt_mod_refit.py.

'''

Previous network code integrated w/ refit.py to refit curve.
Reformulation of Network Optimization code to allow for PRR model refits,
performed on violation of a threshold
(i.e PRR values being consistently under/overestimated).

'''

import numpy as np
import os.path
from refit import reformulation_main

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


def f(x, y, prrw, prrz, Dt, hw, hz):
    """ f(x,y) first searches (using provided x,y values) for values
    in the outer power-transmission values matrix and obtains the corresponding
    inner matrix. Algorithm is run to check if ratio of data transfer rate
    and throughput exceed 1. f(x,y) then reverse searches outer matrix values
    for best fit and outputs x,y for outer matrix.  f(x,y) also assumes
    that values x,y are given with regards to dbm transmission settings for
    the Wifi and Zigbee modules (wifi transmission values from 1-21 and zigbee
    transmission values from -6 to 5). f(x,y) converts x,y values for
    easier indexing of arrays and lists. Correlated values are
    hardcoded as lists below.
    """
    col = [-9, -6, -3, 0, 1, 2, 3, 4, 5]
    row = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
           12, 13, 14, 15, 16, 17, 18, 19, 20, 21]

    # obtains inner_matrix for input power_value
    def find_nearest_wifi(array, val):
        if np.abs(array - val).min() > prr_limit:
            prr_state = np.abs(array - val).argmin()
            refit_data = reformulation_main(
                orig_prr_w[prr_state], row.index(x), val, wifi_dBm)
            return refit_data[np.abs(refit_data - val).argmin()]
        else:
            return array[np.abs(array - val).argmin()]

    def find_nearest_zigbee(array, val):
        if np.abs(array - val).min() > prr_limit:
            prr_state = np.abs(array - val).argmin()
            refit_data = reformulation_main(
                orig_prr_z[prr_state], col.index(y), val, zigbee_dBm)
            return refit_data[np.abs(refit_data - val).argmin()]
        else:
            return array[np.abs(array - val).argmin()]

    # return D/G
    return Dt / ((hw * find_nearest_wifi(prr_w[row.index(x)], prrw)) +
                 (hz * find_nearest_zigbee(prr_z[col.index(y)], prrz)))


def search(x, y, prrw, prrz, Dt, hw, hz):
    """ searches down the outer matrix for the next power transmission
    value that satisfies the ratio of the data transfer rate to the
    goodput < 0.9.
    """
    # if else statement to check if prrz or prrw is 0
    u = x
    v = y
    col = [-9, -6, -3, 0, 1, 2, 3, 4, 5]
    row = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
           12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    feasible = []
    power_val = []
    loc = []
    if x == 21:
        x = 20
    if f(x, y, prrw, prrz, Dt, hw, hz) < 0.9:
        pass
    else:
        exclude = power_table[:row.index(x) + 1, :col.index(y) + 1]
        for (x, y), value in np.ndenumerate(power_table):
            if value in exclude.flatten():
                pass
            else:
                a = (np.where(power_table == value)[0])[0]
                b = (np.where(power_table == value)[1])[0]
                current_settings = f(row[a], col[b], prrw, prrz, Dt, hw, hz)
                if current_settings > 0.9:
                    pass
                else:
                    feasible.append(value)
                    power_val.append(current_settings * value)
                    loc.append((x, y))
    if not power_val:
        return search(u, v, prrw + 0.1, prrz + 0.05, Dt, hw, hz)
    else:
        with open('pyout.dat', 'w') as outp:
            outp.write(str(row[loc[power_val.index(min(power_val))][0]]) +
                       " " + str(col[loc[power_val.index(min(power_val))][1]]))


def search_test(x, y, prrw, prrz, Dt, hw, hz):
    """ search function with exclusion condition removed to test results """
    if prrw == 0:
        prrw = 0.4
    # checks if x = 21 (because of the exclude statement,
    # an input of 21 will result in an empty output)
    if x == 21:
        x = 20
    # check for 0 dB values in the wifi and defaults to a median value
    if x == 0:
        x = 11
    u = x
    v = y
    col = [-9, -6, -3, 0, 1, 2, 3, 4, 5]
    row = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
           12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
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
            current_settings = f(row[a], col[b], prrw, prrz, Dt, hw, hz)
            if current_settings > 0.9:
                pass
            else:
                feasible.append(value)
                power_val.append(current_settings * value)
                loc.append((x, y))
    if not power_val:
        if prrw > 1:
            return x, y
        else:
            return search_test(u, v, prrw + 0.1, prrz + 0.05, Dt, hw, hz)
    else:
        idx = power_val.index(min(power_val))
        return row[loc[idx][0]], col[loc[idx][1]]


# erroneous runtime error catching
np.seterr(all='ignore')
