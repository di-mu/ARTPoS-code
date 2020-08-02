# online_opt_mod.py.
# nravicha@buffalo.edu; contact if any troubles are encountered
# This version contains changes to input conditions; f and search now require data rate and throughput values
import numpy as np
import datetime
import warnings
import csv #reading csv data
import os.path
import random
import warnings
#check for converted data structures
if os.path.exists('power_table.npy') and os.path.exists('prr_w.npy'):
    power_table = np.load('power_table.npy')
    prr_w = np.load('prr_w.npy')
    prr_z = np.load('prr_z.npy')#load python-converted datasets
else:
    from genfile import gen
    gen()
    """ looks in the working director for the following files: 
    'PRR_4_cases.mat' - matlab 21x9 struct with inner matrices
    'power_table.csv' - the power table in a csv computer-readable format
    'power_table.npy' - power tables stored as numpy python-readable arrays
    'inner_matrices.npy' - inner matrices stored as numpy python-readable arrays
    
    If we find that the .npy files do not exist, they are generated using the genfile.py module
    (on the asusmption that the .csv and .mat datasets are present in the working directory.
    
    """
    
def f(x, y, prrw, prrz, Dt, hw, hz): #currently does not accept a Dt (Data transmission rate) value, random one is used
    """ f(x,y) first searches (using provided x,y values) for values in the outer power-transmission values matrix
    and obtains the corresponding inner matrix. Algorithm is run to check if ratio of data transfer rate and 
    throughput exceed 1. 

    f(x,y) then reverse searches outer matrix values for best fit and outputs x,y for outer matrix. 
    f(x,y) also assumes that input values x,y are given with regards to dbm transmission settings for the Wifi and Zigbee modules
    (wifi transmission values from 1-21 and zigbee transmission values from -6 to 5). f(x,y) converts x,y values for easier indexing
    of arrays and lists. Correlated values are hardcoded as lists below.
    """
    col = [-9, -6, -3, 0, 1, 2, 3, 4, 5]
    row = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    
    outer_val = power_table[row.index(x), col.index(y)]#obtains inner_matrix for input power_value
    def find_nearest(array,val):
        return array.flat[np.abs(array - val).argmin()] #quick function to search inner matrix and find quality of current transmission (high, med, low, poor)
    return Dt/((hw*find_nearest(prr_w[row.index(x)], prrw)) + (hz * find_nearest(prr_z[col.index(y)], prrz))) #return D/G


def search(x,y,prrw,prrz, Dt, hw, hz): 
    """ searches down the outer matrix for the next power transmission value that satisfies the ratio of the data transfer rate
    to the goodput < 0.9.  
    """
    #if else statement to check if prrz or prrw is 0
    u = x; v = y
    col = [-9, -6, -3, 0, 1, 2, 3, 4, 5]
    row = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    feasible=[]
    power_val = []
    loc = []
    if x == 21:
        x = 20
    if f(x, y, prrw, prrz, Dt, hw, hz) < 0.9:
        pass
    else:
        exclude = power_table[:row.index(x)+1, :col.index(y)+1] #exclusion matrix with values that automatically do not satisfy condition
        for (x,y), value in np.ndenumerate(power_table):
            if value in exclude.flatten():
                pass
            else:
                a = (np.where(power_table==value)[0])[0]
                b = (np.where(power_table==value)[1])[0]
                current_settings = f(row[a], col[b], prrw, prrz, Dt, hw, hz)
                if current_settings > 0.9:
                    pass
                else:
                    feasible.append(value)
                    power_val.append(current_settings * value)
                    loc.append((x,y))
   # return loc[power_val.index(min(power_val))], feasible[power_val.index(min(power_val))], min(power_val)
                    #returns minimum power value that satisfies condition
    if not power_val:
        return search(u,v,prrw+0.1,prrz+0.05,Dt,hw,hz)
    else:
        with open('pyout.dat', 'w') as outp:
            outp.write(str(row[loc[power_val.index(min(power_val))][0]])+" "+str(col[loc[power_val.index(min(power_val))][1]]))
        
               

def search_test(x,y,prrw,prrz, Dt, hw, hz):
    """ search function with exclusion condition removed to test for results """
    if prrw == 0:
        prrw == 0.4
    #checks if x = 21 (because of the exclude statement, an input of 21 will result in an empty output)
    if x == 21:
        x = 20
    #check for 0 dB values in the wifi and defaults to a median value
    if x == 0:
        x = 11
    u = x; v = y
    col = [-9, -6, -3, 0, 1, 2, 3, 4, 5]
    row = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    feasible=[]
    power_val = []
    loc = []
    exclude = power_table[:row.index(x)+1, :col.index(y)+1]
    for (x,y), value in np.ndenumerate(power_table):
        if value in exclude.flatten():
            pass
        else:
            a = (np.where(power_table==value)[0])[0]
            b = (np.where(power_table==value)[1])[0]
            current_settings = f(row[a], col[b], prrw, prrz, Dt, hw, hz)
            if current_settings > 0.9:
                pass
            else:
                feasible.append(value)
                power_val.append(current_settings * value)
                loc.append((x,y))
    if not power_val:
        if prrw > 1:
            return x, y
        else:
            return search_test(u,v,prrw+0.1,prrz+0.05,Dt,hw,hz)
    else:
        idx = power_val.index(min(power_val))
        return loc[idx][0], loc[idx][1]



#erroneous runtime error catching
np.seterr(all='ignore') 
