# small script to generate numpy files from data for use w/ Python
import numpy as np
import os.path
import scipy.io

def gen():
    power_table = np.genfromtxt('power_table.csv', delimiter=",")
    mat = scipy.io.loadmat('PRR_4_compact.mat') #load mat struct file
    #generates numpy files
    np.save('power_table.npy', power_table)
    np.save('prr_w.npy', mat['prr_w'])
    np.save('prr_z.npy', mat['prr_z'])

