import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import os

for i in range(26032):
    for j in range(1,11):
        if (os.path.exists("dataset/SVHN/test/"+str(i)+"_"+str(j)+"_.jpg")):
            f = open("dataset/SVHN/SVHN_test_labels.txt", "a+")
            f.writelines(str(i)+"_"+str(j)+"_.jpg "+str(j%10)+'\n')
            f.close()
    print(i)