from types import ClassMethodDescriptorType
import numpy as np
import csv
from datetime import date, datetime
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from tqdm import tqdm
from functools import *
import sys
sys.path.append('../algorithms')
sys.path.append('../denoisers')
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

NAME = 'CSMRI'
ALG = 'PNPSVRG'
# eta, mini_batch_size, T2 = args

output_fn = '../data/' + NAME + '/' + ALG + '/' + NAME + '-hyperparam-tuning' + datetime.now().strftime('-%y-%m-%d-%H-%M') + '.csv'

with open(output_fn, 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(['CSMRI', 'PNPSVRG'])
    writer.writerow(['eta', 'mini batch size', 'T2'])

