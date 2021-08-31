import numpy as np
import csv
from datetime import date, datetime
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.hp import quniform
from hyperopt.pyll import scope
from tqdm import tqdm
from functools import *
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import sys
sys.path.append('./problems/')
from CSMRI import CSMRI
sys.path.append('./algorithms/')
from pnp_svrg import tune_pnp_svrg
from pnp_sgd import tune_pnp_sgd
from pnp_gd import tune_pnp_gd
from pnp_sarah import tune_pnp_sarah
from pnp_saga import tune_pnp_saga


sys.path.append('./denoisers/')
from NLM import NLMDenoiser
denoiser = NLMDenoiser(filter_size=1, patch_size=4, patch_distance=5)

height, width = 256, 256
main_problem = CSMRI('./data/Set12/13.png', H=height, W=width, sample_prob=0.5, sigma=5)

output_fn = 'hyperparam-tuning' + datetime.now().strftime('-%y-%m-%d-%H-%M') + '.csv'
TIME_PER_TRIAL = 20
MAX_EVALS = 10

with open(output_fn, 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(['CSMRI', 'PNPSVRG'])

    # create proxy function for hyperopt tuning
    svrg_proxy = partial(tune_pnp_svrg, problem=main_problem, denoiser=denoiser, tt=TIME_PER_TRIAL, verbose=False, lr_decay=1, converge_check=True, diverge_check=True)

    # create parameter space
    pspace = (
        hp.uniform('eta', 1e-5, 1),
        scope.int(quniform('mini_batch_size', 1, 1000, q=1)),
        scope.int(quniform('T2', 1, 1000, q=1))
    )

    pbar = tqdm(total=MAX_EVALS, desc="Hyperopt PNPSVRG")
    trials = Trials()
    results = fmin(
        svrg_proxy,
        space=pspace,
        algo=tpe.suggest,
        trials=trials,
        max_evals=MAX_EVALS
    )
    pbar.close()

    out = svrg_proxy((results['eta'], int(results['mini_batch_size']), int(results['T2'])))

    writer.writerow(['eta', 'mini_batch_size', 'T2'])
    writer.writerow([results['eta'], results['mini_batch_size'], results['T2']])
    writer.writerow(['loss', 'z'])
    writer.writerow([out['loss'], out['psnr_per_iter'][0], out['psnr_per_iter'][-1]])

    print(results['eta'], results['mini_batch_size'], results['T2'], out['loss'], out['psnr_per_iter'][-1])

    writer.writerow(['CSMRI', 'PNPSGD'])

    # create proxy function for hyperopt tuning
    sgd_proxy = partial(tune_pnp_sgd, problem=main_problem, denoiser=denoiser, tt=TIME_PER_TRIAL, verbose=False, lr_decay=1, converge_check=True, diverge_check=True)

    # create parameter space
    pspace = (
        hp.uniform('eta', 1e-5, 1),
        scope.int(quniform('mini_batch_size', 1, 1000, q=1))
    )

    pbar = tqdm(total=MAX_EVALS, desc="Hyperopt PNPSGD")
    trials = Trials()
    results = fmin(
        sgd_proxy,
        space=pspace,
        algo=tpe.suggest,
        trials=trials,
        max_evals=MAX_EVALS
    )
    pbar.close()

    out = sgd_proxy((results['eta'], int(results['mini_batch_size'])))

    writer.writerow(['eta', 'mini_batch_size'])
    writer.writerow([results['eta'], results['mini_batch_size']])
    writer.writerow(['loss', 'z'])
    writer.writerow([out['loss'], out['psnr_per_iter'][0], out['psnr_per_iter'][-1]])

    print(results['eta'], results['mini_batch_size'], out['loss'], out['psnr_per_iter'][-1])
 


