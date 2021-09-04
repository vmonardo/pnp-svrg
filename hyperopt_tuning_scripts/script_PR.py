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
from PR import PhaseRetrieval

sys.path.append('./algorithms/')
from pnp_svrg import tune_pnp_svrg
from pnp_sgd import tune_pnp_sgd
from pnp_gd import tune_pnp_gd
from pnp_sarah import tune_pnp_sarah
from pnp_saga import tune_pnp_saga


sys.path.append('./denoisers/')
from NLM import NLMDenoiser
denoiser = NLMDenoiser(filter_size=1, patch_size=4, patch_distance=5)

height, width = 32, 32

alpha = 20 # meas / dims
main_problem = PhaseRetrieval(img_path='./data/Set12/01.png', H=height, W=width, num_meas = alpha*height*width)

PROBLEM_NAME = 'PR'

output_fn = 'hyperparam-tuning' + PROBLEM_NAME + datetime.now().strftime('-%y-%m-%d-%H-%M') + '.csv'

TIME_PER_TRIAL = 30
MAX_EVALS = 1000

eta_min, eta_max = 0, .1
mb_min, mb_max = 1, main_problem.M
T2_min, T2_max = 1, main_problem.M
dstr_min, dstr_max = 0, 2

with open(output_fn, 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')

    ##########################
    # PNP SVRG TUNING
    ##########################

    writer.writerow([PROBLEM_NAME, 'PNPSVRG'])

    # # create proxy function for hyperopt tuning
    # svrg_proxy = partial(tune_pnp_svrg, problem=main_problem, denoiser=denoiser, tt=TIME_PER_TRIAL, verbose=False, lr_decay=1, converge_check=True, diverge_check=True)

    # # create parameter space
    # pspace = (
    #     hp.uniform('eta', eta_min, eta_max),   
    #     scope.int(quniform('mini_batch_size', mb_min, mb_max, q=1)),
    #     scope.int(quniform('T2', T2_min, T2_max, q=1)),
    #     hp.uniform('dstrength', dstr_min, dstr_max)
    # )

    # pbar = tqdm(total=MAX_EVALS, desc="Hyperopt PNPSVRG")
    # trials = Trials()
    # results = fmin(
    #     svrg_proxy,
    #     space=pspace,
    #     algo=tpe.suggest,
    #     trials=trials,
    #     max_evals=MAX_EVALS
    # )
    # pbar.close()

    # print(results)

    # out = svrg_proxy((results['eta'], int(results['mini_batch_size']), int(results['T2']), results['dstrength']))

    # writer.writerow(['eta', 'mini_batch_size', 'T2'])
    # writer.writerow([results['eta'], results['mini_batch_size'], results['T2']])
    # writer.writerow(['loss', 'initial PSNR', 'output PSNR'])
    # writer.writerow([out['loss'], out['psnr_per_iter'][0], out['psnr_per_iter'][-1]])

    # print(results['eta'], results['mini_batch_size'], results['T2'], out['loss'], out['psnr_per_iter'][0], out['psnr_per_iter'][-1])

    # ##########################
    # # PNP SGD TUNING
    # ##########################

    # writer.writerow([PROBLEM_NAME, 'PNPSGD'])

    # # create proxy function for hyperopt tuning
    # sgd_proxy = partial(tune_pnp_sgd, problem=main_problem, denoiser=denoiser, tt=TIME_PER_TRIAL, verbose=False, lr_decay=1, converge_check=True, diverge_check=True)

    # # create parameter space
    # pspace = (
    #     hp.uniform('eta', eta_min, eta_max),
    #     scope.int(quniform('mini_batch_size', mb_min, mb_max, q=1)),
    #     hp.uniform('dstrength', dstr_min, dstr_max)
    # )

    # pbar = tqdm(total=MAX_EVALS, desc="Hyperopt PNPSGD")
    # trials = Trials()
    # results = fmin(
    #     sgd_proxy,
    #     space=pspace,
    #     algo=tpe.suggest,
    #     trials=trials,
    #     max_evals=MAX_EVALS
    # )
    # pbar.close()

    # print(results)

    # out = sgd_proxy((results['eta'], int(results['mini_batch_size']), results['dstrength']))

    # writer.writerow(['eta', 'mini_batch_size'])
    # writer.writerow([results['eta'], results['mini_batch_size']])
    # writer.writerow(['loss', 'initial PSNR', 'output PSNR'])
    # writer.writerow([out['loss'], out['psnr_per_iter'][0], out['psnr_per_iter'][-1]])

    # print(results['eta'], results['mini_batch_size'], out['loss'], out['psnr_per_iter'][0], out['psnr_per_iter'][-1])

    ##########################
    # PNP GD TUNING
    ##########################

    writer.writerow([PROBLEM_NAME, 'PNPGD'])

    # create proxy function for hyperopt tuning
    gd_proxy = partial(tune_pnp_gd, problem=main_problem, denoiser=denoiser, tt=TIME_PER_TRIAL, verbose=False, lr_decay=1, converge_check=True, diverge_check=True)

    # create parameter space
    pspace = (
        hp.uniform('eta', eta_min, eta_max),
        hp.uniform('dstrength', dstr_min, dstr_max)
    )

    pbar = tqdm(total=MAX_EVALS, desc="Hyperopt PNPGD")
    trials = Trials()
    results = fmin(
        gd_proxy,
        space=pspace,
        algo=tpe.suggest,
        trials=trials,
        max_evals=MAX_EVALS
    )
    pbar.close()

    print(results)

    out = gd_proxy((results['eta'], results['dstrength']))

    writer.writerow(['eta'])
    writer.writerow([results['eta']])
    writer.writerow(['loss', 'initial PSNR', 'output PSNR'])
    writer.writerow([out['loss'], out['psnr_per_iter'][0], out['psnr_per_iter'][-1]])

    print(results['eta'], out['loss'], out['psnr_per_iter'][0], out['psnr_per_iter'][-1])

    # ##########################
    # # PNP SAGA TUNING
    # ##########################

    # writer.writerow([PROBLEM_NAME, 'PNPSAGA'])

    # # create proxy function for hyperopt tuning
    # saga_proxy = partial(tune_pnp_saga, problem=main_problem, denoiser=denoiser, tt=TIME_PER_TRIAL, verbose=False, lr_decay=1, converge_check=True, diverge_check=True)

    # # create parameter space
    # pspace = (
    #     hp.uniform('eta', eta_min, eta_max),
    #     scope.int(quniform('mini_batch_size', mb_min, mb_max, q=1)),
    #     hp.uniform('dstrength', dstr_min, dstr_max)
    # )

    # pbar = tqdm(total=MAX_EVALS, desc="Hyperopt PNPSAGA")
    # trials = Trials()
    # results = fmin(
    #     saga_proxy,
    #     space=pspace,
    #     algo=tpe.suggest,
    #     trials=trials,
    #     max_evals=MAX_EVALS
    # )
    # pbar.close()

    # print(results)

    # out = saga_proxy((results['eta'], int(results['mini_batch_size']), results['dstrength']))

    # writer.writerow(['eta', 'mini_batch_size'])
    # writer.writerow([results['eta'], results['mini_batch_size']])
    # writer.writerow(['loss', 'initial PSNR', 'output PSNR'])
    # writer.writerow([out['loss'], out['psnr_per_iter'][0], out['psnr_per_iter'][-1]])

    # print(results['eta'], results['mini_batch_size'], out['loss'], out['psnr_per_iter'][0], out['psnr_per_iter'][-1])

    # ##########################
    # # PNPSARAH TUNING
    # ##########################

    # writer.writerow([PROBLEM_NAME, 'PNPSARAH'])

    # # create proxy function for hyperopt tuning
    # sarah_proxy = partial(tune_pnp_sarah, problem=main_problem, denoiser=denoiser, tt=TIME_PER_TRIAL, verbose=False, lr_decay=1, converge_check=True, diverge_check=True)

    # # create parameter space
    # pspace = (
    #     hp.uniform('eta', eta_min, eta_max),
    #     scope.int(quniform('mini_batch_size', mb_min, mb_max, q=1)),
    #     scope.int(quniform('T2', T2_min, T2_max, q=1)),
    #     hp.uniform('dstrength', dstr_min, dstr_max)
    # )

    # pbar = tqdm(total=MAX_EVALS, desc="Hyperopt PNPSARAH")
    # trials = Trials()
    # results = fmin(
    #     sarah_proxy,
    #     space=pspace,
    #     algo=tpe.suggest,
    #     trials=trials,
    #     max_evals=MAX_EVALS
    # )
    # pbar.close()

    # print(results)

    # out = sarah_proxy((results['eta'], int(results['mini_batch_size']), int(results['T2']), results['dstrength']))

    # writer.writerow(['eta', 'mini_batch_size', 'T2'])
    # writer.writerow([results['eta'], results['mini_batch_size'], results['T2']])
    # writer.writerow(['loss', 'initial PSNR', 'output PSNR'])
    # writer.writerow([out['loss'], out['psnr_per_iter'][0], out['psnr_per_iter'][-1]])

    # print(results['eta'], results['mini_batch_size'], results['T2'], out['loss'], out['psnr_per_iter'][0], out['psnr_per_iter'][-1])
 

