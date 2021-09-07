from hyperopt import fmin, tpe, hp, Trials
from datetime import datetime
import csv
from hyperopt.hp import quniform
from hyperopt.pyll import scope
from tqdm import tqdm
from functools import *
import os
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

from problems import *
from algorithms import *
from denoisers import *

PROBLEM_LIST = ['CSMRI', 'DeblurSR', 'PR']
ALGO_LIST = ['pnp_gd', 'pnp_sgd', 'pnp_saga', 'pnp_sarah', 'pnp_svrg']
DENOISER_LIST = ['NLM', 'CNN', 'BM3D', 'TV']

SIGMA = 0.1
KERNEL = "Minimal"
ALPHA = 20
IM_HEIGHT = 32
IM_WIDTH = 32
IM_PATH = './data/Set12/01.png'

TIME_PER_TRIAL = 2
MAX_EVALS = 2

eta_min, eta_max = 0, .1
mb_min, mb_max = 1, 100
T2_min, T2_max = 1, 100
dstr_min, dstr_max = 0, 2

def get_problem(prob_name):
    if prob_name == 'CSMRI':
        return CSMRI(img_path=IM_PATH, H=IM_HEIGHT, W=IM_WIDTH, sample_prob=0.5, sigma=SIGMA)
    if prob_name == 'DeblurSR':
        return Deblur(img_path=IM_PATH, kernel=KERNEL, H=IM_HEIGHT, W=IM_WIDTH, sigma=SIGMA, scale_percent=50)
    if prob_name == 'PR':
        return PhaseRetrieval(img_path=IM_PATH, H=IM_HEIGHT, W=IM_WIDTH, num_meas = ALPHA*IM_HEIGHT*IM_WIDTH)
    else:
        raise Exception('Problem name "{0}" not found'.format(prob_name)) 

def get_denoiser(dnr_name):
    if dnr_name == 'BM3D':
        return BM3DDenoiser()
    if dnr_name == 'NLM':
        return NLMDenoiser()
    if dnr_name == 'TV':
        return TVDenoiser()
    if dnr_name == 'CNN':
        return CNNDenoiser()
    else:
        raise Exception('Denoiser name "{0}" not found'.format(dnr_name)) 

def get_proxy_pspace(main_problem, algo_name, denoiser):
    if algo_name == 'pnp_gd':
        proxy_fn = partial( tune_pnp_gd, problem=main_problem, denoiser=denoiser, tt=TIME_PER_TRIAL, verbose=False, lr_decay=1, converge_check=True, diverge_check=True)
        psp =   (
                    hp.uniform('eta', 1e-5, 1),
                    hp.uniform('dstrength', dstr_min, dstr_max)
                )
        return proxy_fn, psp
    if algo_name == 'pnp_sgd':
        proxy_fn = partial(tune_pnp_sgd, problem=main_problem, denoiser=denoiser, tt=TIME_PER_TRIAL, verbose=False, lr_decay=1, converge_check=True, diverge_check=True)
        psp =   (
                    hp.uniform('eta', 1e-5, 1),
                    scope.int(quniform('mini_batch_size', 1, 1000, q=1)),
                    hp.uniform('dstrength', dstr_min, dstr_max)
                ) 
        return proxy_fn, psp
    if algo_name == 'pnp_saga':
        proxy_fn = partial(tune_pnp_saga, problem=main_problem, denoiser=denoiser, tt=TIME_PER_TRIAL, verbose=False, lr_decay=1, converge_check=True, diverge_check=True)
        psp =   (
                    hp.uniform('eta', 1e-5, 1),
                    scope.int(quniform('mini_batch_size', 1, 1000, q=1)),
                    hp.uniform('dstrength', dstr_min, dstr_max)
                )
        return proxy_fn, psp
    if algo_name == 'pnp_sarah':
        proxy_fn = partial(tune_pnp_sarah, problem=main_problem, denoiser=denoiser, tt=TIME_PER_TRIAL, verbose=False, lr_decay=1, converge_check=True, diverge_check=True)
        psp =   (
                    hp.uniform('eta', 1e-5, 1),
                    scope.int(quniform('mini_batch_size', 1, 1000, q=1)),
                    scope.int(quniform('T2', 1, 1000, q=1)),
                    hp.uniform('dstrength', dstr_min, dstr_max)
                )
        return proxy_fn, psp
    if algo_name == 'pnp_svrg':
        proxy_fn = partial(tune_pnp_svrg, problem=main_problem, denoiser=denoiser, tt=TIME_PER_TRIAL, verbose=False, lr_decay=1, converge_check=True, diverge_check=True)
        psp =   (
                    hp.uniform('eta', 1e-5, 1),   
                    scope.int(quniform('mini_batch_size', 1, 1000, q=1)),
                    scope.int(quniform('T2', 1, 100, q=1)),
                    hp.uniform('dstrength', dstr_min, dstr_max)
                )
        return proxy_fn, psp
    else:
        raise Exception('Algorithm name "{0}" not found'.format(algo_name))    

output_fn = 'hyperparam-tuning/' + datetime.now().strftime('-%y-%m-%d-%H-%M') + '.csv'
os.makedirs(output_fn, exist_ok=True)

with open(output_fn, 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    for a in PROBLEM_LIST:
        for b in ALGO_LIST:
            for c in DENOISER_LIST:
                writer.writerow([a,b,c])

                p = get_problem(a)
                dnr = get_denoiser(c)
                proxy, pspace = get_proxy_pspace(p, b, dnr)

                pbar = tqdm(total=MAX_EVALS, desc="Hyperopt" + " " + a + " " + b + " " + c)
                trials = Trials()
                results = fmin(
                    proxy,
                    space=pspace,
                    algo=tpe.suggest,
                    trials=trials,
                    max_evals=MAX_EVALS
                )
                pbar.close()

                print(results)

                
