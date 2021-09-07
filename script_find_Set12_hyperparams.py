from hyperopt import fmin, tpe, hp, Trials
from datetime import datetime
import csv
from hyperopt.hp import quniform
from hyperopt.pyll import scope
from tqdm import tqdm
from functools import *
import os
import glob
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(device)

from problems import *
from algorithms import *
from denoisers import *

PROBLEM_LIST = ['CSMRI', 'DeblurSR', 'PR']
ALGO_LIST = ['pnp_gd', 'pnp_sgd', 'pnp_saga', 'pnp_sarah', 'pnp_svrg']
DENOISER_LIST = ['NLM', 'CNN', 'BM3D', 'TV']

SNR_LIST = [-10, -5, 0, 5, 10, 15, 20, 25, 30]
ALPHA_LIST = [1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]
SET12_LIST = glob.glob('./data/Set12/*.png')

KERNEL = "Minimal"
IM_HEIGHT = 32
IM_WIDTH = 32
IM_PATH = './data/Set12/01.png'

TIME_PER_TRIAL = 20
MAX_EVALS = 100

eta_min, eta_max = 0, 100
mb_min, mb_max = 1, 100
T2_min, T2_max = 1, 100
dstr_min, dstr_max = 0, 2

def get_problem(prob_name, im_path, alpha, SNR):
    if prob_name == 'CSMRI':
        return CSMRI(img_path=im_path, H=256, W=256, sample_prob=alpha/10, snr=SNR)
    if prob_name == 'DeblurSR':
        scale = int(alpha*10)
        return Deblur(img_path=im_path, kernel=KERNEL, H=256, W=256, scale_percent=scale, snr=SNR)
    if prob_name == 'PR':
        return PhaseRetrieval(img_path=im_path, H=32, W=32, num_meas = int(alpha*32*32), snr=SNR)
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
                    hp.uniform('eta', eta_min, eta_max),
                    hp.uniform('dstrength', dstr_min, dstr_max)
                )
        return proxy_fn, psp
    if algo_name == 'pnp_sgd':
        proxy_fn = partial(tune_pnp_sgd, problem=main_problem, denoiser=denoiser, tt=TIME_PER_TRIAL, verbose=False, lr_decay=1, converge_check=True, diverge_check=True)
        psp =   (
                    hp.uniform('eta', eta_min, eta_max),
                    scope.int(quniform('mini_batch_size', mb_min, mb_max, q=1)),
                    hp.uniform('dstrength', dstr_min, dstr_max)
                ) 
        return proxy_fn, psp
    if algo_name == 'pnp_saga':
        proxy_fn = partial(tune_pnp_saga, problem=main_problem, denoiser=denoiser, tt=TIME_PER_TRIAL, verbose=False, lr_decay=1, converge_check=True, diverge_check=True)
        psp =   (
                    hp.uniform('eta', eta_min, eta_max),
                    scope.int(quniform('mini_batch_size', mb_min, mb_max, q=1)),
                    hp.uniform('dstrength', dstr_min, dstr_max)
                )
        return proxy_fn, psp
    if algo_name == 'pnp_sarah':
        proxy_fn = partial(tune_pnp_sarah, problem=main_problem, denoiser=denoiser, tt=TIME_PER_TRIAL, verbose=False, lr_decay=1, converge_check=True, diverge_check=True)
        psp =   (
                    hp.uniform('eta', eta_min, eta_max),
                    scope.int(quniform('mini_batch_size', mb_min, mb_max, q=1)),
                    scope.int(quniform('T2', T2_min, T2_max, q=1)),
                    hp.uniform('dstrength', dstr_min, dstr_max)
                )
        return proxy_fn, psp
    if algo_name == 'pnp_svrg':
        proxy_fn = partial(tune_pnp_svrg, problem=main_problem, denoiser=denoiser, tt=TIME_PER_TRIAL, verbose=False, lr_decay=1, converge_check=True, diverge_check=True)
        psp =   (
                    hp.uniform('eta', eta_min, eta_max), 
                    scope.int(quniform('mini_batch_size', mb_min, mb_max, q=1)),
                    scope.int(quniform('T2', T2_min, T2_max, q=1)),
                    hp.uniform('dstrength', dstr_min, dstr_max)
                )
        return proxy_fn, psp
    else:
        raise Exception('Algorithm name "{0}" not found'.format(algo_name))    

os.makedirs('hyperparam-tuning/', exist_ok=True)
output_fn = 'hyperparam-tuning/' + 'Set12-AllAlgo-AllProblem-AllDenoisers' + datetime.now().strftime('%y-%m-%d-%H-%M') + '.csv'
# os.makedirs(output_fn, exist_ok=True)

with open(output_fn, 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    for img in SET12_LIST:
        writer.writerow(['img: ', img])
        for a in PROBLEM_LIST:
            writer.writerow(['problem: ', a])
            for c in DENOISER_LIST:
                writer.writerow(['denoiser: ', c])
                for b in ALGO_LIST:
                    writer.writerow(['algo: ', b]) 
                    for alp in ALPHA_LIST:
                        writer.writerow(['alpha: ', alp])
                        writer.writerow(['snr', 'output PSNR'])
                        for snr in SNR_LIST:
                            p = get_problem(a, img, alp, snr)
                            dnr = get_denoiser(c)
                            proxy, pspace = get_proxy_pspace(p, b, dnr)

                            pbar = tqdm(total=MAX_EVALS, desc="Hyperopt" + " " + img + " " + str(snr) + " " + str(alp) + " " + a + " " + b + " " + c)
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
                            print('snr: ', snr, 'loss: ', trials.best_trial['result']['loss'])
                            writer.writerow([snr, trials.best_trial['result']['loss']])

                            for key in results:
                                print(key, results[key])

                
