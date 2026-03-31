##########################################################################################
# Machine Environment Config

DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0


##########################################################################################
# Path Config

import os
import sys
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils


##########################################################################################
# import

import logging
from utils.utils import create_logger, copy_all_src

from TSPTester import TSPTester as Tester

from matplotlib import pyplot as plt
from matplotlib.patches import Circle
import matplotlib as mpl
mpl.style.use('default')
##########################################################################################
# parameters

env_params = {
    'problem_size': 20,
    'pomo_size': 20,
}

model_params = {
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128**(1/2),
    'encoder_layer_num': 6,
    'qkv_dim': 16,
    'head_num': 8,
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'eval_type': 'argmax',
}

tester_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'model_load': {
        'path': './result/20230823_200406_train__tsp_n20',  # directory path of pre-trained model and log files saved.
        'epoch': 10,  # epoch version of pre-trained model to laod.
    },
    'test_episodes': 1,
    'test_batch_size': 1,
    'augmentation_enable': False,
    'aug_factor': 8,
    'aug_batch_size': 1000,
}
if tester_params['augmentation_enable']:
    tester_params['test_batch_size'] = tester_params['aug_batch_size']

logger_params = {
    'log_file': {
        'desc': 'test__tsp_n20',
        'filename': 'run_log'
    }
}

##########################################################################################
# main

def main():
    if DEBUG_MODE:
        _set_debug_mode()

    create_logger(**logger_params)
    _print_config()

    tester = Tester(env_params=env_params,
                    model_params=model_params,
                    tester_params=tester_params)

    # copy_all_src(tester.result_folder)

    x_list, y_list, node_xy, radius = tester.run()
    # max_pomo_loc_list = max_pomo_loc_list.squeeze()
    # max_pomo_loc_list = np.concatenate((max_pomo_loc_list, node_xy[0][0][None, :]), 0)
    # max_aug_loc_list = max_aug_loc_list.squeeze()

    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111)
    for i in range(node_xy.shape[1]):
        circle = Circle(xy=node_xy[0][i], radius=radius[0][i], alpha=0.1, edgecolor='g')
        ax.add_patch(circle)
    ax.scatter(node_xy[0][0, 0], node_xy[0][0, 1], marker='s', s=30, c='b')
    ax.scatter(node_xy[0][1:, 0], node_xy[0][1:, 1], marker='*', s=1, c='r')
    list_x= []
    list_y=[]
    for i in range(len(x_list)):
        list_x.append(x_list[i])
        list_y.append(y_list[i])
    loc = np.concatenate((np.array(list_x)[:, None], np.array(list_y)[:, None]), -1)
    loc = np.concatenate((loc, node_xy[0][0][None, :]), 0)
    ax.scatter(loc[:, 0], loc[:, 1], marker='^', s=20, c='pink')
    ax.plot(loc[:, 0], loc[:, 1])
    plt.show()


def _set_debug_mode():
    global tester_params
    tester_params['test_episodes'] = 100


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]



##########################################################################################

if __name__ == "__main__":
    main()
