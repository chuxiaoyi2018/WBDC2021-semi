import os
import pandas as pd
import numpy as np
import tensorflow as tf
import math

from time import time
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names

import sys

# import warnings
# warnings.filterwarnings("ignore")

action_list =  ["read_comment", "like", "click_avatar", "forward", "favorite", "comment", "follow"]

def avg_operate(submit_path):
    for i, temp_path in enumerate(os.listdir(submit_path)):
        path = submit_path + temp_path
        if i == 0:
            submit = pd.read_csv(path)
        else:
            temp_submit = pd.read_csv(path)
            for action in action_list:
                submit[action] = temp_submit[action] + submit[action]

    for action in action_list:
        submit[action] = submit[action] / len(os.listdir(submit_path))
    return submit

def harmonic_operate(submit_path):
    for i, temp_path in enumerate(os.listdir(submit_path)):
        path = submit_path + temp_path
        if os.path.splitext(path)[-1] == '.csv':
            if i == 0:
                submit = pd.read_csv(path)
                for action in action_list:
                    submit[action] = 1/submit[action]
            else:
                temp_submit = pd.read_csv(path)
                for action in action_list:
                    submit[action] = 1/temp_submit[action] + submit[action]

    for action in action_list:
        submit[action] = 1/(submit[action] / len(os.listdir(submit_path)))
    return submit


def geometric_operate(submit_path):
    for i, temp_path in enumerate(os.listdir(submit_path)):
        path = submit_path + temp_path
        if os.path.splitext(path)[-1] == '.csv':
            if i == 0:
                submit = pd.read_csv(path)
            else:
                temp_submit = pd.read_csv(path)
                for action in action_list:
                    submit[action] = temp_submit[action] * submit[action]

    for action in action_list:
        submit[action] = np.power(submit[action], 1/len(os.listdir(submit_path)))
    return submit

def submit_avg(submit_path):
    harmonic_submit = harmonic_operate(submit_path)
    geometric_submit = geometric_operate(submit_path)
    return (harmonic_submit + geometric_submit)/2