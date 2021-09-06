# coding: utf-8
import time
from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_auc_score
import tensorflow as tf


def uAUC(labels, preds, user_id_list):
    """Calculate user AUC"""
    user_pred = defaultdict(lambda: [])
    user_truth = defaultdict(lambda: [])
    for idx, truth in enumerate(labels):
        user_id = user_id_list[idx]
        pred = preds[idx]
        truth = labels[idx]
        user_pred[user_id].append(pred)
        user_truth[user_id].append(truth)

    user_flag = defaultdict(lambda: False)
    for user_id in set(user_id_list):
        truths = user_truth[user_id]
        flag = False
        # 若全是正样本或全是负样本，则flag为False
        for i in range(len(truths) - 1):
            if truths[i] != truths[i + 1]:
                flag = True
                break
        user_flag[user_id] = flag

    total_auc = 0.0
    size = 0.0
    for user_id in user_flag:
        if user_flag[user_id]:
            auc = roc_auc_score(np.asarray(user_truth[user_id]), np.asarray(user_pred[user_id]))
            total_auc += auc 
            size += 1.0
    user_auc = float(total_auc)/size
    return user_auc


def compute_weighted_score(score_dict, weight_dict):
    '''基于多个行为的uAUC值，计算加权uAUC
    Input:
        scores_dict: 多个行为的uAUC值映射字典, dict
        weights_dict: 多个行为的权重映射字典, dict
    Output:
        score: 加权uAUC值, float
    '''
    score = 0.0
    weight_sum = 0.0
    for action in score_dict:
        weight = float(weight_dict[action])
        score += weight*score_dict[action]
        weight_sum += weight
    score /= float(weight_sum)
    score = round(score, 6)
    return score


def evaluate_deepctr(val_labels,val_pred_ans,userid_list,target):
    eval_dict = {}
    for i, action in enumerate(target):
        eval_dict[action] = uAUC(val_labels[i], val_pred_ans[i], userid_list)
    print(eval_dict)
    weight_dict = {"read_comment": 4, "like": 3, "click_avatar": 2, "favorite": 1, "forward": 1,
                   "comment": 1, "follow": 1}
    weight_auc = compute_weighted_score(eval_dict, weight_dict)
    print("Weighted uAUC: ", weight_auc)
    return weight_auc
    
def uAUC_one_task(labels, preds, user_id_list):
    from collections import defaultdict
    from sklearn.metrics import roc_auc_score
    """Calculate user AUC"""
    user_pred = defaultdict(lambda: [])
    user_truth = defaultdict(lambda: [])
    for idx, truth in enumerate(labels):
        user_id = user_id_list[idx]
        pred = preds[idx]
        truth = labels[idx]
        user_pred[user_id].append(pred)
        user_truth[user_id].append(truth)

    user_flag = defaultdict(lambda: False)
    for user_id in set(user_id_list):
        truths = user_truth[user_id]
        flag = False
        # 若全是正样本或全是负样本，则flag为False
        for i in range(len(truths) - 1):
            if truths[i] != truths[i + 1]:
                flag = True
                break
        user_flag[user_id] = flag

    total_auc = 0.0
    size = 0.0
    for user_id in user_flag:
        if user_flag[user_id]:
            auc = roc_auc_score(np.asarray(user_truth[user_id]), np.asarray(user_pred[user_id]))
            total_auc += auc
            size += 1.0
    user_auc = float(total_auc)/size
    return user_auc

#####################################################################################################################################
### train one task
#####################################################################################################################################
import tensorflow as tf

def train_one_task(test, i, train_model, lr, loss_weights, epochs, 
                   action, train_model_input, train_labels, batch_size, 
                   num_tasks, val_model_input, val_labels, userid_list, test_model_input, target):
    print("*"*100)
    print("*"*100)
    print(action)
    print("*"*100)
    print("*"*100)
    opt = tf.train.AdamOptimizer(learning_rate=lr)
    train_model.compile(opt, loss='binary_crossentropy', loss_weights=loss_weights) 
    # print(train_model.summary())
    best_uauc = 0
    for epoch in range(epochs):
        history = train_model.fit(train_model_input, train_labels,
                                  batch_size=batch_size, epochs=1, verbose=1)

        val_pred_ans = train_model.predict(val_model_input, batch_size=batch_size * num_tasks)
        # uauc = uAUC_one_task(val_labels[i], val_pred_ans[i], userid_list)
        # print('action ' + str(action) + ' valid uauc is ' + str(uauc))
        evaluate_deepctr(val_labels, val_pred_ans, userid_list, target)
        '''
        if uauc > best_uauc:
            train_model.save(f'../save/{action}_task')
        '''
    pred_ans = train_model.predict(test_model_input, batch_size=batch_size * 20)
    test[action] = pred_ans[i]
    return test


#####################################################################################################################################
### sample weights
#####################################################################################################################################

def get_sample_weights(data, num_tasks):
    sample_weight = np.array(data['date_']/data['date_'].max())
    return [sample_weight]*num_tasks