import os
import pandas as pd
import numpy as np
import tensorflow as tf
import gc

from time import time, sleep
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OrdinalEncoder, StandardScaler

import sys
os.chdir("/home/tione/notebook/wbdc2021-semi/src")
sys.path.append("prepare/")
from final_feature_utils import add_origin_feature, preprocessing_feed_info, reduce_mem
from inference_avg import submit_avg
sys.path.append("model/")
from mmoe import MMOE, MMOEFM
sys.path.append("train/")
from evaluation import evaluate_deepctr

from tensorflow.python.keras.initializers import RandomUniform, TruncatedNormal

# GPU相关设置
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# 设置GPU按需增长
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

def train_sparse_task(arg_dict):
    lr = arg_dict['lr']
    epochs = arg_dict['epochs']
    num_tasks = arg_dict['num_tasks']
    batch_size = arg_dict['batch_size']
    embedding_dim = arg_dict['embedding_dim']
    expert_dim = arg_dict['expert_dim']
    num_experts = arg_dict['num_experts']
    l2_reg_embedding = arg_dict['l2_reg_embedding']
    l2_reg_dnn = arg_dict['l2_reg_dnn']
    dnn_dropout = arg_dict['dnn_dropout']
    loss_weights = arg_dict['loss_weights']
    dnn_hidden_units = arg_dict['dnn_hidden_units']
    use_bn = arg_dict['use_bn']
    task_dnn_units = arg_dict['task_dnn_units']
    competition = arg_dict['competition']
    save_name = arg_dict['save_name']
    train_mode = arg_dict['train_mode']
    add_feature = arg_dict['add_feature']
    drop_duplicate = arg_dict['drop_duplicate']
    more_tasks = arg_dict['more_tasks']
    data = arg_dict['data']
    test = arg_dict['test']
    feed = arg_dict['feed']
    submit = arg_dict['submit']
    dense_features = arg_dict['dense_features']
    sparse_features = arg_dict['sparse_features']
    target = arg_dict['target']
    more_target = arg_dict['more_target']
    model_name = arg_dict['model_name']
    seed = arg_dict['seed']
    test_path = arg_dict['test_path']
    
    
    print('data.shape', data.shape)
    print('data.columns', data.columns.tolist())
    print('unique date_: ', data['date_'].unique())

    train = data[data['date_'] <= 14] if train_mode == 'train' else data[data['date_'] < 14]
    val = data[data['date_'] == 14]  # 第14天样本作为验证集

    # 2.count #unique features for each sparse field,and record dense feature field name
    fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].max() + 1, 
                                         embedding_dim=embedding_dim,
                                         embeddings_initializer = TruncatedNormal(mean=0.0, stddev=0.0001, seed=2020))
                              for feat in sparse_features] + [DenseFeat(feat, 1) for feat in dense_features]
    
    dnn_feature_columns = fixlen_feature_columns
    feature_names = get_feature_names(dnn_feature_columns)
    
    # 3.generate input data for model
    test_model_input = {name: test[name] for name in feature_names}

    if more_tasks:
        num_tasks = 9
    else:
        pass
    
    del train, val
    gc.collect()
    
    # 4.Define Model,train,predict and evaluate
    if model_name == 'fm':
        train_model = MMOEFM(dnn_feature_columns, num_tasks=num_tasks, expert_dim=expert_dim, 
                           dnn_hidden_units=dnn_hidden_units, task_dnn_units=task_dnn_units,
                           l2_reg_embedding=l2_reg_embedding, l2_reg_dnn=l2_reg_dnn,
                           dnn_dropout=dnn_dropout, use_bn=use_bn, seed=seed,
                           num_experts=num_experts, tasks=['binary' for _ in range(num_tasks)])
    elif model_name == 'mmoe':
        train_model = MMOE(dnn_feature_columns, num_tasks=num_tasks, expert_dim=expert_dim, 
                           dnn_hidden_units=dnn_hidden_units, task_dnn_units=task_dnn_units,
                           l2_reg_embedding=l2_reg_embedding, l2_reg_dnn=l2_reg_dnn,
                           dnn_dropout=dnn_dropout, use_bn=use_bn,
                           num_experts=num_experts, tasks=['binary' for _ in range(num_tasks)])
    

    # print(train_model.summary())
    t1 = time()
    # 加载第四个epoch的模型
    train_model.load_weights(f'/home/tione/notebook/wbdc2021-semi/data/model/{save_name}_3.h5')
    pred_ans_4 = train_model.predict(test_model_input, batch_size=batch_size)
    
    
    # 加载第五个epoch模型
    # 4.Define Model,train,predict and evaluate
    train_model.load_weights(f'/home/tione/notebook/wbdc2021-semi/data/model/{save_name}_4.h5')
    pred_ans_5 = train_model.predict(test_model_input, batch_size=batch_size)
    t2 = time()
    print('7个目标行为%d条样本预测耗时（毫秒）：%.3f' % (len(test), (t2 - t1) * 1000.0))
    
    # 用14天数据验证一下，看看有没有严重的错误
    # val_pred_ans = train_model.predict(val_model_input, batch_size=batch_size)
    # evaluate_deepctr(val_labels, val_pred_ans, userid_list, target)


    # 5.生成提交文件
    for i, action in enumerate(target):
        submit[action] = 0.6*pred_ans_4[i] + 0.4*pred_ans_5[i]
    submit[['userid', 'feedid'] + target].to_csv(f'/home/tione/notebook/wbdc2021-semi/data/submission/{save_name}.csv', index=None, float_format='%.6f')
    print('to_csv ok')
    return 

def main(argv):
    test_path = argv[2]
    # competition = 'wechat_algo_data2'
    # test_mode = 'test_a'
    # test_path = f'../../wbdc2021/data/wedata/{competition}/{test_mode}.csv'
    
    add_feature = True
    drop_duplicate = True
    pca = True
    preference = True
    cross = 0
    kmeans = True
    exporsure = 0
    competition = 'wechat_algo_data2'
    train_mode = 'inference'
    
    target = ["read_comment", "like", "click_avatar", "forward", "favorite", "comment", "follow"]
    more_target = ["read_comment", "like", "click_avatar", "forward", "favorite", "comment", "follow", "stay", "play"]
    tag_features = ['manual_keyword_list_1', 'machine_keyword_list_1', 'manual_tag_list_1', 'machine_tag_list_1'
            , 'manual_keyword_list_2', 'machine_keyword_list_2', 'manual_tag_list_2', 'machine_tag_list_2'
            , 'manual_keyword_list_3', 'machine_keyword_list_3', 'manual_tag_list_3', 'machine_tag_list_3']
    sparse_features = ['userid', 'feedid', 'authorid', 'bgm_song_id', 'bgm_singer_id'] + tag_features
    dense_features = ['videoplayseconds']
    merge_features = ['feedid', 'authorid', 'videoplayseconds', 'bgm_song_id', 'bgm_singer_id'] + tag_features

    if train_mode == 'debug':
        data = pd.read_csv(f'/home/tione/notebook/wbdc2021/data/wedata/{competition}/user_action.csv', nrows=1000)
        # test = pd.read_csv(f'../../wbdc2021/data/wedata/{competition}/{test_mode}.csv', nrows=1000)
        test = pd.read_csv(test_path, nrows=1000)
    else:
        data = pd.read_csv(f'/home/tione/notebook/wbdc2021/data/wedata/{competition}/user_action.csv')
        test = pd.read_csv(test_path)
        # test = pd.read_csv(f'../../wbdc2021/data/wedata/{competition}/{test_mode}.csv')
        if drop_duplicate:
            data = data.drop_duplicates(['userid', 'feedid'], keep='last')

    feed = pd.read_csv(f'/home/tione/notebook/wbdc2021/data/wedata/{competition}/feed_info.csv')
    feed[["bgm_song_id", "bgm_singer_id"]] += 1  # 0 用于填未知
    feed[["bgm_song_id", "bgm_singer_id", "videoplayseconds"]] = \
        feed[["bgm_song_id", "bgm_singer_id", "videoplayseconds"]].fillna(0)
    feed['bgm_song_id'] = feed['bgm_song_id'].astype('int64')
    feed['bgm_singer_id'] = feed['bgm_singer_id'].astype('int64')
    feed = preprocessing_feed_info(feed, sparse_features)
    
    data = data.merge(feed[merge_features], how='left', on='feedid')

    test['date_'] = 15
    test = test.merge(feed[merge_features], how='left',on='feedid')
    submit = test[['userid', 'feedid']]
    
    # 对幂律分布log norm 感觉基本上都是幂律分布
    data[dense_features] = np.log(data[dense_features] + 1.0)
    test[dense_features] = np.log(test[dense_features] + 1.0)
    
    
    # add all feature
    if add_feature:
        data, test, dense_features, sparse_features = add_origin_feature(data, test, dense_features, sparse_features, 
                                                                         unique=0, cross=cross, pca=pca, preference=preference, 
                                                                         kmeans=kmeans, date=True, exporsure=exporsure)
    

    arg_dict = {
        'lr' : 3e-4,
        'epochs' : 5,
        'num_tasks' : 7,
        'batch_size' : 8192,
        'embedding_dim' : 'auto',
        'expert_dim' : 256,
        'num_experts' : 4,
        'l2_reg_embedding' : 1e-5,
        'l2_reg_dnn' : 0,
        'dnn_dropout' : 0.2,
        'loss_weights' : [4,3,2,1,1,1,1], # 10:10:4:2:1:1:1
        'dnn_hidden_units' : (4096,),
        'use_bn' : False,
        'task_dnn_units' : None,
        'competition' : 'wechat_algo_data2',
        'save_name' : 'baseline_0.2_dropout',
        'train_mode' : 'train',
        'add_feature' : True,
        'drop_duplicate' : True,
        'more_tasks' : True,
        'data' : data,
        'test' : test,
        'feed' : feed,
        'submit' : submit,
        'dense_features' : dense_features,
        'sparse_features' : sparse_features,
        'target' : target,
        'more_target' : more_target,
        'model_name' : 'mmoe',
        'seed' : 3072,
        'test_path' : test_path
    }
    print("*"*100)
    print("*"*100)
    # print(arg_dict)
    print("*"*100)
    print("*"*100)
    train_sparse_task(arg_dict)
    gc.collect()
    
    arg_dict = {
        'lr' : 3e-4,
        'epochs' : 5,
        'num_tasks' : 7,
        'batch_size' : 8192,
        'embedding_dim' : 'auto',
        'expert_dim' : 256,
        'num_experts' : 4,
        'l2_reg_embedding' : 1e-5,
        'l2_reg_dnn' : 0,
        'dnn_dropout' : 0.1,
        'loss_weights' : [4,3,2,1,1,1,1], # 10:10:4:2:1:1:1
        'dnn_hidden_units' : (4096,),
        'use_bn' : False,
        'task_dnn_units' : None,
        'competition' : 'wechat_algo_data2',
        'save_name' : 'baseline_7_tasks',
        'train_mode' : 'train',
        'add_feature' : True,
        'drop_duplicate' : True,
        'more_tasks' : False,
        'data' : data,
        'test' : test,
        'feed' : feed,
        'submit' : submit,
        'dense_features' : dense_features,
        'sparse_features' : sparse_features,
        'target' : target,
        'more_target' : more_target,
        'model_name' : 'mmoe',
        'seed' : 2048,
        'test_path' : test_path
    }
    print("*"*100)
    print("*"*100)
    # print(arg_dict)
    print("*"*100)
    print("*"*100)
    train_sparse_task(arg_dict)
    gc.collect()

    arg_dict = {
        'lr' : 2e-4,
        'epochs' : 5,
        'num_tasks' : 7,
        'batch_size' : 8192,
        'embedding_dim' : 'auto',
        'expert_dim' : 256,
        'num_experts' : 4,
        'l2_reg_embedding' : 1e-5,
        'l2_reg_dnn' : 0,
        'dnn_dropout' : 0.1,
        'loss_weights' : [4,3,2,1,1,1,1], # 10:10:4:2:1:1:1
        'dnn_hidden_units' : (4096,),
        'use_bn' : False,
        'task_dnn_units' : None,
        'competition' : 'wechat_algo_data2',
        'save_name' : 'baseline_2e-4_lr',
        'train_mode' : 'train',
        'add_feature' : True,
        'drop_duplicate' : True,
        'more_tasks' : True,
        'data' : data,
        'test' : test,
        'feed' : feed,
        'submit' : submit,
        'dense_features' : dense_features,
        'sparse_features' : sparse_features,
        'target' : target,
        'more_target' : more_target,
        'model_name' : 'mmoe',
        'seed' : 4096,
        'test_path' : test_path
    }
    print("*"*100)
    print("*"*100)
    # print(arg_dict)
    print("*"*100)
    print("*"*100)
    train_sparse_task(arg_dict)
    gc.collect()
    
    
    
    # v100 训练的模型
    arg_dict = {
        'lr' : 3e-4,
        'epochs' : 5,
        'num_tasks' : 7,
        'batch_size' : 8192,
        'embedding_dim' : 64,
        'expert_dim' : 256,
        'num_experts' : 4,
        'l2_reg_embedding' : 1e-5,
        'l2_reg_dnn' : 0,
        'dnn_dropout' : 0.1,
        'loss_weights' : [4,3,2,1,1,1,1], # 10:10:4:2:1:1:1
        'dnn_hidden_units' : (4096,),
        'use_bn' : False,
        'task_dnn_units' : None,
        'competition' : 'wechat_algo_data2',
        'save_name' : 'fm',
        'train_mode' : 'train',
        'add_feature' : True,
        'drop_duplicate' : True,
        'more_tasks' : False,
        'data' : data,
        'test' : test,
        'feed' : feed,
        'submit' : submit,
        'dense_features' : dense_features,
        'sparse_features' : sparse_features,
        'target' : target,
        'more_target' : more_target,
        'model_name' : 'fm',
        'seed' : 1024,
        'test_path' : test_path
    }
    print("*"*100)
    print("*"*100)
    # print(arg_dict)
    print("*"*100)
    print("*"*100)
    train_sparse_task(arg_dict)
    gc.collect()
    
    
    arg_dict = {
        'lr' : 3e-4,
        'epochs' : 5,
        'num_tasks' : 7,
        'batch_size' : 8192,
        'embedding_dim' : 'auto',
        'expert_dim' : 256,
        'num_experts' : 4,
        'l2_reg_embedding' : 1e-5,
        'l2_reg_dnn' : 0,
        'dnn_dropout' : 0.15,
        'loss_weights' : [4,3,2,1,1,1,1], # 10:10:4:2:1:1:1
        'dnn_hidden_units' : (4096,),
        'use_bn' : False,
        'task_dnn_units' : None,
        'competition' : 'wechat_algo_data2',
        'save_name' : '7tasks',
        'train_mode' : 'train',
        'add_feature' : True,
        'drop_duplicate' : True,
        'more_tasks' : False,
        'data' : data,
        'test' : test,
        'feed' : feed,
        'submit' : submit,
        'dense_features' : dense_features,
        'sparse_features' : sparse_features,
        'target' : target,
        'more_target' : more_target,
        'model_name' : 'mmoe',
        'seed' : 1324,
        'test_path' : test_path
    }
    print("*"*100)
    print("*"*100)
    # print(arg_dict)
    print("*"*100)
    print("*"*100)
    train_sparse_task(arg_dict)
    gc.collect()

    
    arg_dict = {
        'lr' : 3e-4,
        'epochs' : 5,
        'num_tasks' : 7,
        'batch_size' : 8192,
        'embedding_dim' : 'auto',
        'expert_dim' : 256,
        'num_experts' : 4,
        'l2_reg_embedding' : 1e-5,
        'l2_reg_dnn' : 0,
        'dnn_dropout' : 0.3,
        'loss_weights' : [4,3,2,1,1,1,1], # 10:10:4:2:1:1:1
        'dnn_hidden_units' : (4096,),
        'use_bn' : False,
        'task_dnn_units' : None,
        'competition' : 'wechat_algo_data2',
        'save_name' : 'baseline_0.3_dropout',
        'train_mode' : 'train',
        'add_feature' : True,
        'drop_duplicate' : True,
        'more_tasks' : True,
        'data' : data,
        'test' : test,
        'feed' : feed,
        'submit' : submit,
        'dense_features' : dense_features,
        'sparse_features' : sparse_features,
        'target' : target,
        'more_target' : more_target,
        'model_name' : 'mmoe',
        'seed' : 2124,
        'test_path' : test_path
    }
    print("*"*100)
    print("*"*100)
    # print(arg_dict)
    print("*"*100)
    print("*"*100)
    train_sparse_task(arg_dict)
    gc.collect()
    
    arg_dict = {
        'lr' : 3e-4,
        'epochs' : 5,
        'num_tasks' : 7,
        'batch_size' : 8192,
        'embedding_dim' : 'auto',
        'expert_dim' : 256,
        'num_experts' : 4,
        'l2_reg_embedding' : 1e-5,
        'l2_reg_dnn' : 0,
        'dnn_dropout' : 0.1,
        'loss_weights' : [4,3,2,1,1,1,1], # 10:10:4:2:1:1:1
        'dnn_hidden_units' : (4096,),
        'use_bn' : False,
        'task_dnn_units' : None,
        'competition' : 'wechat_algo_data2',
        'save_name' : 'baseline_train',
        'train_mode' : 'train',
        'add_feature' : True,
        'drop_duplicate' : True,
        'more_tasks' : True,
        'data' : data,
        'test' : test,
        'feed' : feed,
        'submit' : submit,
        'dense_features' : dense_features,
        'sparse_features' : sparse_features,
        'target' : target,
        'more_target' : more_target,
        'model_name' : 'mmoe',
        'seed' : 1024,
        'test_path' : test_path
    }
    print("*"*100)
    print("*"*100)
    # print(arg_dict)
    print("*"*100)
    print("*"*100)
    train_sparse_task(arg_dict)
    gc.collect()
    

    arg_dict = {
        'lr' : 3e-4,
        'epochs' : 5,
        'num_tasks' : 7,
        'batch_size' : 8192,
        'embedding_dim' : 'auto',
        'expert_dim' : 256,
        'num_experts' : 4,
        'l2_reg_embedding' : 1e-5,
        'l2_reg_dnn' : 0,
        'dnn_dropout' : 0.15,
        'loss_weights' : [4,3,2,1,1,1,1], # 10:10:4:2:1:1:1
        'dnn_hidden_units' : (4096,),
        'use_bn' : False,
        'task_dnn_units' : None,
        'competition' : 'wechat_algo_data2',
        'save_name' : 'baseline',
        'train_mode' : 'train',
        'add_feature' : True,
        'drop_duplicate' : True,
        'more_tasks' : True,
        'data' : data,
        'test' : test,
        'feed' : feed,
        'submit' : submit,
        'dense_features' : dense_features,
        'sparse_features' : sparse_features,
        'target' : target,
        'more_target' : more_target,
        'model_name' : 'mmoe',
        'seed' : 1124,
        'test_path' : test_path
    }
    print("*"*100)
    print("*"*100)
    # print(arg_dict)
    print("*"*100)
    print("*"*100)
    train_sparse_task(arg_dict)
    gc.collect()
    

    arg_dict = {
        'lr' : 3e-4,
        'epochs' : 5,
        'num_tasks' : 7,
        'batch_size' : 8192,
        'embedding_dim' : 'auto',
        'expert_dim' : 256,
        'num_experts' : 4,
        'l2_reg_embedding' : 1e-5,
        'l2_reg_dnn' : 0,
        'dnn_dropout' : 0.4,
        'loss_weights' : [4,3,2,1,1,1,1], # 10:10:4:2:1:1:1
        'dnn_hidden_units' : (4096,),
        'use_bn' : False,
        'task_dnn_units' : None,
        'competition' : 'wechat_algo_data2',
        'save_name' : 'bseline_0.4_dropout',
        'train_mode' : 'train',
        'add_feature' : True,
        'drop_duplicate' : True,
        'more_tasks' : True,
        'data' : data,
        'test' : test,
        'feed' : feed,
        'submit' : submit,
        'dense_features' : dense_features,
        'sparse_features' : sparse_features,
        'target' : target,
        'more_target' : more_target,
        'model_name' : 'mmoe',
        'seed' : 1124,
        'test_path' : test_path
    }
    print("*"*100)
    print("*"*100)
    # print(arg_dict)
    print("*"*100)
    print("*"*100)
    train_sparse_task(arg_dict)
    gc.collect()

    arg_dict = {
        'lr' : 1e-4,
        'epochs' : 5,
        'num_tasks' : 7,
        'batch_size' : 8192,
        'embedding_dim' : 'auto',
        'expert_dim' : 256,
        'num_experts' : 4,
        'l2_reg_embedding' : 1e-5,
        'l2_reg_dnn' : 0,
        'dnn_dropout' : 0.1,
        'loss_weights' : [4,3,2,1,1,1,1], # 10:10:4:2:1:1:1
        'dnn_hidden_units' : (4096,),
        'use_bn' : False,
        'task_dnn_units' : None,
        'competition' : 'wechat_algo_data2',
        'save_name' : 'baseline_1e-4_lr',
        'train_mode' : 'train',
        'add_feature' : True,
        'drop_duplicate' : True,
        'more_tasks' : True,
        'data' : data,
        'test' : test,
        'feed' : feed,
        'submit' : submit,
        'dense_features' : dense_features,
        'sparse_features' : sparse_features,
        'target' : target,
        'more_target' : more_target,
        'model_name' : 'mmoe',
        'seed' : 1124,
        'test_path' : test_path
    }
    print("*"*100)
    print("*"*100)
    # print(arg_dict)
    print("*"*100)
    print("*"*100)
    train_sparse_task(arg_dict)
    gc.collect()
    
    arg_dict = {
        'lr' : 2e-4,
        'epochs' : 5,
        'num_tasks' : 7,
        'batch_size' : 8192,
        'embedding_dim' : 64,
        'expert_dim' : 256,
        'num_experts' : 4,
        'l2_reg_embedding' : 1e-5,
        'l2_reg_dnn' : 0,
        'dnn_dropout' : 0.1,
        'loss_weights' : [4,3,2,1,1,1,1], # 10:10:4:2:1:1:1
        'dnn_hidden_units' : (4096,),
        'use_bn' : False,
        'task_dnn_units' : None,
        'competition' : 'wechat_algo_data2',
        'save_name' : 'fm_lr2e-4',
        'train_mode' : 'train',
        'add_feature' : True,
        'drop_duplicate' : True,
        'more_tasks' : False,
        'data' : data,
        'test' : test,
        'feed' : feed,
        'submit' : submit,
        'dense_features' : dense_features,
        'sparse_features' : sparse_features,
        'target' : target,
        'more_target' : more_target,
        'model_name' : 'fm',
        'seed' : 1124,
        'test_path' : test_path
    }
    print("*"*100)
    print("*"*100)
    # print(arg_dict)
    print("*"*100)
    print("*"*100)
    train_sparse_task(arg_dict)
    gc.collect()


    arg_dict = {
        'lr' : 3e-4,
        'epochs' : 5,
        'num_tasks' : 7,
        'batch_size' : 8192,
        'embedding_dim' : 64,
        'expert_dim' : 256,
        'num_experts' : 4,
        'l2_reg_embedding' : 1e-5,
        'l2_reg_dnn' : 0,
        'dnn_dropout' : 0.3,
        'loss_weights' : [4,3,2,1,1,1,1], # 10:10:4:2:1:1:1
        'dnn_hidden_units' : (4096,),
        'use_bn' : False,
        'task_dnn_units' : None,
        'competition' : 'wechat_algo_data2',
        'save_name' : 'fm_dropout_0.3',
        'train_mode' : 'train',
        'add_feature' : True,
        'drop_duplicate' : True,
        'more_tasks' : False,
        'data' : data,
        'test' : test,
        'feed' : feed,
        'submit' : submit,
        'dense_features' : dense_features,
        'sparse_features' : sparse_features,
        'target' : target,
        'more_target' : more_target,
        'model_name' : 'fm',
        'seed' : 1124,
        'test_path' : test_path
    }
    print("*"*100)
    print("*"*100)
    # print(arg_dict)
    print("*"*100)
    print("*"*100)
    train_sparse_task(arg_dict)
    gc.collect()

    
    
    print("Begin Submit Avg")
    submit_path = '/home/tione/notebook/wbdc2021-semi/data/submission/'
    result = submit_avg(submit_path)
    result['userid'] = submit['userid']
    result['feedid'] = submit['feedid']
    result.to_csv(submit_path + 'result.csv', index=False)
    print("End Submit Avg")
    
    
if __name__ == "__main__":
    tf.app.run(main)