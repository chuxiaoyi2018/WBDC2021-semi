import os
import pandas as pd
import numpy as np
import tensorflow as tf
import gc

from time import time, sleep
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names

from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OrdinalEncoder, StandardScaler

from evaluation import evaluate_deepctr
import sys
sys.path.append("../prepare/")
sys.path.append("../model/")
from mmoe import MMOE, MMOEFM
from final_feature_utils import add_origin_feature, preprocessing_feed_info, reduce_mem

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
    pca = arg_dict['pca']
    preference = arg_dict['preference']
    kmeans = arg_dict['kmeans']
    cross = arg_dict['cross']
    exporsure = arg_dict['exporsure']
    model_name = arg_dict['model_name']
    seed = arg_dict['seed']
    
    
    target = ["read_comment", "like", "click_avatar", "forward", "favorite", "comment", "follow"]
    more_target = ["read_comment", "like", "click_avatar", "forward", "favorite", "comment", "follow", "stay", "play"]
    tag_features = ['manual_keyword_list_1', 'machine_keyword_list_1', 'manual_tag_list_1', 'machine_tag_list_1'
            , 'manual_keyword_list_2', 'machine_keyword_list_2', 'manual_tag_list_2', 'machine_tag_list_2'
            , 'manual_keyword_list_3', 'machine_keyword_list_3', 'manual_tag_list_3', 'machine_tag_list_3']
    sparse_features = ['userid', 'feedid', 'authorid', 'bgm_song_id', 'bgm_singer_id'] + tag_features
    dense_features = ['videoplayseconds']
    merge_features = ['feedid', 'authorid', 'videoplayseconds', 'bgm_song_id', 'bgm_singer_id'] + tag_features

    if train_mode == 'debug':
        data = pd.read_csv(f'../../../wbdc2021/data/wedata/{competition}/user_action.csv', nrows=1000)
        test = pd.read_csv(f'../../../wbdc2021/data/wedata/{competition}/test_a.csv', nrows=1000)
    else:
        data = pd.read_csv(f'../../../wbdc2021/data/wedata/{competition}/user_action.csv')
        test = pd.read_csv(f'../../../wbdc2021/data/wedata/{competition}/test_a.csv')
        if drop_duplicate:
            data = data.drop_duplicates(['userid', 'feedid'], keep='last')

    feed = pd.read_csv(f'../../../wbdc2021/data/wedata/{competition}/feed_info.csv')
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
    train_model_input = {name: train[name] for name in feature_names}
    val_model_input = {name: val[name] for name in feature_names}
    userid_list = val['userid'].astype(str).tolist()
    
    test_model_input = {name: test[name] for name in feature_names}

    if more_tasks:
        # stay play
        for name in ['stay', 'play']:
            train[name] = np.log(train[name]/1000 + 1.0)
            mms = MinMaxScaler()
            train[name] = mms.fit_transform(np.array(train[name]).reshape(-1, 1)).reshape(-1,)
            
        train_labels = [train[y].values for y in more_target]
        val_labels = [val[y].values for y in more_target]
        num_tasks = 9
        loss_weights = [4,3,2,1,1,1,1,0.1,0.1]

    else:
        train_labels = [train[y].values for y in target]
        val_labels = [val[y].values for y in target]
    
    
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
    

    print(train_model.summary())
    opt = tf.train.AdamOptimizer(learning_rate=lr)
    train_model.compile(opt, loss='binary_crossentropy', loss_weights=loss_weights) 
    for epoch in range(epochs):
        history = train_model.fit(train_model_input, train_labels,
                                  batch_size=batch_size, epochs=1, verbose=2)

        val_pred_ans = train_model.predict(val_model_input, batch_size=batch_size)
        evaluate_deepctr(val_labels, val_pred_ans, userid_list, target)
        train_model.save(f'../../data/model/{save_name}_{epoch}')

    t1 = time()
    pred_ans = train_model.predict(test_model_input, batch_size=batch_size)
    t2 = time()
    print('4个目标行为%d条样本预测耗时（毫秒）：%.3f' % (len(test), (t2 - t1) * 1000.0))
    ts = (t2 - t1) * 1000.0 / len(test) * 2000.0
    print('4个目标行为2000条样本平均预测耗时（毫秒）：%.3f' % ts)

    # 5.生成提交文件
    for i, action in enumerate(target):
        test[action] = pred_ans[i]
    test['userid'] = submit['userid']
    test['feedid'] = submit['feedid']
    test[['userid', 'feedid'] + target].to_csv(f'../../data/submission/{save_name}.csv', index=None, float_format='%.6f')
    print('to_csv ok')

'''
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
    'pca' : True,
    'preference' : True,
    'cross' : 0,
    'kmeans' : True,
    'exporsure' :0,
    'model_name' : 'fm',
    'seed' : 1024
}
print("*"*100)
print("*"*100)
print(arg_dict)
print("*"*100)
print("*"*100)
train_sparse_task(arg_dict)
gc.collect()
'''

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
    'pca' : True,
    'preference' : True,
    'cross' : 0,
    'kmeans' : True,
    'exporsure' :0,
    'model_name' : 'mmoe',
    'seed' : 1124
}
print("*"*100)
print("*"*100)
print(arg_dict)
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
    'pca' : True,
    'preference' : True,
    'cross' : 0,
    'kmeans' : True,
    'exporsure' :0,
    'model_name' : 'mmoe',
    'seed' : 1324
}
print("*"*100)
print("*"*100)
print(arg_dict)
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
    'dnn_dropout' : 0.2,
    'loss_weights' : [4,3,2,1,1,1,1], # 10:10:4:2:1:1:1
    'dnn_hidden_units' : (4096,),
    'use_bn' : False,
    'task_dnn_units' : None,
    'competition' : 'wechat_algo_data2',
    'save_name' : 'no_pca',
    'train_mode' : 'train',
    'add_feature' : True,
    'drop_duplicate' : True,
    'more_tasks' : True,
    'pca' : False,
    'preference' : True,
    'cross' : 0,
    'kmeans' : True,
    'exporsure' :0,
    'model_name' : 'mmoe',
    'seed' : 1424
}
print("*"*100)
print("*"*100)
print(arg_dict)
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
    'dnn_dropout' : 0.2,
    'loss_weights' : [4,3,2,1,1,1,1], # 10:10:4:2:1:1:1
    'dnn_hidden_units' : (4096,),
    'use_bn' : False,
    'task_dnn_units' : None,
    'competition' : 'wechat_algo_data2',
    'save_name' : 'no_cross',
    'train_mode' : 'train',
    'add_feature' : True,
    'drop_duplicate' : True,
    'more_tasks' : True,
    'pca' : True,
    'preference' : True,
    'cross' : -1,
    'kmeans' : True,
    'exporsure' :0,
    'model_name' : 'mmoe',
    'seed' : 1524
}
print("*"*100)
print("*"*100)
print(arg_dict)
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
    'dnn_dropout' : 0.25,
    'loss_weights' : [4,3,2,1,1,1,1], # 10:10:4:2:1:1:1
    'dnn_hidden_units' : (4096,),
    'use_bn' : False,
    'task_dnn_units' : None,
    'competition' : 'wechat_algo_data2',
    'save_name' : 'no_kmeans',
    'train_mode' : 'train',
    'add_feature' : True,
    'drop_duplicate' : True,
    'more_tasks' : True,
    'pca' : True,
    'preference' : True,
    'cross' : 0,
    'kmeans' : False,
    'exporsure' :0,
    'model_name' : 'mmoe',
    'seed' : 1624
}
print("*"*100)
print("*"*100)
print(arg_dict)
print("*"*100)
print("*"*100)
train_sparse_task(arg_dict)
gc.collect()