import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OrdinalEncoder
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from collections import defaultdict
import os 
import gc
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import time
import random
from sklearn.cluster import KMeans
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from scipy import stats
import pickle
import multiprocessing
from scipy.special import erfinv 

# os.chdir("/home/tione/notebook/wbdc2021-semi/src/prepare")

from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names, VarLenSparseFeat

# In[ ]:
def reduce_mem(df, cols):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in cols:
        col_type = df[col].dtypes
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('{:.2f} Mb, {:.2f} Mb ({:.2f} %)'.format(start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem))
    gc.collect()
    return df

def add_origin_feature(train, test, dense_features, sparse_features, unique, cross, pca, preference, kmeans, date, exporsure):
    train_len = len(train)
    test_len = len(test)
    data_and_test = pd.concat((train, test), sort=False).reset_index(drop=True)
    data_and_test['count'] = 1

    data_and_test = reduce_mem(data_and_test, dense_features+sparse_features)
            
    if date:
        print("Begin date")
        data_and_test, sparse_features = convert_date(data_and_test, sparse_features)
    if exporsure!=-1:
        print("Begin Add Exporusre")
        if exporsure==0:
            data_and_test, dense_features = more_exposure(data_and_test, dense_features)
        elif exporsure==1:
            data_and_test, sparse_features = more_exposure(data_and_test, sparse_features)
        data_and_test = reduce_mem(data_and_test, dense_features)
    if unique!=-1:
        print("Begin Unique")
        if unique==0:
            data_and_test, dense_features = add_nunique_feature(data_and_test, dense_features)
        elif unique==1:
            data_and_test, sparse_features = add_nunique_feature(data_and_test, sparse_features)
    if cross!=-1:
        print("Begin Id Cross 2")
        if cross==0:
            data_and_test, dense_features = cross_2_feature(data_and_test, dense_features)
        elif cross==1:
            data_and_test, sparse_features = cross_2_feature(data_and_test, sparse_features)
        data_and_test = reduce_mem(data_and_test, dense_features)
    if pca:
        print("Begin PCA")
        data_and_test, dense_features = add_pca_features(data_and_test, dense_features)
        data_and_test = reduce_mem(data_and_test, dense_features)
    if preference:
        print("Begin preference")
        data_and_test, dense_features = add_preference_feature(data_and_test, dense_features)
        data_and_test = reduce_mem(data_and_test, dense_features)
    if kmeans:
        print("Begin KMeans")
        data_and_test, sparse_features = kmeans_features(data_and_test, sparse_features)

    # data_and_test = reduce_mem(data_and_test, dense_features+sparse_features)
    # 1.fill nan dense_feature and do simple Transformation for dense features
    data_and_test = data_and_test.fillna(0, )

    # nan补零 因为存在std
    data_and_test = data_and_test.drop(columns=['count'])
    
    
    # label encode
    print("Begin Label Encoder")
    for feat in sparse_features:
        if feat!='userid' and feat!='new_date_' and feat!='kmeans_label':
            pkl_file = open(f'/home/tione/notebook/wbdc2021-semi/data/encoder/Departure_encoder_{feat}.pkl', 'rb')
            lbe = pickle.load(pkl_file) 
            pkl_file.close()
            data_and_test[feat] = lbe.transform(data_and_test[feat].astype(int)) + 1
        else:
            lbe = LabelEncoder()
            data_and_test[feat] = lbe.fit_transform(data_and_test[feat].astype(int)) + 1
        data_and_test[feat] = data_and_test[feat].astype(int)
    data_and_test = reduce_mem(data_and_test, dense_features + sparse_features)
    
    # MinMaxScale
    print("Begin Minmax Scaler")
    for feat in dense_features:
        if 'pca' not in feat:
            if not os.path.exists(f'/home/tione/notebook/wbdc2021-semi/data/minmaxscaler/Departure_encoder_{feat}.pkl'):
                mms = MinMaxScaler(feature_range=(0, 1))
                data_and_test[feat] = mms.fit_transform(data_and_test[feat].to_numpy().reshape(-1, 1)).reshape(-1,)
                pkl_file = open(f'/home/tione/notebook/wbdc2021-semi/data/minmaxscaler/Departure_encoder_{feat}.pkl', 'wb')
                pickle.dump(mms, pkl_file)
                pkl_file.close()
            else:
                pkl_file = open(f'/home/tione/notebook/wbdc2021-semi/data/minmaxscaler/Departure_encoder_{feat}.pkl', 'rb')
                mms = pickle.load(pkl_file) 
                pkl_file.close()
                data_and_test[feat] = mms.transform(data_and_test[feat].to_numpy().reshape(-1, 1)).reshape(-1,)
                
    data_and_test = reduce_mem(data_and_test, dense_features + sparse_features)
    print("End feature")
    data, test = data_and_test.iloc[:train_len].reset_index(drop=True), data_and_test.iloc[train_len:].reset_index(
        drop=True)
    
    # data = reduce_mem(data, dense_features+sparse_features)
    # test = reduce_mem(test, dense_features+sparse_features)
    return data, test, dense_features, sparse_features


def kmeans_features(data, sparse_features):
    dataset_path = '/home/tione/notebook/wbdc2021-semi/data/'
    if not os.path.exists(dataset_path + 'kmeans.npy'):
        embed_dict = np.load(dataset_path + 'feed_embedding_dict.npy', allow_pickle=True).item()
        embed_value = np.array([value for value in embed_dict.values()])
        kmeans = KMeans(n_clusters=100, random_state=2021, init='k-means++', max_iter=100).fit(embed_value)
        kmeans_dict = {key:value for key,value in zip(embed_dict.keys(), kmeans.labels_)}
        feedid_lst = np.array(data['feedid'])
        data['kmeans_label'] = [kmeans_dict[feedid] for feedid in feedid_lst]
        np.save(dataset_path + 'kmeans.npy', kmeans_dict)
    else:
        kmeans_dict = np.load(dataset_path + 'kmeans.npy', allow_pickle=True).item()
        data['kmeans_label'] = [kmeans_dict[feedid] for feedid in np.array(data['feedid'])]
    sparse_features += ['kmeans_label']
    return data, sparse_features

# 添加baseline里面的unique等特征
def add_nunique_feature(data, dense_features):
    for f1, f2 in [
        ['userid', 'feedid'],
        ['userid', 'authorid']
    ]:
        data['{}_in_{}_nunique'.format(f1, f2)] = data.groupby(f2)[f1].transform('nunique').astype(int)

        data['{}_in_{}_nunique'.format(f2, f1)] = data.groupby(f1)[f2].transform('nunique').astype(int)
        
        dense_features += ['{}_in_{}_nunique'.format(f1, f2)]
        dense_features += ['{}_in_{}_nunique'.format(f2, f1)]
        
    return data, dense_features
    

def cross_2_feature(data, dense_features):
    
    # userid feedid 之间的交叉可以去掉
    id_lst_1 = ['userid']
    id_lst_2 = ['authorid', 'new_date_']
    for i, id_1 in enumerate(id_lst_1):
        for id_2 in id_lst_2:
            cross_count_name = id_1 + '_' + id_2 + '_' + 'count'
            data[cross_count_name] = data.groupby([id_1, id_2])['count'].transform('count')
            data[cross_count_name] = data[cross_count_name].astype(int)
            dense_features += [cross_count_name]
                
    id_lst_1 = ['feedid']
    id_lst_2 = ['authorid', 'new_date_']
    for i, id_1 in enumerate(id_lst_1):
        for id_2 in id_lst_2:
            cross_count_name = id_1 + '_' + id_2 + '_' + 'count'
            data[cross_count_name] = data.groupby([id_1, id_2])['count'].transform('count')
            data[cross_count_name] = data[cross_count_name].astype(int)
            dense_features += [cross_count_name]
            
    id_lst_1 = ['authorid']
    id_lst_2 = ['new_date_', 'manual_keyword_list_1']
    for i, id_1 in enumerate(id_lst_1):
        for id_2 in id_lst_2:
            cross_count_name = id_1 + '_' + id_2 + '_' + 'count'
            data[cross_count_name] = data.groupby([id_1, id_2])['count'].transform('count')
            data[cross_count_name] = data[cross_count_name].astype(int)
            dense_features += [cross_count_name]
            
    return data, dense_features

def convert_date(data, sparse_features):
    converted_date = np.array(data['date_']) % 7
    data['new_date_'] = converted_date
    sparse_features += ['new_date_']
    return data, sparse_features

def compute_exposure(data, var_name, dense_features):
    # data[var_name + '_count'] = data[var_name].map(data[var_name].value_counts())
    data[var_name + '_count'] = data.groupby(var_name)[var_name].transform('count')
    data[var_name + '_count'] = data[var_name + '_count'].astype(int)
    # counter = np.array(data[var_name].value_counts().reset_index())
    # counter = {key:value for key,value in zip(counter[:,0], counter[:,1])}
    # data[var_name + '_count'] = np.array([counter[x] for x in np.array(data[var_name])])
    dense_features += [var_name + '_count']
    return data, dense_features

def more_exposure(data, dense_features):
    name_lst = ['userid', 'feedid', 'authorid', 'new_date_', 'manual_keyword_list_1', 
                'machine_keyword_list_1', 'manual_tag_list_1', 'machine_tag_list_1']
    for name in name_lst:
        data, dense_features = compute_exposure(data, name, dense_features)
    return data, dense_features


#####################################################################################################################################
### tag keyword
#####################################################################################################################################

def preprocessing_feed_info(feed_info, sparse_features):
    feed_info['manual_tag_list'] = feed_info['manual_tag_list'].fillna('-1')
    feed_info['manual_tag_len'] = feed_info.apply(lambda x: len(x['manual_tag_list'].split(';')), axis=1).astype(int)
    feed_info['manual_tag_list_1'] = feed_info.apply(lambda x: (x['manual_tag_list']+';-1;-1').split(';',3)[0], axis=1).astype(int)
    feed_info['manual_tag_list_2'] = feed_info.apply(lambda x: (x['manual_tag_list']+';-1;-1').split(';',3)[1], axis=1).astype(int)
    feed_info['manual_tag_list_3'] = feed_info.apply(lambda x: (x['manual_tag_list']+';-1;-1').split(';',3)[2], axis=1).astype(int)

    feed_info['machine_tag_list'] = feed_info['machine_tag_list'].fillna('-1')
    feed_info['machine_tag_len'] = feed_info.apply(lambda x: len(x['machine_tag_list'].split(';')), axis=1).astype(int)
    feed_info['machine_tag_list_1'] = feed_info.apply(lambda x: (x['machine_tag_list']+';-1;-1').split(';',1)[0].split(' ',1)[0], axis=1).astype(int)
    feed_info['machine_tag_list_2'] = feed_info.apply(lambda x: (x['machine_tag_list']+';-1;-1').split(';',2)[1].split(' ',1)[0], axis=1).astype(int)
    feed_info['machine_tag_list_3'] = feed_info.apply(lambda x: (x['machine_tag_list']+';-1;-1').split(';',3)[2].split(' ',1)[0], axis=1).astype(int)

    feed_info['manual_keyword_list'] = feed_info['manual_keyword_list'].fillna('-1')
    feed_info['manual_keyword_len'] = feed_info.apply(lambda x: len(x['manual_keyword_list'].split(';')), axis=1)
    feed_info['manual_keyword_list_1'] = feed_info.apply(lambda x: (x['manual_keyword_list']+';-1;-1').split(';',3)[0], axis=1).astype(int)
    feed_info['manual_keyword_list_2'] = feed_info.apply(lambda x: (x['manual_keyword_list']+';-1;-1').split(';',3)[1], axis=1).astype(int)
    feed_info['manual_keyword_list_3'] = feed_info.apply(lambda x: (x['manual_keyword_list']+';-1;-1').split(';',3)[2], axis=1).astype(int)

    feed_info['machine_keyword_list'] = feed_info['machine_keyword_list'].fillna('-1')
    feed_info['machine_keyword_len'] = feed_info.apply(lambda x: len(x['machine_keyword_list'].split(';')), axis=1).astype(int)
    feed_info['machine_keyword_list_1'] = feed_info.apply(lambda x: (x['machine_keyword_list']+';-1;-1').split(';',1)[0].split(' ',1)[0], axis=1).astype(int)
    feed_info['machine_keyword_list_2'] = feed_info.apply(lambda x: (x['machine_keyword_list']+';-1;-1').split(';',2)[1].split(' ',1)[0], axis=1).astype(int)
    feed_info['machine_keyword_list_3'] = feed_info.apply(lambda x: (x['machine_keyword_list']+';-1;-1').split(';',3)[2].split(' ',1)[0], axis=1).astype(int)
    
    for name in ['manual_tag_list_1', 'manual_tag_list_2', 'manual_tag_list_3', 
                 'machine_tag_list_1', 'machine_tag_list_2', 'machine_tag_list_3', 
                 'manual_keyword_list_1', 'manual_keyword_list_2', 'manual_keyword_list_3', 
                 'machine_keyword_list_1', 'machine_keyword_list_2', 'machine_keyword_list_3']:
        feed_info[name] = feed_info[name] + 1
    
    for name in ['description', 'ocr', 'asr', 'description_char', 'ocr_char', 'asr_char']:
        feed_info[name] = feed_info[name].fillna('-1')
        feed_info[name + '_len'] = feed_info.apply(lambda x: len(x[name].split(' ')), axis=1).astype(int)
        
    DATASET_PATH = '/home/tione/notebook/wbdc2021-semi/data/'
    if not os.path.exists('/home/tione/notebook/wbdc2021-semi/data/encoder'):
        os.mkdir('/home/tione/notebook/wbdc2021-semi/data/encoder')
    for name in sparse_features:
        if name != 'userid':
            if not os.path.exists(DATASET_PATH + f'encoder/Departure_encoder_{name}.pkl'):
                lbe = LabelEncoder()
                lbe.fit_transform(feed_info[name])

                output = open(DATASET_PATH + f'encoder/Departure_encoder_{name}.pkl', 'wb')
                pickle.dump(lbe, output)
                output.close()
    return feed_info

#####################################################################################################################################
### feed embedding
#####################################################################################################################################

def generate_feed_embedding_dict():
    feed_embed_path = '/home/tione/notebook/wbdc2021/data/wedata/wechat_algo_data2/feed_embeddings.csv'
    if not os.path.exists('/home/tione/notebook/wbdc2021-semi/data/feed_embedding_dict.npy'):
        str_feed_embed = pd.read_csv(feed_embed_path)
        str_feed_embed = np.array(str_feed_embed)

        str_feed_embed = str_feed_embed[:,1]
        feed_embed = np.zeros((str_feed_embed.shape[0], 512))
        for i, embed in tqdm(enumerate(str_feed_embed)):
            feed_embed[i, :] = [np.round(float(item), 6) for item in embed.split(' ', 512-1)]

        str_feed_embed = pd.read_csv(feed_embed_path)
        feed_id_lst = np.array(str_feed_embed)[:,0]
        feed_embed_dict = {}
        for feed_id, embed in tqdm(zip(feed_id_lst, feed_embed)):
            feed_embed_dict[feed_id] = embed
        np.save('/home/tione/notebook/wbdc2021-semi/data/feed_embedding_dict.npy', feed_embed_dict)
    else:
        feed_embed_dict = np.load('/home/tione/notebook/wbdc2021-semi/data/feed_embedding_dict.npy', allow_pickle=True).item()
    return feed_embed_dict
        
    
def generate_pca(feed_embed_dict, dim):
    dataset_path = '/home/tione/notebook/wbdc2021-semi/data/'
    if not os.path.exists(dataset_path + 'pca.npy'):
        feed_embed_value = [value for value in feed_embed_dict.values()]
        estimator = PCA(n_components=dim) 
        pca_embed = estimator.fit_transform(feed_embed_value)
        pca_dict = {key:value for key,value in zip(feed_embed_dict.keys(), pca_embed)}
        np.save(dataset_path + 'pca.npy', pca_dict)
    else:
        pca_dict = np.load(dataset_path + 'pca.npy', allow_pickle=True).item()
    return pca_dict

def add_pca_features(data, dense_features):
    dim = 16
    dataset_path = '/home/tione/notebook/wbdc2021-semi/data/'
    feed_embed_dict = generate_feed_embedding_dict()
    feed_embed_dict = np.load(dataset_path + 'feed_embedding_dict.npy',
                              allow_pickle=True).item()
    pca_dict = generate_pca(feed_embed_dict, dim)
    feed_id_lst = np.array(data['feedid'])
    feed_embed_lst = np.array([pca_dict[feedid] for feedid in feed_id_lst])
    for i in range(dim):
        data['pca_' + str(i)] = feed_embed_lst[:,i]
        dense_features += ['pca_' + str(i)]
    return data, dense_features


def add_preference_feature(df, dense_features):

    df['videoplayseconds_in_userid_mean'] = df.groupby('userid')['videoplayseconds'].transform('mean')
    df['videoplayseconds_in_userid_std'] = df.groupby('userid')['videoplayseconds'].transform('std')

    df['videoplayseconds_in_authorid_mean'] = df.groupby('authorid')['videoplayseconds'].transform('mean')
    df['videoplayseconds_in_authorid_std'] = df.groupby('authorid')['videoplayseconds'].transform('std')

    df['feedid_in_authorid_nunique'] = df.groupby('authorid')['feedid'].transform('nunique').astype(int)
    
    dense_features += ['videoplayseconds_in_userid_mean','videoplayseconds_in_authorid_mean', 
                       'feedid_in_authorid_nunique', 'videoplayseconds_in_userid_std', 'videoplayseconds_in_authorid_std']
    return df, dense_features