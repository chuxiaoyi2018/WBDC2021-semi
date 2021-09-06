#### 2021 WBDC 微信大数据挑战赛 总结

##### 【比赛网址】
https://algo.weixin.qq.com/

##### 【比赛任务】
* 多目标预测：给定user与feed，要求预测用户是否读评论、点赞、点击头像、收藏、转发、发表评论、关注
* 其中发生行为为1，未发生行为为0
* 是一个==点击率预测==任务（click through rate prediction）

##### 【训练集与测试集】
* 训练集：1~14天用户行为数据
* 测试集：15天用户行为数据，要求预测7个行为

##### 【比赛名次】
* 初赛A榜：0.675，70名左右
* 初赛B榜：0.671，64名
* 复赛A榜：0.701，55名左右
* 复赛B榜：0.700，40名

##### 【队伍名称】
* 夏天的第一顿小火锅

##### 【比赛关键】
* ==item冷启动问题==
    * 在初赛中，测试集有17%的feed在训练集中没有出现。在复赛中，有14%的feed在训练集中没有出现。
    * 因此用word2vec预训练测试集的数据会取得很好的效果（听群里面大佬说，必须加上第15天训练，用window=128）
* ==用户共现性==
    * 由于微信视频号的属性，会给你显示有哪些好友对该视频点了赞。你有朋友点赞，那么你也会更有可能点赞。因此使用用户共现矩阵构成的tf-idf特征与svd特征是一个强特。
* 总而言之：特征 > 调参 > 结构

##### 【收获】
* 完成了==常见的特征工程==
* 使用debug模式
* 将训练过程、特征工程、模型结构放在不同python文件中，工程化的第一步
* 尝试了大量的方法、模型、调参
* jupyter lab将命令行与写代码结合在一起，真的香
* 事后对于top方案的学习【未开始】

----------------------------------------
##### 【初赛特征】
* cross num：所有sparse id的二阶交叉特征
* today特征：userid/authorid 在今天所查看/被查看的feedid count、var、sum等特征
* pca特征：feed embedding经过pca降维之后得到的64维pca特征
* tags/keyword特征：将变长的tag/keyword只取前面三个，所构成的固定长度的var sparse特征
* kmeans特征：feed embedding经过kmeans聚类之后所构成的sparse id（种类为100）
* exporsure特征：userid feedid等id特征在14天内对应的曝光数
* history特征：userid在过去5天（或者7天）内每个action的sum、var、mean、median
* unique特征：userid/feedid/authorid在14天内的nunique
* preference特征：userid videoseconds的均值与方差
* new date：date%7，相当于引入星期的概念
* 负采样：在使用catboost的时候，听取另外一个小伙伴的建议，使用df.sample

#### 【复赛特征】
* pca特征：feed embedding经过pca降维之后得到的64维pca特征
* tags/keyword特征：将变长的tag/keyword只取前面三个，所构成的固定长度的var sparse特征
* kmeans特征：feed embedding经过kmeans聚类之后所构成的sparse id（种类为100）
* exporsure特征：userid feedid等id特征在14天内对应的曝光数
* unique特征：userid/feedid/authorid在14天内的nunique
* preference特征：userid videoseconds的均值与方差
* new date：date%7，相当于引入星期的概念
* 复赛令人幸喜的是基本没啥过拟合，因此在最后B榜的时候排名涨一截，猜想是没有用word2vec以及minmaxscale与labelencoder都保存下来的原因，因此对于数据的放缩与labelencoder在训练与推理是一致的
* mmoe:在复赛的时候只使用mmoe,进行了大量的调参尝试,最后感觉有用的也就是embedding='auto'
* 模型融合：使用之前比赛的融合方式，0.5 * 几何平均 + 0.5 * 算数平均


#### 【其他尝试】
* 弱监督学习：将15天推理出来的标签取sigmoid比较高(>0.9)的或者比较低（<0.1）的，拿到网络中进行训练，效果降低一截
* 联邦学习：将上一个action训练完的网络作为下一个action的初始化，没啥变化。或者每个action训练一个epoch之后权重平均在一起，也没有啥变化
* word2vec：使用在腾讯广告赛里面的做法，将第15天的看作-1，1~14取0.05作为-1，基本没啥变化。与强特差了一点
* today feedid sequence：在今天看的所有feed中随机取50个feed作为边长特征输入进去，有提升，但是后来还是放弃了
* description：截取description的前面100个，有提升，但是由于爆内存，放弃了
* DIN、DSIN：各种尝试没啥变化，放弃了
* 三阶cross：没啥变化
* tag len、description len等等长度特征：没啥变化
* doc2vec特征：没啥变化
* window特征:将所有的特征全部转化为历史特征,全部使用n day=7,全部不使用统计类特征. 没啥变化,甚至有下降
* 各种尝试transformer,layernorm,only-scale layernorm,batchnorm,残差结构等等,均没啥用
