import argparse
import nni
from nni.experiment import Experiment
#设置参数搜索空间：
#choice为选择列表中的一个
#loguniform为对数均匀随机选择
#uniform为均匀随机选择
search_space = {
    # "learning_rate":{"_type": "choice", "_value": [0.01,0.0001,0.00001]},
    
    # "patch_len": {"_type": "choice", "_value": [27]},
    # "patch_stride": {"_type": "choice", "_value": [19,21,23,25,27]},
    # "d_model": {"_type": "choice", "_value": [64,128]},
    # "lr": {"_type": "loguniform", "_value": [0.0001, 0.0005]},
    # "dropout": {"_type": "choice", "_value": [0.1,0.2,0.3,0.4,0.5]},
    # "head_dropout": {"_type": "choice", "_value": [0.1,0.2,0.3,0.4,0.5]},
    # head_dropout
    # #"delta": {"_type": "loguniform", "_value": [0.0001,0.001]},
    # "batch_size": {"_type": "choice", "_value": [16,32,64]}
    # "patch_len": {"_type": "choice", "_value": [32,64,128,256,512]}
    # "d_model": {"_type": "choice", "_value": [256]},
    # "lr": {"_type": "loguniform", "_value": [0.0001,0.01]}
    # "dropout": {"_type": "loguniform", "_value": [0.1,0.5]},
    #"batch_size": {"_type": "choice", "_value": [64,128]},
    # "delta": {"_type": "loguniform", "_value": [0.0001,0.0015]}
    #"lamda": {"_type": "loguniform", "_value": [0.001,0.02]},# 0.004
    #"alpha": {"_type": "choice", "_value": [0.01,0.1,1.000001]},# 0.05
    #"m": {"_type": "choice", "_value": [0.3,0.4,0.5,0.6,0.7,0.8]}# 0.05
    # "std": {"_type": "loguniform", "_value": [0.2, 1.2]}
    #"latent_dim": {"_type": "choice", "_value": [16,32]}
    #"patch_stride": {"_type": "choice", "_value": [10,13]}
    # "e_layers": {"_type": "choice", "_value": [1,2,3]}
    # "batch_size": {"_type": "choice", "_value": [16]},
    # "lr": {"_type": "loguniform", "_value": [0.0001, 0.1]},
    # "l2": {"_type": "loguniform", "_value": [0.00001, 0.001]},
    # # "margin": {"_type": "choice", "_value": [1, 2, 10]},
    # "num_neg_samples_per_link": {"_type": "choice", "_value": [1, 8, 32]},
    # "dropout": {"_type": "uniform", "_value": [0.1, 0.5]},
    # "edge_dropout": {"_type": "uniform", "_value": [0.1, 0.5]},
    # # "gnn_agg_type": {"_type": "choice", "_value": ["sum", "mlp", "gru"]},
    # "rel_emb_dim": {"_type": "choice", "_value": [32, 64]},
    # "attn_rel_emb_dim": {"_type": "choice", "_value": [ 32, 64]},
    # "emb_dim": {"_type": "choice", "_value": [32, 64]}
}

# 下面是一些NNI的设置
experiment = Experiment('local')
# 这里把之前的训练命令行写过来，同时可以把一些需要的但不是超参的argument加上，如数据集
experiment.config.trial_command ='sh /data/ZhaiMengjie/work/TaiYang/TinySalientSleepNet_taiyang_green/train.sh'
# 选择代码的目录，这里同目录就是一个.
experiment.config.trial_code_directory = '.'
# nni工作时log放哪里
experiment.config.experiment_working_directory = '/data/ZhaiMengjie/work/TaiYang/TinySalientSleepNet_taiyang_green/nni-experiments-250630'
# 使用刚刚的搜索空间
experiment.config.search_space = search_space
# 搜索模式
experiment.config.tuner.name = 'TPE'
experiment.config.tuner.class_args['optimize_mode'] = 'maximize'
# 做几次实验？
experiment.config.max_trial_number = 50
# 并行数
experiment.config.trial_concurrency = 1
# 一次最多跑多久？
experiment.config.max_trial_duration = '100h'
# 把刚刚的port拿来启动NNI
experiment.run(8080)