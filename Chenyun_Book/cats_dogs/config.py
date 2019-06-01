#coding=utf-8
import warnings
import torch as t

class DefaultConfig(object):
    env = 'default' # visdom环境
    vis_port = 8097
    model = 'SqueezeNet'

    train_data_root = './data/train/'
    test_data_root = './data/test1'
    load_model_path = None #加载预训练模型的路径，为None表示不加载

    batch_size = 32
    use_gpu = True
    num_workers = 4
    print_freq = 20 # print info every N batch

    debug_file = './debug'
    result_file = 'result.csv'

    max_epoch = 10
    lr = 0.1
    lr_decay = 0.95
    weight_decay = 1e-4


    def _parse(self, kwards):
        for k, v in kwards.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attibute %s" %k)
            setattr(self, k, v)
        self.device = t.device('cuda')

        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                print(k, getattr(self, k))

opt = DefaultConfig()