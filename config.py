import os
import torch


class DefaultConfig(object):
    class_name = 'screw'
    data_root = r'D:\IMSN-YHM\dataset\mvtec_loco_anomaly_detection'
    # train_raw_data_root = os.path.join(data_root, class_name)
    # test_raw_data_root = os.path.join(data_root, "validation")

    train_raw_data_root = os.path.join(data_root, class_name, 'train')
    validate_raw_data_root = os.path.join(data_root, class_name, 'validation')
    test_raw_data_root = os.path.join(data_root, class_name, 'test')
    load_model_path = r'./weights/'

    training_state_path = './temp/'
    measure_save_path = './measure/'

    backbone_name = 'Resnet34'
    is_STVT = ['ST', 'MAE', 'TR'][2]

    use_gpu = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_batch_size = 8
    print_freq = 20
    max_epoch = 501
    lr = 0.0001
    lr_decay = 0.90
    weight_decay = 1e-5
    momentum = 0.9
    nz = 100
    nc = 3
    ngf = 64

    def parse_model_root(self, dicts):
        for k, v in dicts.items():
            if hasattr(self, k):
                setattr(self, k, v)

    def parse(self, dicts):
        for k, v in dicts.items():
            if hasattr(self, k):
                setattr(self, k, v)
                data_root = r'D:\IMSN-YHM\dataset\mvtec_loco_anomaly_detection'
                setattr(self, 'train_raw_data_root', os.path.join(data_root, v, 'train'))
                setattr(self, 'test_raw_data_root', os.path.join(data_root, v, 'test'))
                setattr(self, 'validate_raw_data_root', os.path.join(data_root, v, 'validation'))


