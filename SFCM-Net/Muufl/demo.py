# -*- coding:utf-8 -*-
import datetime
import argparse
import torch.nn as nn
import torch.utils.data as Data
from scipy.io import loadmat
from SFCM_Net import SFCM
import numpy as np
import time
import os
from utils import train_patch, setup_seed, output_metric, print_args, train_epoch, valid_epoch
import torch
import record

# -------------------------------------------------------------------------------
# Parameter Setting
parser = argparse.ArgumentParser("SFCM")
parser.add_argument('--gpu_id', default='0', help='gpu id')
parser.add_argument('--seed', type=int, default=0, help='number of seed')
parser.add_argument('--test_freq', type=int, default=20, help='number of evaluation')
parser.add_argument('--epoches', type=int, default=350, help='epoch number')  # Muufl 200
parser.add_argument('--learning_rate', type=float, default=5e-4, help='learning rate')  # diffGrad 1e-3
parser.add_argument('--gamma', type=float, default=0.9, help='gamma')
parser.add_argument('--weight_decay', type=float, default=0, help='weight_decay')
parser.add_argument('--dataset', choices=['Muufl', 'Trento', 'Houston', 'Augsburg'], default='Muufl',
                    help='dataset to use')
parser.add_argument('--num_classes', choices=[11, 6, 15, 7], default=11, help='number of classes')
parser.add_argument('--flag_test', choices=['test', 'train', 'pretrain'], default='train', help='testing mark')
parser.add_argument('--batch_size', type=int, default=64, help='number of batch size')
parser.add_argument('--patches1', type=int, default=6, help='number1 of patches')
parser.add_argument('--patches2', type=int, default=12, help='number2 of patches')
parser.add_argument('--patches3', type=int, default=18, help='number3 of patches')
parser.add_argument('--training_mode', choices=['one_time', 'ten_times', 'test_all', 'train_standard'],
                    default='one_time', help='training times')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)


def train_1times():
    setup_seed(args.seed)
    ITER = 10
    num_train = 20
    LabelPath = ''
    KAPPAlist = []
    OAlist = []
    AAlist = []
    TRAINING_TIME = []
    ELEMENT_ACC = np.zeros((ITER, args.num_classes))
    day = datetime.datetime.now()
    day_str = day.strftime('%m_%d_%H_%M')
    # -------------------------------------------------------------------------------
    # prepare data
    for i in range(ITER):
        if args.dataset == 'Houston':
            DataPath1 = r'../../dataset/Houston/Houston.mat'
            DataPath2 = r'../../dataset/Houston/LiDAR.mat'
            Data1 = loadmat(DataPath1)['img']  # (349,1905,144)
            Data2 = loadmat(DataPath2)['img']
            LabelPath = r'../../dataset/Houston/train_test/%d/train_test_gt_%d.mat' % (num_train, i+1)
            TrLabel_10TIMES = loadmat(LabelPath)['train_data']
            TsLabel_10TIMES = loadmat(LabelPath)['test_data']
        elif args.dataset == 'Trento':
            DataPath1 = r'./dataset/Trento/HSI_Trento.mat'
            DataPath2 = r'./dataset/Trento/Lidar_Trento.mat'
            Data1 = loadmat(DataPath1)['HSI_Trento']
            Data2 = loadmat(DataPath2)['Lidar_Trento']
            LabelPath = r'./dataset/Trento/train_test/%d/train_test_gt_%d.mat' % (num_train, i + 1)
            TrLabel_10TIMES = loadmat(LabelPath)['train_data']
            TsLabel_10TIMES = loadmat(LabelPath)['test_data']
        elif args.dataset == 'Muufl':
            DataPath1 = r'../dataset/Muufl/hsi.mat'
            DataPath2 = r'../dataset/Muufl/lidar_DEM.mat'
            Data1 = loadmat(DataPath1)['hsi']
            Data2 = loadmat(DataPath2)['lidar']
            LabelPath = r'../dataset/Muufl/train_test/%d/train_test_gt_%d.mat' % (num_train, i +1)
            TrLabel_10TIMES = loadmat(LabelPath)['train_data']
            TsLabel_10TIMES = loadmat(LabelPath)['test_data']

        print("**************************************************")
        print(LabelPath)
        print("**************************************************")
        Data1 = Data1.astype(np.float32)
        Data2 = Data2.astype(np.float32)
        patchsize1 = args.patches1  # input spatial size for 2D-CNN
        pad_width1 = np.floor(patchsize1 / 2)
        pad_width1 = int(pad_width1)  # 8
        patchsize2 = args.patches2  # input spatial size for 2D-CNN
        pad_width2 = np.floor(patchsize2 / 2)
        pad_width2 = int(pad_width2)  # 8
        patchsize3 = args.patches3  # input spatial size for 2D-CNN
        pad_width3 = np.floor(patchsize3 / 2)
        pad_width3 = int(pad_width3)  # 8
        TrainPatch11, TrainPatch21, TrainLabel = train_patch(Data1, Data2, patchsize1, pad_width1, TrLabel_10TIMES)
        TestPatch11, TestPatch21, TestLabel = train_patch(Data1, Data2, patchsize1, pad_width1, TsLabel_10TIMES)
        TrainPatch12, TrainPatch22, _ = train_patch(Data1, Data2, patchsize2, pad_width2, TrLabel_10TIMES)
        TestPatch12, TestPatch22, _ = train_patch(Data1, Data2, patchsize2, pad_width2, TsLabel_10TIMES)
        TrainPatch13, TrainPatch23, _ = train_patch(Data1, Data2, patchsize3, pad_width3, TrLabel_10TIMES)
        TestPatch13, TestPatch23, _ = train_patch(Data1, Data2, patchsize3, pad_width3, TsLabel_10TIMES)

        train_dataset = Data.TensorDataset(TrainPatch11, TrainPatch21, TrainPatch12, TrainPatch22, TrainPatch13,
                                           TrainPatch23, TrainLabel)
        train_loader = Data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_dataset = Data.TensorDataset(TestPatch11, TestPatch21, TestPatch12, TestPatch22, TestPatch13, TestPatch23,
                                          TestLabel)
        test_loader = Data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
        [m1, n1, l1] = np.shape(Data1)
        Data2 = Data2.reshape([m1, n1, -1])  # when lidar is one band, this is used
        height1, width1, band1 = Data1.shape
        height2, width2, band2 = Data2.shape
        # # data size
        print("height1={0},width1={1},band1={2}".format(height1, width1, band1))
        print("height2={0},width2={1},band2={2}".format(height2, width2, band2))
        # # -------------------------------------------------------------------------------
        # # create model
        model = SFCM(l1=band1, l2=band2, num_classes=args.num_classes)
        model = model.cuda()
        # # criterion
        criterion = nn.CrossEntropyLoss().cuda()
        # # optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epoches // 10, gamma=args.gamma)
        # # -------------------------------------------------------------------------------
        # # train and test
        if args.flag_test == 'train':
            BestAcc = 0
            val_acc = []
            print("start training")
            tic = time.time()
            for epoch in range(args.epoches):
                # train model
                model.train()
                train_acc, train_obj, tar_t, pre_t = train_epoch(model, train_loader, criterion, optimizer)
                OA1, AA1, Kappa1, CA1, matrix1 = output_metric(tar_t, pre_t)
                print("Epoch: {:03d} | train_loss: {:.4f} | train_OA: {:.4f} | train_AA: {:.4f} | train_Kappa: {:.4f}"
                      .format(epoch + 1, train_obj, OA1, AA1, Kappa1))
                scheduler.step()

                if (epoch % args.test_freq == 0) | (epoch == args.epoches - 1):
                    model.eval()
                    tar_v, pre_v = valid_epoch(model, test_loader, criterion)
                    OA2, AA2, Kappa2, CA2, matrix2 = output_metric(tar_v, pre_v)
                    val_acc.append(OA2)
                    print("Every 5 epochs' records:")
                    print("OA: {:.4f} | AA: {:.4f} | Kappa: {:.4f}".format(OA2, AA2, Kappa2))
                    print(CA2)
                    if OA2 > BestAcc:
                        torch.save(model.state_dict(), './SFCM_Net.pkl')
                        BestAcc = OA2

            toc = time.time()
            model.eval()
            model.load_state_dict(torch.load('./SFCM_Net.pkl'))
            tar_v, pre_v = valid_epoch(model, test_loader, criterion)
            # feature = model.features
            OA, AA, Kappa, CA, matrix = output_metric(tar_v, pre_v)
            print("Final records:")
            print("Maxmial Accuracy: %f, index: %i" % (max(val_acc), val_acc.index(max(val_acc))))
            print("OA: {:.4f} | AA: {:.4f} | Kappa: {:.4f}".format(OA, AA, Kappa))
            print(CA)
            print(matrix)
            print("Running Time: {:.2f}".format(toc - tic))
            print("**************************************************")
            print("Parameter:")
            print_args(vars(args))
            print("**************************************************")
            print(LabelPath)
            print("**************************************************")
        KAPPAlist.append(Kappa)
        OAlist.append(OA)
        AAlist.append(AA)
        TRAINING_TIME.append(toc - tic)
        ELEMENT_ACC[i, :] = CA
    record.record_output(LabelPath, OAlist, AAlist, KAPPAlist, ELEMENT_ACC, TRAINING_TIME,
                         './records/' + 'angle64' + day_str + '.txt')



if __name__ == '__main__':
    setup_seed(args.seed)
    if args.training_mode == 'one_time':
        train_1times()



