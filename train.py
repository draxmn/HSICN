import argparse
import os

import main

if __name__ == "__main__":
    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    # Saving, and loading parameters
    parser.add_argument('--save_path', type = str, default = 'track1', choices = ['track1', 'track2'], help = 'saving mode, and by_epoch saving is recommended')
    parser.add_argument('--save_mode', type = str, default = 'epoch', help = 'saving mode, and by_epoch saving is recommended')
    parser.add_argument('--save_by_epoch', type = int, default = 5, help = 'interval between model checkpoints (by epochs)')
    parser.add_argument('--save_by_iter', type = int, default = 100000, help = 'interval between model checkpoints (by iterations)')
    parser.add_argument('--load_name', type = str, default = 'C:/Users/alawy/Desktop/Models/HCNN/1/files/models1/hscnn_5layer_dim10_7.pkl', help = 'load the pre-trained model with certain epoch') #/storage/notebooks/rw8hlfhqgth3gll/V1models/G_epoch40_bs6.pth
    # GPU parameters
    parser.add_argument('--multi_gpu', type = bool, default = False, help = 'True for more than 1 GPU')
    parser.add_argument('--gpu_ids', type = str, default = '0', help = 'gpu_ids: e.g. 0  0,1  0,1,2  use -1 for CPU')
    parser.add_argument('--cudnn_benchmark', type = bool, default = True, help = 'True for unchanged input data type')
    # Training parameters
    parser.add_argument('--epochs', type = int, default = 2000, help = 'number of epochs of training that ensures 100K training iterations')
    parser.add_argument('--batch_size', type = int, default = 6, help = 'size of the batches, 8 is recommended')
    parser.add_argument('--lr', type = float, default = 0.0001, help = 'Adam: learning rate')
    parser.add_argument('--b1', type = float, default = 0.9, help = 'Adam: decay of first order momentum of gradient')
    parser.add_argument('--b2', type = float, default = 0.999, help = 'Adam: decay of second order momentum of gradient')
    parser.add_argument('--weight_decay', type = float, default = 0, help = 'weight decay for optimizer')
    parser.add_argument('--lr_decrease_mode', type = str, default = 'epoch', help = 'lr decrease mode, by_epoch or by_iter')
    parser.add_argument('--lr_decrease_epoch', type = int, default = 200, help = 'lr decrease at certain epoch and its multiple')
    parser.add_argument('--lr_decrease_iter', type = int, default = 50000, help = 'lr decrease at certain epoch and its multiple')
    parser.add_argument('--lr_decrease_factor', type = float, default = 0.5, help = 'lr decrease factor')
    parser.add_argument('--num_workers', type = int, default = 2, help = 'number of cpu threads to use during batch generation')
    parser.add_argument('--lambda_half', type = float, default = 0.5, help = 'lambda_half for SubNet')
    # Initialization parameters
    parser.add_argument('--pad', type = str, default = 'reflect', help = 'pad type of networks')
    parser.add_argument('--activ', type = str, default = 'lrelu', help = 'activation type of networks')
    parser.add_argument('--norm', type = str, default = 'none', help = 'normalization type of networks')
    parser.add_argument('--in_channels', type = int, default = 3, help = 'input channels for generator')
    parser.add_argument('--out_channels', type = int, default = 25, help = 'output channels for generator')
    parser.add_argument('--start_channels', type = int, default = 64, help = 'start channels for generator')
    parser.add_argument('--init_type', type = str, default = 'xavier', help = 'initialization type of generator')
    parser.add_argument('--init_gain', type = float, default = 0.02, help = 'initialization gain of generator')
    # Dataset parameters
    parser.add_argument('--baseroot', type = str, default = 'C:/Users/alawy/Desktop/Models/data', help = 'baseroot')
    parser.add_argument('--crop_size', type = int, default = 256, help = 'crop size')
    opt = parser.parse_args()

    '''
    # ----------------------------------------
    #        Choose CUDA visible devices
    # ----------------------------------------
    if opt.multi_gpu == True:
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
        print('Multi-GPU mode, %s GPUs are used' % (opt.gpu_ids))
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print('Single-GPU mode')
    '''

    # ----------------------------------------
    #                 Trainer
    # ----------------------------------------
    main.Main(opt)
    
