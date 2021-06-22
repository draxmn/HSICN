from __future__ import division
import gc
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import time
import datetime
import  os
import time
import scipy.io as sio
import dataset
#from dataset import DatasetFromHdf5
from resblock import resblock,conv_relu_res_relu_block
from utils import AverageMeter,initialize_logger,record_loss
from loss import rrmse_loss

def Main(opt):
    
    cudnn.benchmark = True
    # Define the dataset
    trainset = dataset.HS_multiscale_DSet(opt)
    print('The overall number of images:', len(trainset))

    # Define the dataloader
    dataloader = DataLoader(trainset, batch_size = opt.batch_size, shuffle = True, num_workers = opt.num_workers, pin_memory = True)
    

    # Count start time
    
    # Model               
    model = resblock(conv_relu_res_relu_block, 16, 3,25)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    if torch.cuda.is_available():
        model.cuda()
  
    # Parameters, Loss and Optimizer
    start_epoch = 0
    end_epoch = 1000
    init_lr = 0.0001
    iteration = 0
    record_test_loss = 1000
    criterion = rrmse_loss
    optimizer=torch.optim.Adam(model.parameters(), lr = opt.lr, betas = (opt.b1, opt.b2), weight_decay = opt.weight_decay)
    
    model_path = './models/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    loss_csv = open(os.path.join(model_path,'loss.csv'), 'w+')
    
    log_dir = os.path.join(model_path,'train.log')
    logger = initialize_logger(log_dir)

    # Resume
    resume_file = 'C:/Users/alawy/Desktop/Models/HCNN/1/files/models1/hscnn_5layer_dim10_7.pkl' 
    if resume_file:
        if os.path.isfile(resume_file):
            print("=> loading checkpoint '{}'".format(resume_file))
            checkpoint = torch.load(resume_file)
            start_epoch = checkpoint['epoch']
            iteration = checkpoint['iter']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
    prev_time = time.time()
    for epoch in range(opt.epochs):
        
        start_time = time.time()         
        #train_loss, iteration, lr, time_left = train(dataloader, model, criterion, optimizer, iteration, init_lr, end_epoch,epoch)
        
        losses = AverageMeter()
        for i, (images, labels) in enumerate(dataloader):
            labels = labels.cuda()
            images = images.cuda()
            images = Variable(images)
            labels = Variable(labels)    
        
        # Decaying Learning Rate
        
            lr = poly_lr_scheduler(optimizer, init_lr, iteration, max_iter=968000, power=1.5) 
            iteration = iteration + 1
        # Forward + Backward + Optimize       
            output = model(images)
        
            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
        
        # Calling the step function on an Optimizer makes an update to its parameters
            optimizer.step()
        
        #  record loss
            losses.update(loss.item())
            iters_done = epoch * len(dataloader) + i
            iters_left = end_epoch * len(dataloader) - iters_done
            time_left = datetime.timedelta(seconds = iters_left * (time.time() - prev_time))
            prev_time = time.time()
            train_loss=losses.avg
       # test_loss = validate(val_loader, model, criterion)
        
 
        
        # Save model

        #save_model(opt, (epoch + 1), (iters_done + 1), len(dataloader), generator)
        save_checkpoint(model_path, epoch, iteration, model, optimizer)

        end_time = time.time()
        epoch_time = end_time - start_time

        print("\r[Epoch %d/%d] [Batch %d/%d] [Total Loss: %.4f] Time_left: %s" % ((epoch), opt.epochs, i, len(dataloader), loss.item(), time_left))
        
        record_loss(loss_csv,epoch, iteration, epoch_time, lr, train_loss)     
        logger.info("Epoch [%d], Iter[%d], Time:%.9f, learning rate : %.9f, Train Loss: %.9f" %(epoch, iteration, epoch_time, lr, train_loss))
        gc.collect()
# Training 
def save_checkpoint(model_path, epoch, iteration, model, optimizer):
    """Save the checkpoint."""
    if (epoch % 5 == 0):
        state = {
                'epoch': epoch,
                'iter': iteration,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                }
        print('The trained model is successfully saved at epoch %d' % (epoch))
        torch.save(state, os.path.join(model_path, 'hscnn_5layer_dim10_%d.pkl' %(epoch)))

# Validate
def validate(val_loader, model, criterion):
    
    
    model.eval()
    losses = AverageMeter()

    for i, (input, target) in enumerate(val_loader):
        input = input.cuda()
        target = target.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)      
        loss = criterion(output, target_var)

        #  record loss
        losses.update(loss.item())

    return losses.avg

# Learning rate
def poly_lr_scheduler(optimizer, init_lr, iteraion, lr_decay_iter=1,
                      max_iter=100, power=0.9):
    """Polynomial decay of learning rate
        :param init_lr is base learning rate
        :param iter is a current iteration
        :param lr_decay_iter how frequently decay occurs, default is 1
        :param max_iter is number of maximum iterations
        :param power is a polymomial power

    """
    if iteraion % lr_decay_iter or iteraion > max_iter:
        return optimizer

    lr = init_lr*(1 - iteraion/max_iter)**power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


if __name__ == '__main__':
    main()
