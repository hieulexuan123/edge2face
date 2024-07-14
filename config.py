import torch

log_img_dir = 'log/images'
log_loss_dir = 'log/losses'
checkpoint_dir = 'checkpoints'
experiment_name = 'default_experiment'
dataset_name = 'edge2face'
num_threads = 4 #Threads for loading data
save_epoch_freq = 1
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

gan_mode = 'lsgan'
learning_rate = 0.0002
l1_lambda = 100
continue_train = False
load_epoch = 0
batch_size = 4
num_epochs = 100 #num of epochs without decay 
num_epochs_decay = 100 #num of epochs with decay
