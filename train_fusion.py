import numpy as np
import random

# Save training curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from common.arguments import parse_args # Load argument

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sys
import errno

from torch.utils.data import DataLoader, TensorDataset

from common.loss import *
from time import time
from common.utils import *
from common.logging import Logger
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from fusion_net.fusion import FusionNet

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

args = parse_args()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# initial setting
TIMESTAMP = "{0:%Y%m%dT%H-%M-%S/}".format(datetime.now())
# tensorboard
if not args.nolog:
    writer = SummaryWriter(args.log + '_' + TIMESTAMP)
    # writer.add_text('description', description)
    writer.add_text('command', 'python ' + ' '.join(sys.argv))
    # logging setting
    logfile = os.path.join(args.log + '_' + TIMESTAMP, 'logging.log')
    sys.stdout = Logger(logfile)
print("train fusion hypotheses network!!")
print('python ' + ' '.join(sys.argv))
print("CUDA Device Count: ", torch.cuda.device_count())
print(args)

manualSeed = 0  # default is 1
random.seed(manualSeed)
torch.manual_seed(manualSeed)
np.random.seed(manualSeed)
torch.cuda.manual_seed_all(manualSeed)

# if not assign checkpoint path, Save checkpoint file into log folder
if args.checkpoint == '':
    args.checkpoint = args.log + '_' + TIMESTAMP
try:
    # Create checkpoint directory if it does not exist
    os.makedirs(args.checkpoint)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise RuntimeError('Unable to create checkpoint directory:', args.checkpoint)
    
# set receptive_field as number assigned
receptive_field = args.number_of_frames
print('INFO: Receptive field: {} frames'.format(receptive_field))
if not args.nolog:
    writer.add_text(args.log + '_' + TIMESTAMP + '/Receptive field', str(receptive_field))
pad = (receptive_field - 1) // 2  # Padding on each side
min_loss = args.min_loss

# # feature fusion model (poseformer for feature extractor)
hidden_channel = 24  # 24, 64
model_fusionformer = FusionNet(num_frame=243, num_joints=17, hidden_channel=hidden_channel, out_channel=3)
model_fusionformer_test = FusionNet(num_frame=243, num_joints=17, hidden_channel=hidden_channel, out_channel=3)

# model
if torch.cuda.is_available():
    model_fusionformer = nn.DataParallel(model_fusionformer).cuda()
    model_fusionformer_test = nn.DataParallel(model_fusionformer_test).cuda()

# ================================================================================
model_params = 0
for parameter in model_fusionformer.parameters():
    model_params += parameter.numel()
print('INFO: Trainable parameter count:', model_params/1000000, 'Million')
if not args.nolog:
    writer.add_text(args.log+'_'+TIMESTAMP + '/Trainable parameter count', str(model_params/1000000) + ' Million')

# optimizer
scheduler_name = 'cyclicLR'
lr = args.learning_rate  # test: 0.001
optimizer = optim.AdamW(model_fusionformer.parameters(), lr=lr, weight_decay=1e-5, amsgrad=False)  # default: weight_decay=0.1 -> 0.01, 0.0004, test: 0.95
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=2e-4, max_lr=2e-3, cycle_momentum=False,
                                              step_size_up=8000, mode='exp_range', gamma=0.99996)

if args.resume:
    # load checkpoint fusion former
    # checkpoint = torch.load(os.path.join(args.checkpoint, args.evaluate), map_location=lambda storage, loc: storage)
    checkpoint = torch.load(os.path.join(args.checkpoint, args.resume), map_location=lambda storage, loc: storage)

    model_fusionformer.load_state_dict(checkpoint['model_pos'], strict=False)
    epoch = checkpoint['epoch']
    if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
        # train_generator.set_random_state(checkpoint['random_state'])
    else:
        print('WARNING: this checkpoint does not contain an optimizer state. The optimizer will be reinitialized.')
    if not args.coverlr:
        lr = checkpoint['lr']

# Load Path of Data D3DP hypotheses
# train_hypotheses = np.load("data/H5_d3dp_hypotheses_training.npz")['data_hypotheses']
# train_3d = np.load("data/H5_d3dp_3d_training_GT.npz", allow_pickle=True)['data_gt']
# test_hypotheses = np.load("data/H5_d3dp_hypotheses_testing.npz", allow_pickle=True)['data_hypotheses']
# test_3d = np.load("data/H5_d3dp_3d_testing_GT.npz", allow_pickle=True)['data_gt']

train_hypotheses = np.load("dataset/frame1_d3dp_5_hypotheses_training.npz")['data_hypotheses']
train_3d = np.load("dataset/frame1_d3dp_3d_training_gt.npz", allow_pickle=True)['data_gt']
test_hypotheses = np.load("dataset/frame1_d3dp_5_hypotheses_testing_augmented.npz", allow_pickle=True)['data_hypotheses']
test_3d = np.load("dataset/frame1_d3dp_3d_testing_gt_augmented.npz", allow_pickle=True)['data_gt']

# Data hypotheses train and test -----------------------------
data_train = torch.tensor(train_hypotheses, dtype=torch.float32)
data_test = torch.tensor(test_hypotheses, dtype=torch.float32)

# Data Ground Truth train and test ---------------------------
data_3d_train = torch.tensor(train_3d, dtype=torch.float32)
data_3d_test = torch.tensor(test_3d, dtype=torch.float32)

train_data_hypo = TensorDataset(data_train, data_3d_train)
test_data_hypo = TensorDataset(data_test, data_3d_test)

data_loader_train = DataLoader(train_data_hypo, batch_size=8, shuffle=True, num_workers=30)
data_loader_test = DataLoader(test_data_hypo, batch_size=4, shuffle=False, num_workers=30)

print("Train Hypotheses Shape:", data_train.shape)
print("Train 3D Shape:", data_3d_train.shape)
print("Test Hypotheses Shape:", data_test.shape)
print("Test 3D Shape:", data_3d_test.shape)

def train():
    quickdebug = False
    # quickdebug = True
    lr = args.learning_rate
    epoch = 0
#     initial_momentum = 0.1
#     final_momentum = 0.001
#     min_loss = args.min_loss

#     lr_decay = args.lr_decay
#     losses_3d_train = []
#     losses_3d_train_eval = []
#     losses_3d_valid = []

    # print(model_fusionformer)

    for epoch in range(args.epochs):
        start_time = time()
        model_fusionformer.train()
        iteration = 0

#         epoch_loss_3d_train = 0
#         # epoch_loss_3d_pos_train = 0

#         N = 0
#         print('Learning rate: ', '{:.6f}'.format(optimizer.param_groups[0]['lr']))

#         # Training start
#         for i, (predicted_3d_pos, inputs_3d) in enumerate(data_loader_train):
#             print("Training loop starting...")
#             print("Number of training batches:", len(data_loader_train))

#             if i % 1000 == 0:
#                 print(f"{i}/{len(data_loader_train)}")

#             # d3dp_start = time.time()
#             d3dp_start = time()
#             if torch.cuda.is_available():
#                 predicted_3d_pos = predicted_3d_pos.cuda()
#                 inputs_3d = inputs_3d.cuda()
#             d3dp_time = time() - d3dp_start

#             # Zero the root joint
#             inputs_3d[:, :, 0] = 0

#             optimizer.zero_grad()

#             # Time the FusionFormer model ---------------------
#             # fusion_start = time.time()
#             fusion_start = time()
#             predicted_3d_pos = predicted_3d_pos[:, :1, :, :, :]
#             predicted_3d_pos = model_fusionformer(predicted_3d_pos)
#             fusion_time = time() - fusion_start

#             # # Print timing
#             # print(f"D3DP: {d3dp_time:.4f}s, FusionFormer: {fusion_time:.4f}s")

#             # # loss function
#             loss_3d_pos = mpjpe(predicted_3d_pos, inputs_3d)
#             # loss_smoothL1 = SmoothL1Loss(predicted_3d_pos, inputs_3d)
#             # loss_mpjpe = mpjpe(predicted_3d_pos, inputs_3d)
#             # loss_3d_pos = MSE_Loss(predicted_3d_pos, inputs_3d)

#             loss_total = loss_3d_pos
#             # loss_total = alpha * loss_smoothL1 + (1-alpha) * loss_mpjpe

#             # loss_total.backward(loss_total.clone().detach())
#             loss_total.backward()

#             # # Clip gradients
#             # clip_grad_norm_(model_fusionformer.parameters(), max_norm=1.0)
#             optimizer.step()

#             # loss_total = torch.mean(loss_total)

#             epoch_loss_3d_train += inputs_3d.shape[0] * inputs_3d.shape[1] * loss_total.item()

#             N += inputs_3d.shape[0] * inputs_3d.shape[1]
#             # del inputs_3d, loss_3d_pos, predicted_3d_pos
#             # torch.cuda.empty_cache()

#             if scheduler_name == 'cyclicLR':
#                 scheduler.step()

#             iteration += 1

#             if quickdebug:
#                 if N == inputs_3d.shape[0] * inputs_3d.shape[1]:
#                     break

#         losses_3d_train.append(epoch_loss_3d_train / N)
#         # losses_3d_pos_train.append(epoch_loss_3d_pos_train / N)

#         # Testing and then select best epoch
#         with torch.no_grad():
#             model_fusionformer_test.load_state_dict(model_fusionformer.state_dict())
#             model_fusionformer_test.eval()

#             epoch_loss_3d_valid = None
#             N = 0
#             iteration = 0

#             for i, (predicted_3d_pos, inputs_3d) in enumerate(data_loader_test):
#                 if torch.cuda.is_available():
#                     predicted_3d_pos = predicted_3d_pos.cuda()
#                     inputs_3d = inputs_3d.cuda()

#                 inputs_3d[:, :, 0] = 0
#                 predicted_3d_pos = predicted_3d_pos[:, :1, :, :, :]

#                 predicted_3d_pos_test = model_fusionformer_test(predicted_3d_pos)

#                 predicted_3d_pos_test[:, :, 0, :] = 0

#                 error = mpjpe(predicted_3d_pos_test, inputs_3d)

#                 if iteration == 0:
#                     epoch_loss_3d_valid = inputs_3d.shape[0] * inputs_3d.shape[1] * error.clone()
#                 else:
#                     epoch_loss_3d_valid += inputs_3d.shape[0] * inputs_3d.shape[1] * error.clone()

#                 N += inputs_3d.shape[0] * inputs_3d.shape[1]

#                 iteration += 1

#                 if quickdebug:
#                     if N == inputs_3d.shape[0] * inputs_3d.shape[1]:
#                         break

#             losses_3d_valid.append(epoch_loss_3d_valid / N)

#         print(f"D3DP: {d3dp_time:.4f}s, FusionFormer: {fusion_time:.4f}s")
#         elapsed = (time() - start_time) / 60
#         # elapsed = (time.time() - start_time) / 60
#         print("Time: %.2fmin/epoch" % elapsed)
#         print('Epoch: %.d' % (epoch + 1),
#               '|Time: %.2fmin/epoch' % elapsed,
#               '|MPJPE_train: %.4fmm' % (losses_3d_train[-1] * 1000),
#               '|MPJPE_valid: %.4fmm' % (losses_3d_valid[-1] * 1000))

#         log_path = os.path.join(args.checkpoint, 'training_log.txt')
#         f = open(log_path, mode='a')
#         f.write('[%d] time %.2f 3d_train %f 3d_valid %f\n' % (
#             epoch + 1,
#             elapsed,
#             losses_3d_train[-1] * 1000,
#             losses_3d_valid[-1] * 1000
#         ))
#         f.close()
#         # update learning and epoch
#         # lr *= lr_decay
#         # for param_group in optimizer.param_groups:
#         #     param_group['lr'] *= lr_decay
#         epoch += 1

#         # Save checkpoint if necessary
#         chk_path = os.path.join(args.checkpoint, 'fusionNet.bin')
#         print('Saving checkpoint to', chk_path)

#         torch.save({
#             'epoch': epoch,
#             'lr': lr,
#             # 'random_state': train_generator.random_state(),
#             'optimizer': optimizer.state_dict(),
#             'model_pos': model_fusionformer.state_dict(),
#             'min_loss': min_loss
#             # 'model_traj': model_traj_train.state_dict() if semi_supervised else None,
#             # 'random_state_semi': semi_generator.random_state() if semi_supervised else None,
#         }, chk_path)

#         #### save best checkpoint
#         best_chk_path = os.path.join(args.checkpoint, 'best_fusionNet_epoch.bin')
#         if losses_3d_valid[-1] * 1000 < min_loss:
#             min_loss = losses_3d_valid[-1] * 1000
#             best_epoch = epoch
#             print("save best checkpoint")
#             torch.save({
#                 'epoch': epoch,
#                 'lr': lr,
#                 # 'random_state': train_generator.random_state(),
#                 'optimizer': optimizer.state_dict(),
#                 'model_pos': model_fusionformer.state_dict(),
#                 'min_loss': min_loss,
#                 # 'model_traj': model_traj_train.state_dict() if semi_supervised else None,
#                 # 'random_state_semi': semi_generator.random_state() if semi_supervised else None,
#             }, best_chk_path)

#             f = open(log_path, mode='a')
#             f.write('best epoch\n')
#             f.close()

#         if len(losses_3d_train) > 3:
#             plt.figure()
#             epoch_x = np.arange(3, len(losses_3d_train) + 1)

#             train_losses_np = np.array(losses_3d_train[2:])
#             valid_losses_np = np.array([loss.cpu().item() for loss in losses_3d_valid[2:]])

#             plt.plot(epoch_x, train_losses_np*1000, '--', color='C0')
#             plt.plot(epoch_x, valid_losses_np*1000, color='C1')
#             plt.legend(['3D Train', '3D Valid (Eval)'])
#             plt.ylabel('MPJPE (mm)')
#             plt.xlabel('Epoch')
#             plt.xlim((3, epoch))
#             os.makedirs(args.checkpoint, exist_ok=True)
#             plt.savefig(os.path.join(args.checkpoint, 'loss_3d.png'))
#             plt.close()
# # Training end


# # Evaluation Start
# def evaluate_model():
#     print('Evaluating pretrained model...')

#     from time import time
#     start_time = time()

#     losses_3d_valid = []
#     N = 0

#     # Load pretrained weights
#     checkpoint_path = os.path.join(args.checkpoint_fusion, 'v2_best_epoch.bin') # best_epoch residual 40.02
#     # checkpoint_path = os.path.join(args.checkpoint, 'v1_best_epoch.bin')  # test single frame
#     print("Loading checkpoint from:", checkpoint_path)
#     checkpoint = torch.load(checkpoint_path)

#     model_fusionformer_test.load_state_dict(checkpoint['model_pos'])
#     model_fusionformer_test.eval()

#     with torch.no_grad():
#         epoch_loss_3d_valid = 0

#         for predicted_3d_pos, inputs_3d in data_loader_test:
#             if torch.cuda.is_available():
#                 predicted_3d_pos = predicted_3d_pos[:, :, None].cuda()
#                 inputs_3d = inputs_3d.cuda()

#             inputs_3d[:, :, 0] = 0

#             predicted_3d_pos_test = model_fusionformer_test(predicted_3d_pos)
#             predicted_3d_pos_test[:, :, 0, :] = 0

#             error = mpjpe(predicted_3d_pos_test, inputs_3d)
#             epoch_loss_3d_valid += inputs_3d.shape[0] * inputs_3d.shape[1] * error.item()
#             N += inputs_3d.shape[0] * inputs_3d.shape[1]

#         mpjpe_valid = epoch_loss_3d_valid / N
#         losses_3d_valid.append(mpjpe_valid)

#     elapsed = (time() - start_time) / 60
#     print("Time: %.2fmin" % elapsed)
#     print('MPJPE_valid: %.4f mm' % (mpjpe_valid * 1000))

#     log_path = os.path.join(args.checkpoint, 'eval_log.txt')
#     with open(log_path, 'a') as f:
#         f.write('Eval - Time: %.2f min | MPJPE_valid: %.4f mm\n' % (elapsed, mpjpe_valid * 1000))

#     print("Evaluation Done!!")
# # Evaluation end


if __name__ == "__main__":
    train()
#     evaluate_model()