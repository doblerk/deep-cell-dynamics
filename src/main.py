import os
import json
import utils
import h5py
import model_arch
import argparse
import time, datetime

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms


########## MODEL TRAINING ##########
def train_vqvae(args):

    # Print arguments
    print('\n'.join('%s: %s' % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))

    # Set device to CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Benchmark multiple convolution algorithms and select the fastest
    cudnn.benchmark = True
    
    # Prepare data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop(32),
        transforms.Resize(96, interpolation=3),
        transforms.Normalize(tuple([0.2789]*args.in_channels), 
                             tuple([0.0597]*args.in_channels)),
    ])

    dataset = CroppedCells(root=args.data_path, depth=args.in_channels, transform=transform)

    training_loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=True,
        pin_memory=True,
    )

    data_variance = 0.011 # precomputed

    print('Data loaded successfully!')
    
    # Build network and move it to GPU
    model = model_arch.VQ_VAE(args.in_channels,
                              args.out_channels,
                              args.num_hiddens, 
                              args.num_residual_layers,
                              args.num_residual_hiddens,
                              args.num_embeddings,
                              args.embedding_dim,
                              args.commitment_cost).to(device)
    
    print('Network built successfully!')

    #### Build truncated VGG16 model to compute perceptual loss ####
    #### results were not satisfying ####
    # vgg_model = models.vgg16(pretrained=True).to(device)
    
    # pre_conv_layer = [nn.Conv2d(in_channels=args.in_channels,
    #                             out_channels=3,
    #                             kernel_size=3,
    #                             stride=1,
    #                             padding=1)]
    
    # pre_conv_layer.extend(list(vgg_model.features)[:10]) # to keep the first two blocks of convolutions
    
    # trunc_vgg_model = nn.Sequential(*pre_conv_layer).to(device)
    
    # for param in model.features[1:].parameters():
            # param.requires_grad = False

    # vgg_model = models.vgg16(pretrained=True)
    # vgg_model = models.vgg19(pretrained=True)
    # trunc_vgg_model = utils.CustomVGG19(args.in_channels).to(device)
    
    # Retrieve modules until the fourth conv2d block
    # trunc_vgg_model = nn.Sequential(*(list(vgg_model.children())[:-2][0][:10])).to(device)
    # trunc_vgg_model.eval()
    # trunc_vgg_model.requires_grad_(False)

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, amsgrad=False)

    print('Model parameters are ready!')

    # Optionally resume from checkpoint
    to_restore = {'epoch': 0}
    ckpt_path = os.path.join(args.output_dir, 'checkpoint.pth')
    if os.path.isfile(ckpt_path):
        ckpt = torch.torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        to_restore['epoch'] = ckpt['epoch']
        print(f'Checkpoint found at {args.output_dir} and restored.')
    start_epoch = to_restore['epoch']

    # Start training
    start_time = time.time()
    print(utils.header)
    model.train()
    train_res_recon_error = []
    train_res_perplexity = []
    for epoch in range(start_epoch, args.epochs):

        # Train one epoch
        # reconstructed_error, perplexity = train_one_epoch(epoch, training_loader, device, optimizer, model, trunc_vgg_model, data_variance, args)
        reconstructed_error, perplexity = train_one_epoch(epoch, training_loader, device, optimizer, model, data_variance, args)
        train_res_recon_error.append(reconstructed_error)
        train_res_perplexity.append(perplexity)

        # Save state_dict after each and x epoch
        save_dict = {
            'model_state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
        }
        torch.save(save_dict, os.path.join(args.output_dir, f'checkpoint.pth'))
        if epoch % args.saveckp_freq == 0:
            torch.save(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))

        # Write logs
        log_stats = {'Loss': train_res_recon_error[-1], 'Perplexity': train_res_perplexity[-1], 'Epoch': epoch}
        with open(os.path.join(args.output_dir, 'log.txt'), 'a') as f:
            f.write(json.dumps(log_stats) + '\n')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'Training time {total_time_str}')

def train_one_epoch(epoch, data_loader, device, optimizer, model, var, args):
    
    train_res_recon_error = []
    train_res_perplexity = []

    for it, (images, _) in enumerate(data_loader):

        # Move images to GPU
        images = images.to(device)

        # Initialize gradient
        optimizer.zero_grad()

        # Run model
        vq_loss, data_reconstructed, perplexity = model(images)
        reconstructed_error = F.mse_loss(data_reconstructed, images) / var
        loss = reconstructed_error + vq_loss
        # with torch.no_grad():
            # reconstructed_error_perceptual_loss = F.mse_loss(trunc_model(data_reconstructed), trunc_model(images)) / var
        # loss = reconstructed_error + reconstructed_error_perceptual_loss + vq_loss
        
        # Compute the gradient of the loss w.r.t. the parameters and update the weights
        loss.backward()
        optimizer.step()
        
        # Append outputs
        train_res_recon_error.append(reconstructed_error.item())
        train_res_perplexity.append(perplexity.item())
        
    print(f'Epoch [{epoch}/{args.epochs}] | Reconstruction error: {np.mean(train_res_recon_error):.3f} | Perplexity: {np.mean(train_res_perplexity):.3f}')

    return np.mean(train_res_recon_error), np.mean(train_res_perplexity)


class CroppedCells(torch.utils.data.Dataset):

    def __init__(self, root: str, depth: int, transform: transforms = None):
        self.root = root
        self.depth = depth 
        self.transform = transform
        
        self.n_frames = 0
        self.len = 0

        self.fill_size()

        assert self.n_frames >= self.depth

    def fill_size(self):
        with h5py.File(self.root, 'r') as hf:
            tensor_all = hf['default']
            self.len = len(tensor_all)
            assert self.len > 0
            first_tensor = tensor_all[0]
            self.n_frames = first_tensor.shape[2]
        
    def create_img_tensor(self, data):
        start_frame = int( (self.n_frames - self.depth) / 2 )
        new_tensor = data[:,:,start_frame:start_frame+self.depth]
        if self.transform is not None:
            return list((self.transform(new_tensor), 0))
        else:
            return list((new_tensor, 0))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        with h5py.File(self.root, 'r') as hf:
            tensor = hf['default'][idx]
        return self.create_img_tensor(tensor)

    def __len__(self):
        return self.len



########## CONSTRUCT ARGUMENT PARSER ##########
def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
    parser.add_argument('--data_path', type=str, help='Path to data')
    parser.add_argument('--in_channels', default=3, type=int, help='Number of input channels of the decoder')
    parser.add_argument('--out_channels', default=3, type=int, help='Number of output channels of the decoder')
    parser.add_argument('--num_hiddens', default=128, type=int, help='Number of hiddens')
    parser.add_argument('--num_residual_layers', default=2, type=int, help='Number of residual layers')
    parser.add_argument('--num_residual_hiddens', default=32, type=int, help='Number of residual hiddens')
    parser.add_argument('--num_embeddings', default=512, type=int, help='number of embeddings K')
    parser.add_argument('--embedding_dim', default=64, type=int, help='Dimensionality of each latent embedding vector')
    parser.add_argument('--commitment_cost', default=0.25, type=float, help='Beta parameter in the loss function')
    parser.add_argument('--lr', default=3e-4, type=float, help='Learning rate')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training')
    parser.add_argument('--saveckp_freq', default=20, type=int, help='Save checkpoint every x epochs')
    parser.add_argument('--output_dir', default='.', type=str, help='Path to save logs and checkpoints')
    return parser


########## DEFINE MAIN FUNCTION ##########
def main():
    parser = get_args_parser()
    args = parser.parse_args()
    train_vqvae(args)


if __name__ == '__main__':
    main()