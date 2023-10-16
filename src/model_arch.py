import utils
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchsummary import summary

########## ENCODER ##########
class Encoder(nn.Module):
    '''
    A class defining the encoder part of the network.

    Args:
        in_channels: number of channels in the input image.
        num_hiddens: number of channels produced by the residual block.
        num_residual_hiddens: number of channels produced by the convolution.
        num_residual_hiddens: number of channels produced by the convolution.
    '''
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Encoder, self).__init__()

        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens,
                                 kernel_size=4,
                                 stride=2, padding=1)

        self._conv_2 = nn.Conv2d(in_channels=num_hiddens,
                                 out_channels=num_hiddens,
                                 kernel_size=4,
                                 stride=2, padding=1)

        self._residual_stack = utils.ResidualStack(in_channels=num_hiddens,
                                                   num_hiddens=num_hiddens,
                                                   num_residual_layers=num_residual_layers,
                                                   num_residual_hiddens=num_residual_hiddens)

        self._post_encoder_conv = nn.Conv2d(in_channels=num_hiddens,
                                            out_channels=num_hiddens//4,
                                            kernel_size=2,
                                            stride=1, padding=0)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = F.relu(x)
        
        x = self._conv_2(x)
        x = F.relu(x)

        x = self._residual_stack(x)
        return self._post_encoder_conv(x)


########## VECTOR QUANTIZER ##########
class VectorQuantizer(nn.Module):
    '''
    A class defining the vector quantizer layer.
    
    Args:
        num_emmbeddings: number of embeddings K.
        embedding_dim: dimensionality of each latent embedding vector.
        commitment_cost: beta parameter in the loss function.
    '''
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float):
        super(VectorQuantizer, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._commitment_cost = commitment_cost
        
        # Create look-up table and initialize the weights with an uniform distribution
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        
    def forward(self, inputs: torch.Tensor):
        # Convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input from BHWC -> B*H*WC
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings


########## DECODER ##########
class Decoder(nn.Module):
    '''
    A class defining the decoder part of the network.

    Args:
        in_channels: number of channels in the input image.
        num_hiddens: number of channels produced by the residual block.
        num_residual_hiddens: number of channels produced by the convolution.
        num_residual_hiddens: number of channels produced by the convolution.
    '''
    def __init__(self, in_channels,  out_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Decoder, self).__init__()
        
        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens,
                                 kernel_size=4, #3
                                 stride=1, padding=2)
        
        self._residual_stack = utils.ResidualStack(in_channels=num_hiddens,
                                                   num_hiddens=num_hiddens,
                                                   num_residual_layers=num_residual_layers,
                                                   num_residual_hiddens=num_residual_hiddens)

        self._conv_trans_1 = nn.ConvTranspose2d(in_channels=num_hiddens, 
                                                out_channels=num_hiddens,
                                                kernel_size=4, 
                                                stride=2, padding=1)

        self._conv_trans_2 = nn.ConvTranspose2d(in_channels=num_hiddens, 
                                                out_channels=out_channels, 
                                                kernel_size=4, 
                                                stride=2, padding=1)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        
        x = self._residual_stack(x)
        
        x = self._conv_trans_1(x)
        x = F.relu(x)
        
        return self._conv_trans_2(x)


########## VQ-VAE MODEL ##########
class VQ_VAE(nn.Module):
    '''
    A class defining the VQ-VAE model.

    Args:
        in_channels: number of input channels.
        num_hiddens: number of channels produced by the residual block.
        num_residual_hiddens: number of channels produced by the convolution.
        num_residual_hiddens: number of channels produced by the convolution.
        num_emmbeddings: number of embeddings K.
        embedding_dim: dimensionality of each latent embedding vector.
        commitment_cost: beta parameter in the loss function.
    '''
    def __init__(self, 
                 in_channels,
                 out_channels,
                 num_hiddens, 
                 num_residual_layers, 
                 num_residual_hiddens, 
                 num_embeddings, 
                 embedding_dim, 
                 commitment_cost):
        super(VQ_VAE, self).__init__()

        self._encoder = Encoder(in_channels, 
                                num_hiddens, 
                                num_residual_layers, 
                                num_residual_hiddens)

        self._bottleneck = VectorQuantizer(num_embeddings, 
                                           embedding_dim, 
                                           commitment_cost)

        self._decoder = Decoder(embedding_dim,
                                out_channels, 
                                num_hiddens,
                                num_residual_layers, 
                                num_residual_hiddens)
        
    def forward(self, x):
        z = self._encoder(x)

        loss, quantized, perplexity, _ = self._bottleneck(z)

        x_reconstructed = self._decoder(quantized)
        
        return loss, x_reconstructed, perplexity

#Check model summary and output dimension of each layer
# device_name = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
# model = VQ_VAE(in_channels=15,
#                out_channels=15,
#                num_hiddens=256, 
#                num_residual_layers=2,
#                num_residual_hiddens=32,
#                num_embeddings=512,
#                embedding_dim=64,
#                commitment_cost=0.25).to(device_name)
# summary(model, (15, 96, 96))