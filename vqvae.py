import torch
import torch.nn as nn
import torch.nn.functional as F

from abc import ABC, abstractmethod

class BaseModel(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def shared_eval(self, batch, optimizer, mode, comet_logger='None'):
        pass

    def configure_optimizers(self, lr=1e-3):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        return optimizer

class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv1d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv1d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False)
        )

    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens)
                                      for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)

        return F.relu(x)


class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, embedding_dim):
        super(Encoder, self).__init__()
        self._conv_1 = nn.Conv1d(in_channels=in_channels,
                                    out_channels=num_hiddens // 2,
                                    kernel_size=4,
                                    stride=2, padding=1)
        self._conv_2 = nn.Conv1d(in_channels=num_hiddens // 2,
                                    out_channels=num_hiddens,
                                    kernel_size=4,
                                    stride=2, padding=1)
        self._conv_3 = nn.Conv1d(in_channels=num_hiddens,
                                    out_channels=num_hiddens,
                                    kernel_size=3,
                                    stride=1, padding=1)
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                                num_hiddens=num_hiddens,
                                                num_residual_layers=num_residual_layers,
                                                num_residual_hiddens=num_residual_hiddens)

        self._pre_vq_conv = nn.Conv1d(in_channels=num_hiddens, out_channels=embedding_dim, kernel_size=1, stride=1)

    def forward(self, inputs):
        x = inputs.view([inputs.shape[0], 1, inputs.shape[-1]])

        x = self._conv_1(x)
        x = F.relu(x)

        x = self._conv_2(x)
        x = F.relu(x)

        x = self._conv_3(x)
        x = self._residual_stack(x)
        x = self._pre_vq_conv(x)

        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Decoder, self).__init__()
        self._conv_1 = nn.Conv1d(in_channels=in_channels,
                                    out_channels=num_hiddens,
                                    kernel_size=3,
                                    stride=1, padding=1)

        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                                num_hiddens=num_hiddens,
                                                num_residual_layers=num_residual_layers,
                                                num_residual_hiddens=num_residual_hiddens)

        self._conv_trans_1 = nn.ConvTranspose1d(in_channels=num_hiddens,
                                                out_channels=num_hiddens // 2,
                                                kernel_size=4,
                                                stride=2, padding=1)

        self._conv_trans_2 = nn.ConvTranspose1d(in_channels=num_hiddens // 2,
                                                out_channels=1,
                                                kernel_size=4,
                                                stride=2, padding=1)


    def forward(self, inputs):
        x = self._conv_1(inputs)

        x = self._residual_stack(x)

        x = self._conv_trans_1(x)
        x = F.relu(x)

        x = self._conv_trans_2(x)

        return torch.squeeze(x)


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1 / self._num_embeddings, 1 / self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True) + torch.sum(self._embedding.weight ** 2, dim=1) - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

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
        return loss, quantized.permute(0, 2, 1).contiguous(), perplexity, self._embedding.weight, encoding_indices, encodings


class vqvae(BaseModel):
    def __init__(self):
        super().__init__()
        num_hiddens = 128
        num_residual_layers = 2
        num_residual_hiddens = 64
        embedding_dim = 64
        num_embeddings = 256
        commitment_cost = 0.25

        self.vq = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.encoder = Encoder(1, num_hiddens, num_residual_layers, num_residual_hiddens, embedding_dim)
        self.decoder = Decoder(embedding_dim, num_hiddens, num_residual_layers, num_residual_hiddens)

    def shared_eval(self, batch, optimizer, mode):
        if mode == 'train':
            optimizer.zero_grad()
            z = self.encoder(batch)
            vq_loss, quantized, perplexity, embedding_weight, encoding_indices, encodings = self.vq(z)
            data_recon = self.decoder(quantized)
            recon_error = F.mse_loss(data_recon, batch)
            loss = recon_error + vq_loss
            loss.backward()
            optimizer.step()

        if mode == 'test':
            with torch.no_grad():
                z = self.encoder(batch)
                vq_loss, quantized, perplexity, embedding_weight, encoding_indices, encodings = self.vq(z)
                data_recon = self.decoder(quantized)
                recon_error = F.mse_loss(data_recon, batch)
                loss = recon_error + vq_loss

        return loss, vq_loss, recon_error, data_recon, perplexity, embedding_weight, encoding_indices, encodings
