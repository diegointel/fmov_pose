import torch
import torch.nn as nn
import numpy as np


# Positional encoding embedding. Code was taken from https://github.com/bmild/nerf.
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs["input_dims"]
        out_dim = 0
        if self.kwargs["include_input"]:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs["max_freq_log2"]
        N_freqs = self.kwargs["num_freqs"]

        if self.kwargs["log_sampling"]:
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2.0**0.0, 2.0**max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs["periodic_fns"]:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, input_dims=3):
    embed_kwargs = {
        "include_input": True,
        "input_dims": input_dims,
        "max_freq_log2": multires - 1,
        "num_freqs": multires,
        "log_sampling": True,
        "periodic_fns": [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)

    def embed(x, eo=embedder_obj):
        return eo.embed(x)

    return embed, embedder_obj.out_dim


class FourierEmbedding(nn.Module):
    def __init__(self, num_cam, embed_dim=128):
        super(FourierEmbedding, self).__init__()
        # monotonous in [-1/4, 1/4] period
        b = np.random.normal(loc=0, scale=1 / (4 * num_cam), size=[embed_dim, 1])
        self.b = nn.Parameter(torch.tensor(b).float(), requires_grad=False)

    def forward(self, cam_id):
        return torch.cat(
            [
                torch.sin((2.0 * np.pi * cam_id) @ self.b.T),
                torch.cos((2.0 * np.pi * cam_id) @ self.b.T),
            ],
            axis=-1,
        )


class OriginalFourierEmbedding(nn.Module):
    def __init__(self, embed_dim=256, scale=10):
        super(OriginalFourierEmbedding, self).__init__()
        b = np.random.normal(loc=0, scale=scale, size=[embed_dim, 1])
        self.a = torch.ones_like(torch.from_numpy(b[:, 0])).float().cuda()
        self.b = nn.Parameter(torch.tensor(b).float(), requires_grad=False)

    def forward(self, cam_id):
        return torch.cat(
            [
                self.a * torch.sin((2.0 * np.pi * cam_id) @ self.b.T),
                self.a * torch.cos((2.0 * np.pi * cam_id) @ self.b.T),
            ],
            axis=-1,
        ) / torch.linalg.norm(self.a)
