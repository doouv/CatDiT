"""Condition prediction heads for LDM auxiliary losses."""

import math
import torch
from torch import nn
from src.models.components.equiformer_v2.so3 import SO3_Embedding, SO3_LinearV2
from src.models.components.equiformer_v2.transformer_block import FeedForwardNetwork

_AVG_NUM_NODES = 76.29003


class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep embedding → MLP, same as DiT."""

    def __init__(self, hidden_dim, frequency_embedding_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_dim, hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
        )
        self.frequency_embedding_dim = frequency_embedding_dim

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_dim)
        return self.mlp(t_freq)


class ConditionScalarHead(nn.Module):
    """Binding energy prediction from noisy SO3 latent embeddings.

    Timestep-conditioned classifier for classifier guidance.
    Adds sinusoidal t embedding to the scalar (l=0) channel of each node
    before predicting per-node energy contributions.

    Args:
        decoder: FeedForwardDecoder instance (for architecture hyperparameters)
    """

    def __init__(self, decoder):
        super().__init__()

        self.lmax_list = decoder.lmax_list
        self.sphere_channels = decoder.sphere_channels
        self.weight_init = decoder.weight_init

        # Timestep embedding → project to sphere_channels for l=0 injection
        self.t_embedder = TimestepEmbedder(decoder.sphere_channels)

        # Energy prediction network (node-level)
        self.energy_block = FeedForwardNetwork(
            decoder.sphere_channels,
            decoder.ffn_hidden_channels,
            1,  # scalar output per node
            decoder.lmax_list,
            decoder.mmax_list,
            decoder.SO3_grid,
            decoder.ffn_activation,
            decoder.use_gate_act,
            decoder.use_grid_mlp,
            decoder.use_sep_s2_act,
        )

        # Weight initialization (same logic as EquiformerV2._init_weights)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, SO3_LinearV2)):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if self.weight_init == "normal":
                std = 1 / math.sqrt(m.in_features)
                nn.init.normal_(m.weight, 0, std)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x_embedding, batch_idx, t):
        """
        Args:
            x_embedding: (n_nodes, num_sh, sphere_channels) SO3 embeddings
            batch_idx: (n_nodes,) batch assignment
            t: (bsz,) diffusion timestep per graph

        Returns:
            binding_energy: (bsz,) predicted binding energies in eV
        """
        num_graphs = batch_idx.max().item() + 1

        # Timestep conditioning: embed t and add to l=0 (scalar) channel
        t_emb = self.t_embedder(t)  # (bsz, sphere_channels)
        t_per_node = t_emb[batch_idx]  # (n_nodes, sphere_channels)
        x_embedding = x_embedding.clone()
        x_embedding[:, 0, :] = x_embedding[:, 0, :] + t_per_node  # inject into l=0

        # Wrap tensor as SO3_Embedding
        x = SO3_Embedding(
            num_graphs,
            self.lmax_list,
            self.sphere_channels,
            x_embedding.device,
            x_embedding.dtype,
        )
        x.embedding = x_embedding

        # Predict node-level energy contributions
        node_output = self.energy_block(x)
        node_energy = node_output.embedding.narrow(1, 0, 1).squeeze(-1).squeeze(-1)

        # Sum contributions per graph, normalized by avg num nodes (EquiformerV2 convention)
        binding_energy = torch.zeros(
            num_graphs,
            device=node_energy.device,
            dtype=node_energy.dtype
        ).scatter_add(0, batch_idx, node_energy)
        binding_energy = binding_energy / _AVG_NUM_NODES

        return binding_energy  # (bsz,) in eV
