import torch
from torch import nn
from torch.nn import functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sinusoidal_embedding(n, d):
    embedding = torch.zeros((n, d))
    position = torch.arange(0, n, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d, 2).float() * (-torch.log(torch.tensor(10000.0)) / d))
    embedding[:, 0::2] = torch.sin(position * div_term)
    embedding[:, 1::2] = torch.cos(position * div_term)
    return embedding


class TabularAutoEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, time_emb_dim=100, n_steps=1000):
        super(TabularAutoEncoder, self).__init__()

        # Time embedding
        self.time_embedding = nn.Embedding(n_steps, time_emb_dim)
        self.time_embedding.weight.data = sinusoidal_embedding(n_steps, time_emb_dim)
        self.time_embedding.requires_grad_(False)

        # Time embedding transformations for different layers
        self.te1 = self._make_te(time_emb_dim, input_dim)
        self.te2 = self._make_te(time_emb_dim, 128)
        self.te3 = self._make_te(time_emb_dim, 64)
        self.te4 = self._make_te(time_emb_dim, 128)

        # Encoder
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)

        # Decoder
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128, output_dim)

    def forward(self, x, t):
        t_embed = self.time_embedding(t)  # (bs, time_emb_dim)

        # Encoding
        x = F.relu(self.fc1(x + self.te1(t_embed)))  # Add transformed time embedding to input
        x = F.relu(self.fc2(x + self.te2(t_embed)))

        # Decoding
        x = F.relu(self.fc3(x + self.te3(t_embed)))
        x = self.fc4(x + self.te4(t_embed))  # Final output, linear
        return x

    def _make_te(self, dim_in, dim_out):
        # A function to create transformation layers for time embeddings
        return nn.Sequential(nn.Linear(dim_in, dim_out), nn.ReLU(), nn.Linear(dim_out, dim_out))
