import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

# VAE architecture
class VAE(nn.Module):
    def __init__(self, latent_dim=20):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 400),
            nn.ReLU(),
            nn.Linear(400, latent_dim * 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 28 * 28),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=1)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon.view(-1, 1, 28, 28)

# Load pretrained model from GitHub
@st.cache_resource
def load_model():
    model = VAE()
    model.load_state_dict(torch.hub.load_state_dict_from_url(
    "https://raw.githubusercontent.com/MT121201/mnist-digit-gen/main/mnist_vae.pth",
    map_location=torch.device("cpu")
))
    model.eval()
    return model

# UI
st.title("🧠 MNIST Digit Generator")
digit = st.selectbox("Select digit to generate:", list(range(10)))

model = load_model()

# For now: sample 5 random vectors (no class conditioning)
with torch.no_grad():
    z = torch.randn(5, 20)
    images = model.decoder(z).view(-1, 1, 28, 28)

# Display
grid = make_grid(images, nrow=5, padding=2, normalize=True)
fig, ax = plt.subplots(figsize=(10, 2))
ax.imshow(grid.permute(1, 2, 0).squeeze(), cmap="gray")
ax.axis("off")
st.pyplot(fig)
