import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms


class BodyDataset(Dataset):
 
  def __init__(self, file_name):
    df = pd.read_csv(file_name)
 
    x = df.iloc[:,0:11].values
    y = df.iloc[:,11].values
 
    self.x_train = torch.tensor(x, dtype=torch.float32)
    self.y_train = torch.tensor(y, dtype=torch.float32)

    self.column_names = df.columns[:11]
 
  def __len__(self):
    return len(self.y_train)
   
  def __getitem__(self,idx):
    return self.x_train[idx],self.y_train[idx]


class Discriminator(nn.Module):

    def __init__(self, in_features):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.disc(x)
    

class Generator(nn.Module):

    def __init__(self, z_dim, in_featues):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 32),  # z_dim
            nn.LeakyReLU(0.1),
            nn.Linear(32, in_featues), # in_features -> length of generated data (1, 11)
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.gen(x)
    

if __name__ == "__main__":

    lr = 3e-4
    z_dim = 64
    in_features = 11
    batch_size = 2
    num_epochs = 2

    disc = Discriminator(in_features)
    gen = Generator(z_dim, in_features)

    dataset = BodyDataset("./body_performance/preprocessed_body_performance.csv")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    opt_disc = optim.Adam(disc.parameters(), lr=lr)
    opt_gen = optim.Adam(gen.parameters(), lr=lr)

    criterion = nn.BCELoss()

    for epoch in range(num_epochs):
       print(f"Epoch {epoch}")
       for batch_idx, (real, _) in enumerate(loader):
          
          real = real.view(-1, 11)                                              # (batch_size, 11) REAL DATA FROM DATASET
          batch_size = real.shape[0]                                            # 32

          ### Train Discriminator: max log(D(real)) + log(1 - D(G(z)))
          ### D -> discriminator || G -> generator || z -> random noise || real -> real data
          noise = torch.randn(batch_size, z_dim)                                # (32, 64)
          fake = gen(noise)                                                     # (batch_size, 11)
          disc_real = disc(real).view(-1)                                       # (1, 32)
          lossD_real = criterion(disc_real, torch.ones_like(disc_real))         # log(D(real))
          disc_fake = disc(fake).view(-1)                                       # (1, 32)
          lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))        # log(1 - D(G(z)))
          lossD = (lossD_real + lossD_fake) / 2
          disc.zero_grad()
          lossD.backward(retain_graph=True)
          opt_disc.step()

          ### Train Generator min log(1 - D(G(z))) <-> max log(D(G(z)))
          output = disc(fake).view(-1)                                          # (1, 32)
          lossG = criterion(output, torch.ones_like(output)) # log(D(real))
          gen.zero_grad()
          lossG.backward()
          opt_gen.step()


    noise = torch.randn(100, z_dim)
    output = gen(noise)
    result = pd.DataFrame(output.to_dense().detach().numpy())
    result.columns = dataset.column_names
    result.to_csv("preliminar_output.csv", index=False)
    print(output)  


########## TO DO TASKS

# * Forma de padronizar os dados de entrada (deve ser possível retornar ao normal)
# * Funções de ativação adequadas para cada rede
# * Como validar a melhora do model (acurácia)??