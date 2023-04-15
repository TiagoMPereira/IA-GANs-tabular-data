import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


# Define the discriminator network
class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size):
        print("DISCRIMINATOR")
        print(f"input: {input_size} = hidden: {hidden_size}")
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Define the generator network
class Generator(nn.Module):
    def __init__(self, noise_size, output_size, hidden_size):
        print("GENERATOR")
        print(f"noise: {noise_size} = output: {output_size} = hidden: {hidden_size}")
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.model(x)

# Define the GAN model
class GAN(nn.Module):
    def __init__(self, noise_size, real_data_size, hidden_size):
        super(GAN, self).__init__()
        self.noise_size = noise_size
        # Generator output == Discriminator input
        generator_output_size, discriminator_input_size = real_data_size, real_data_size
        self.generator = Generator(noise_size, generator_output_size, hidden_size)
        self.discriminator = Discriminator(discriminator_input_size, hidden_size)

    def forward(self, x):
        # Generate a sample from the generator
        z = torch.randn(x.shape[0], self.noise_size)
        fake_samples = self.generator(z)

        # Evaluate the discriminator on the real and fake samples
        real_scores = self.discriminator(x)
        fake_scores = self.discriminator(fake_samples)

        return fake_samples, real_scores, fake_scores
    

# Define the training loop
def train_gan(
    gan: GAN,
    dataloader: DataLoader,
    noise_size: int,
    epochs: int = 100,
    learning_rate: float = 3e-4
):
    # Initialize the loss functions and optimizers
    criterion = nn.BCELoss()
    generator_optimizer = optim.Adam(gan.generator.parameters(), lr=learning_rate)
    discriminator_optimizer = optim.Adam(gan.discriminator.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        print(f"Epoch {epoch}")
        for i, data in enumerate(dataloader):

            # Train the discriminator on real data
            discriminator_optimizer.zero_grad()
            real_samples = data.float()
            real_labels = torch.ones((real_samples.shape[0], 1))
            real_scores = gan.discriminator(real_samples)
            real_loss = criterion(real_scores, real_labels)
            real_loss.backward()

            # Train the discriminator on fake data
            z = torch.randn(real_samples.shape[0], noise_size)
            fake_samples = gan.generator(z)
            fake_labels = torch.zeros((real_samples.shape[0], 1))
            fake_scores = gan.discriminator(fake_samples.detach())
            fake_loss = criterion(fake_scores, fake_labels)
            fake_loss.backward()
            discriminator_optimizer.step()

            # Train the generator
            generator_optimizer.zero_grad()
            z = torch.randn(real_samples.shape[0], noise_size)
            fake_samples = gan.generator(z)
            fake_labels = torch.ones((real_samples.shape[0], 1))
            fake_scores = gan.discriminator(fake_samples)
            generator_loss = criterion(fake_scores, fake_labels)
            generator_loss.backward()
            generator_optimizer.step()


            # Print the losses
            if i % 100 == 0:
                epoch_loss_D = real_loss + fake_loss
                epoch_loss_G = generator_loss

                print(f"Epoch [{epoch+1}/{epochs}] Batch [{i}/{len(dataloader)}] "
                      f"Discriminator Loss: {epoch_loss_D:.4f} Generator Loss: {epoch_loss_G:.4f}")