import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import sys
from datetime import datetime

import pandas as pd
from GAN.dataset import TabularDataset
from GAN.network.networks import GAN, train_gan
from GAN.utils.stats import kolmogorov_smirnov
from GAN.utils.scaler import Scaler



# Define the main function
if __name__ == '__main__':

    if len(sys.argv) == 1:
        now = datetime.now()
        id_ = now.strftime("%m_%d-%H_%M_%S")
    else:
        id_ = sys.argv[1]
    details = "generator > Linear,LeakyReLU,Dropout,Linear,LeakyReLU,Dropout,Linear,LeakyReLU,Dropout,Linear,Tanh\n" +\
              "discriminator > Linear,LeakyReLU,Linear,Sigmoid\n" +\
              ">>Normal scaler added<<\n"

    # Importing dataset
    df = pd.read_csv("./body_performance/preprocessed_body_performance.csv")
    X = df.iloc[:, :11]

    # Set the hyperparameters
    data_shape = X.shape[1]                 # 11
    noise_shape = 1024
    hidden_size = 512
    epochs = 100
    batch_size = 16
    learning_rate = 0.001

    # Initializing Dataset and DataLoader
    dataset = TabularDataset(data=X.values)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Training scaler
    scaler = Scaler()
    scaler.fit(X)

    # Create the GAN model
    gan = GAN(
        real_data_size=data_shape, noise_size=noise_shape,
        hidden_size=hidden_size
    )

    # Train the GAN model
    d_loss, g_loss = train_gan(
        gan=gan, dataloader=dataloader, noise_size=noise_shape, epochs=epochs,
        learning_rate=learning_rate, return_loss=True, scaler=scaler,
        data_columns=X.columns
    )

    # Generating syntethic data
    z = torch.randn(X.shape[0], noise_shape)
    fake_samples = gan.generator(z)
    fake = pd.DataFrame(data=fake_samples.detach().numpy(), columns=X.columns)
    fake = scaler.inverse_transform(fake)

    # Calculating KS metric
    metric = kolmogorov_smirnov(X, fake)

    # Saving values
    details += f"noise {noise_shape}, hidden {hidden_size}, epochs {epochs}, batch {batch_size}, lr {learning_rate}"
    with open(f'./results/{id_}_details.txt', 'w') as f:
        f.write(details)

    fake.to_csv(f'./results/{id_}_output.csv', index=False)
    metric.to_csv(f'./results/{id_}_ks.csv', index=False)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(d_loss, color="green")
    ax2.plot(g_loss, color="red")
    ax1.set_title("Discriminator loss")
    ax2.set_title("Generator loss")
    plt.suptitle(f"{id_}_loss")
    plt.savefig(f"./results/{id_}_loss.jpg")
    plt.show()
