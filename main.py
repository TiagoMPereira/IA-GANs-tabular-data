import torch
from torch.utils.data import DataLoader

import pandas as pd
from GAN.dataset import TabularDataset
from GAN.network.networks import GAN, train_gan


# Define the main function
if __name__ == '__main__':

    # Importing dataset
    df = pd.read_csv("./body_performance/preprocessed_body_performance.csv")
    X = df.iloc[:, :11] 
        

    # Set the hyperparameters
    data_shape = X.shape[1]                 # 11
    noise_shape = 16
    hidden_size = 16
    epochs = 50
    batch_size = 16
    learning_rate = 0.001

    # Initializing Dataset and DataLoader
    dataset = TabularDataset(data=X.values)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Create the GAN model
    gan = GAN(real_data_size=data_shape, noise_size=noise_shape, hidden_size=hidden_size)

    # Train the GAN model
    train_gan(
        gan=gan, dataloader=dataloader, noise_size=noise_shape, epochs=epochs,
        learning_rate=learning_rate
    )

    z = torch.randn(100, noise_shape)
    fake_samples = gan.generator(z)
    fake = pd.DataFrame(data=fake_samples.detach().numpy(), columns=X.columns)


    print(fake.head())
    print(fake.describe())

    # fake.to_csv("preliminar_output.csv", index=False)

