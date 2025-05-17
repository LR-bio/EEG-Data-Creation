import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from torchsummary import summary
from sklearn.metrics import accuracy_score, f1_score
import time
import random

# Constants
EEG_SEQ_LEN = 256
LATENT_DIM = 100
NUM_CLASSES = 2
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.0002
BETA1 = 0.5

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(LATENT_DIM + NUM_CLASSES, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, EEG_SEQ_LEN),
            nn.Tanh()
        )

    def forward(self, z, labels):
        c = torch.nn.functional.one_hot(labels, NUM_CLASSES).float()
        input = torch.cat([z, c], dim=1)
        output = self.model(input)
        return output.unsqueeze(1)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(EEG_SEQ_LEN + NUM_CLASSES, 256)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        x = x.view(x.size(0), -1)
        c = torch.nn.functional.one_hot(labels, NUM_CLASSES).float()
        input = torch.cat([x, c], dim=1)
        return self.model(input)

def train_gan(data, labels):
    G = Generator().to(device)
    D = Discriminator().to(device)

    print("Generator Summary:")
    summary(G, input_size=[(LATENT_DIM,), (1,)])
    print("Discriminator Summary:")
    summary(D, input_size=[(1, EEG_SEQ_LEN), (1,)])

    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(G.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))
    optimizer_D = optim.Adam(D.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))

    loss_G_hist = []
    loss_D_hist = []
    f1_scores = []
    acc_scores = []

    for epoch in range(EPOCHS):
        start_time = time.time()
        predictions = []
        truths = []

        for i in range(0, len(data), BATCH_SIZE):
            batch_data = data[i:i+BATCH_SIZE]
            batch_labels = labels[i:i+BATCH_SIZE]
            batch_size = batch_data.shape[0]

            real = torch.tensor(batch_data, device=device)
            real_y = torch.tensor(batch_labels, dtype=torch.long, device=device)

            real_targets = torch.full((batch_size, 1), 0.9, device=device)
            fake_targets = torch.zeros((batch_size, 1), device=device)

            noise_std = 0.1 * (1 - epoch / EPOCHS)
            real += noise_std * torch.randn_like(real)

            D.zero_grad()
            output_real = D(real, real_y)
            loss_real = criterion(output_real, real_targets)

            z = torch.randn(batch_size, LATENT_DIM, device=device)
            fake_y = torch.randint(0, NUM_CLASSES, (batch_size,), device=device)
            fake = G(z, fake_y)
            fake += noise_std * torch.randn_like(fake)

            output_fake = D(fake.detach(), fake_y)
            loss_fake = criterion(output_fake, fake_targets)

            loss_D = loss_real + loss_fake
            loss_D.backward()
            optimizer_D.step()

            G.zero_grad()
            output = D(fake, fake_y)
            loss_G = criterion(output, real_targets)
            loss_G.backward()
            optimizer_G.step()

            predictions.extend((output > 0.5).cpu().numpy())
            truths.extend(real_targets.cpu().numpy())

        duration = time.time() - start_time
        loss_G_hist.append(loss_G.item())
        loss_D_hist.append(loss_D.item())

        acc = accuracy_score(truths, predictions)
        f1 = f1_score(truths, predictions)
        acc_scores.append(acc)
        f1_scores.append(f1)

        if (epoch+1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss D: {loss_D.item():.4f} | Loss G: {loss_G.item():.4f} | Acc: {acc:.4f} | F1: {f1:.4f} | Duration: {duration:.2f}s")

    plot_training_metrics(loss_G_hist, loss_D_hist, acc_scores, f1_scores)
    return G, D

def plot_training_metrics(loss_G, loss_D, acc, f1):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(loss_G, label='Generator Loss')
    plt.plot(loss_D, label='Discriminator Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(acc, label='Accuracy')
    plt.plot(f1, label='F1 Score')
    plt.title('Validation Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()

    plt.tight_layout()
    plt.show()

def generate_and_plot(generator, n_samples=5):
    generator.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, LATENT_DIM, device=device)
        labels = torch.randint(0, NUM_CLASSES, (n_samples,), device=device)
        samples = generator(z, labels).cpu().numpy()

    root = tk.Tk()
    root.wm_title("Generated EEG Samples")
    fig, axs = plt.subplots(n_samples, 1, figsize=(10, 2 * n_samples))
    for i in range(n_samples):
        axs[i].plot(samples[i][0])
        axs[i].set_title(f"Sample {i+1} | Label: {labels[i].item()}")
        axs[i].grid(True)
    fig.tight_layout()

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack()
    tk.mainloop()

def gui_app():
    def run():
        try:
            sample_count = int(entry.get())
            generate_and_plot(torch.load('generator.pth', map_location=device), sample_count)
        except Exception as e:
            print("Error:", e)

    root = tk.Tk()
    root.title("EEG Generator GUI")

    label = tk.Label(root, text="Number of EEG samples:")
    label.pack(pady=5)

    entry = tk.Entry(root)
    entry.insert(0, "5")
    entry.pack(pady=5)

    button = ttk.Button(root, text="Generate", command=run)
    button.pack(pady=10)

    root.mainloop()

# ----------- Main Entry Point --------------
def main():
    num_samples = 1000
    data = np.random.randn(num_samples, 1, EEG_SEQ_LEN).astype(np.float32)
    labels = np.random.randint(0, NUM_CLASSES, size=(num_samples,))

    G, D = train_gan(data, labels)

    torch.save(G.state_dict(), 'generator.pth')
    torch.save(D.state_dict(), 'discriminator.pth')
    print("Models saved successfully.")

    generate_and_plot(G, n_samples=5)
    gui_app()

if __name__ == '__main__':
    main()
