# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image
import sys
import time
import json
import csv
from datetime import datetime
from pytorch_msssim import ssim as ssim_metric # para medir calidad de reconstrucción

os.chdir(os.environ.get("TFG_RUN_DIR", os.getcwd()))  # Directorio de salidas configurado por el usuario

data_path = os.environ.get("TFG_DATA_DIR", "./datos/clock_shulman") #servidor

device = torch.device(os.environ.get("TFG_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu"))
gpu_name = torch.cuda.get_device_name(1) if device.type == "cuda" else None

# Guardado inmediato en training_log.txt
class Logger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log      = open(filepath, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger("training_log.txt")

# ============================================================
# IDENTIFICACIÓN DEL EXPERIMENTO (para comparar preprocesados)
# ============================================================
# Cambia esto manualmente en cada run para poder distinguir los
# experimentos en el CSV comparativo (p.ej. al probar resize vs
# resize+centercrop, o distintos tamaños de imagen).
preprocessing_mode = "resize"          # "resize" | "resize_centercrop"
experiment_name     = f"{preprocessing_mode}_img{{IMG_SIZE}}"  # se completa mas abajo una vez definido img_size

# Parámetros generales / Configuración
workers = 0 # numero de procesos independientes se utilizan para cargar y preparar las imágenes mientras
# el modelo se está entrenando
latent_dim = 8
batch_size = 128 #probar con 64
img_channels = 3
img_size = 384
lr = 1e-3
epochs = 200
beta = 3


experiment_name = f"{preprocessing_mode}_img{img_size}"

print(torch.cuda.is_available())
print(torch.version.cuda)
print(torch.cuda.device_count())
print(f"Experimento: {experiment_name}")

# Parámetros para la visualización
save_interval = 50
sample_interval = 10
mosaic_size = 4

# ------------------------------------------------------------
# Preprocesado: cambia aquí según preprocessing_mode para que
# el pipeline y el nombre del experimento vayan siempre acordes.
# ------------------------------------------------------------
if preprocessing_mode == "resize":
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: 1.0 - x)
    ])
elif preprocessing_mode == "resize_centercrop":
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: 1.0 - x)
    ])
else:
    raise ValueError(f"preprocessing_mode desconocido: {preprocessing_mode}")

full_dataset = ImageFolder(root=data_path, transform=transform)

if full_dataset.class_to_idx.get('5_perfect_clock') != 5:

    raise ValueError('Unexpected ImageFolder class order: provide the six original classes')

# OPCIÓN 1: solo clase 5
indices = [i for i, (_, label) in enumerate(full_dataset) if label == 5]
dataset = torch.utils.data.Subset(full_dataset, indices)

# OPCIÓN 2: todas las clases (0-5)
# dataset = full_dataset


# ============================================================
# Split del dataset en train / val / test
# Proporciones: 85% train | 10% val | 5% test
# ============================================================
torch.manual_seed(42)

total      = len(dataset)
train_size = int(0.85 * total)
val_size   = int(0.10 * total)
test_size  = total - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    dataset, [train_size, val_size, test_size]
)

# Guardado de particiones para reproducibilidad
os.makedirs("splits", exist_ok=True)
torch.save(train_dataset.indices, "splits/train_indices.pt")
torch.save(val_dataset.indices,   "splits/val_indices.pt")
torch.save(test_dataset.indices,  "splits/test_indices.pt")

print(f"Split guardado -> Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=workers)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=workers)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=workers)
# shuffle = True -> las imágenes se mezclan aleatoriamente al comienzo de cada época evitando que
# el modelo aprenda un orden fijo de las imágenes

print("Clases detectadas:", full_dataset.class_to_idx)

# Mostrar primeras 8 imágenes del train
data_iter = iter(train_loader) # el iterador apuntará al primer batch
images, labels = next(data_iter) # devuelve el primer batch de imágenes y etiquetas

# Genera una muestra con las 8 primeras imágenes del primer batch -> sample_input.png
fig, axes = plt.subplots(2, 4, figsize=(10, 5))
for i, ax in enumerate(axes.flat):
    img = images[i].squeeze()
    if img.ndim == 2:
        ax.imshow(img, cmap='gray')
    else:
        img = img.permute(1, 2, 0)
        ax.imshow(img)
    ax.set_title(f"Label: {labels[i].item()}")
    ax.axis('off')
plt.tight_layout()
plt.savefig('sample_input.png')
plt.close()


class Encoder(nn.Module):
    def __init__(self, img_size=64, img_channels=3, latent_dim=128, kernel_size=3): # por defecto si no se especifica lo contrario
        super().__init__()
        self.img_size = img_size
        self.latent_dim = latent_dim
        self.kernel_size = kernel_size

        padding = (kernel_size - 1) // 2 # de este modo el tamaño solo disminuya por culpa del stride y no por el tamaño del filtro
        self.conv = nn.Sequential(
            nn.Conv2d(img_channels, 32, kernel_size, stride=2, padding=padding),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size, stride=2, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size, stride=2, padding=padding),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size, stride=2, padding=padding),
            nn.ReLU()
        )
        # 3  fm x 256 x 256
        # 32 fm x 128 x 128
        # 64 fm x 64  x 64
        # 128fm x 32  x 32
        # 256fm x 16  x 16

        # Se obtiene el tamaño final de los feature maps después de las 4 capas convolucionales
        size = img_size
        for _ in range(4):
            size = (size + 2*padding - kernel_size)//2 + 1 # salida = |_(entrada + 2*padding - kernel_size)/stride_| + 1
        self.final_size = size

        self.fc_mu     = nn.Linear(256 * self.final_size * self.final_size, latent_dim) # capa densa MU
        self.fc_logvar = nn.Linear(256 * self.final_size * self.final_size, latent_dim) # capa densa LOGVAR

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1) # Aplana los feature maps en un único vector
        mu     = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, img_size=64, img_channels=3, latent_dim=128, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        padding = (kernel_size - 1) // 2

        size = img_size
        for _ in range(4):
            size = (size + 2*padding - kernel_size)//2 + 1
        self.final_size = size

        self.fc = nn.Linear(latent_dim, 256 * self.final_size * self.final_size) # capa densa del esp lat a la primera capa conv de trasposición

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size, stride=2, padding=padding, output_padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size, stride=2, padding=padding, output_padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size, stride=2, padding=padding, output_padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(32, img_channels, kernel_size, stride=2, padding=padding, output_padding=1),
            nn.Sigmoid() # Se garantiza que la salida esté en el rango [0, 1] para imágenes normalizadas
        )

        # 256fm x 16  x 16
        # 128fm x 32  x 32
        # 64 fm x 64  x 64
        # 32 fm x 128 x 128
        # 3  fm x 256 x 256

    def forward(self, z):
        x = self.fc(z).view(-1, 256, self.final_size, self.final_size)
        return self.deconv(x)


class VAE(nn.Module):
    def __init__(self, img_size=64, img_channels=3, latent_dim=128):
        super().__init__()
        self.encoder = Encoder(img_size, img_channels, latent_dim, 3) # se devuelve la media y la logvar
        self.decoder = Decoder(img_size, img_channels, latent_dim, 3)

    # Truco de reparametrización: z = mu + std * eps, donde eps ~ N(0, 1)
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

# GENERACIÓN DE RESULTADOS VISUALES
def generate_and_save_mosaic(decoder, epoch, n=mosaic_size):
    decoder.eval() #no cambia nada porque no hay Dropout ni BatchNorm en el decoder
    with torch.no_grad(): #no se almacenan gradientes para que el proceso sea más rápido y consuma menos memoria
        z = torch.randn(n*n, latent_dim).to(device)
        samples = decoder(z).cpu()
        os.makedirs("mosaics", exist_ok=True)
        save_image(samples, f'mosaics/mosaic_epoch_{epoch}.png', nrow=n, normalize=True)
        fig, axs = plt.subplots(n, n, figsize=(10, 10))
        for i in range(n):
            for j in range(n):
                idx = i * n + j
                img = samples[idx].permute(1, 2, 0).numpy()
                axs[i, j].imshow(img)
                axs[i, j].axis('off')
        plt.tight_layout()
        plt.savefig(f'mosaics/mosaic_detailed_epoch_{epoch}.png')
        plt.close()


# Función de pérdida de entrenamiento: MSE + beta * KL
# Se mantiene en 'sum' porque es la que realmente optimiza el modelo
# (mezclar 'mean' en la loss desequilibraría el término KL frente al de reconstrucción).
def loss_function(recon_x, x, mu, logvar, beta=1.0):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kl_div     = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_div, recon_loss, kl_div


# ------------------------------------------------------------
# Métricas de evaluación del preprocesado (NO se usan como loss)
# ------------------------------------------------------------
# mse_mean: promedio del error por píxel/canal. Al dividir entre el
# número total de elementos, es comparable entre configuraciones con
# distinto tamaño de imagen (256x256 vs 128x128, resize vs crop, etc.),
# a diferencia de recon_loss (sum), que crece con el nº de píxeles.
def compute_mse_mean(recon_x, x):
    return F.mse_loss(recon_x, x, reduction='mean').item()

# SSIM: métrica para evaluar la calidad visual de las reconstrucciones (NO como pérdida de entrenamiento)
def compute_ssim(recon_x, x):
    return ssim_metric(recon_x, x, data_range=1.0, size_average=True).item()
# recon_x          : imagenes reconstruidas por el decodificador (batch x canales x alto x ancho)
# x                : imágenes originales del dataset(batch x canales x alto x ancho)
# data_range=1.0   : el rango de valores de las imágenes es [0, 1] (ya que se normalizan al cargar y al decodificar con sigmoid)
# size_average=True: se hace la media de SSIM de todas las imágenes del batch, devolviendo un único valor por batch
# .item()          : convierte el tensor en un float


vae       = VAE(img_size=img_size, img_channels=img_channels, latent_dim=latent_dim).to(device)
optimizer = torch.optim.Adam(vae.parameters(), lr=lr) # Se utiliza el algoritmo de optimización Adam

# Listas para curvas de aprendizaje — train
losses          = []
recon_losses    = []
kl_losses       = []
train_ssim_list = []
train_mse_mean_list = []

# Listas para curvas de aprendizaje — validación
val_losses       = []
val_recon_losses = []
val_kl_losses    = []
val_ssim_list    = []
val_mse_mean_list = []

# ------------------------------------------------------------
# Listas para coste computacional
# ------------------------------------------------------------
epoch_times_sec   = []   # tiempo total (train+val) por época
train_times_sec   = []   # tiempo solo de la fase de train
val_times_sec     = []   # tiempo solo de la fase de val
# Historical JSON keys ending in _mb are retained for compatibility; their unit is MiB (2**20 bytes).
peak_mem_mb_list  = []   # memoria pico de GPU (MiB) por época, si hay CUDA

# Variables para early stopping
best_val_loss    = float('inf')
patience         = 15
patience_counter = 0

os.makedirs("samples", exist_ok=True)

training_start_time = time.time()

for epoch in range(1, epochs + 1):

    epoch_start = time.time()

    # Reinicia el contador de memoria pico para medir solo lo que consume esta época
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # ----------------------------------------------------------
    # BUCLE DE ENTRENAMIENTO
    # ----------------------------------------------------------
    vae.train()
    train_loss       = 0
    train_recon_loss = 0
    train_kl_loss    = 0
    train_ssim_accum = 0.0
    train_mse_mean_accum = 0.0

    train_phase_start = time.time()

    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()

        recon_batch, mu, logvar = vae(data)
        loss, recon_loss, kl_loss = loss_function(recon_batch, data, mu, logvar, beta)

        loss.backward()
        train_loss       += loss.item()
        train_recon_loss += recon_loss.item()
        train_kl_loss    += kl_loss.item()
        optimizer.step()

        with torch.no_grad():
            train_ssim_accum     += compute_ssim(recon_batch, data)
            train_mse_mean_accum += compute_mse_mean(recon_batch, data)

        if batch_idx % 100 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, '
                  f'Loss: {loss.item():.4f}, Recon: {recon_loss.item():.4f}, KL: {kl_loss.item():.4f}')

    # Sincroniza antes de parar el cronómetro: en GPU las operaciones son
    # asíncronas, así que sin sync el tiempo medido no sería fiable.
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    train_phase_time = time.time() - train_phase_start

    avg_loss       = train_loss       / len(train_loader)
    avg_recon_loss = train_recon_loss / len(train_loader)
    avg_kl_loss    = train_kl_loss    / len(train_loader)
    avg_train_ssim = train_ssim_accum / len(train_loader)
    avg_train_mse_mean = train_mse_mean_accum / len(train_loader)

    losses.append(avg_loss)
    recon_losses.append(avg_recon_loss)
    kl_losses.append(avg_kl_loss)
    train_ssim_list.append(avg_train_ssim)
    train_mse_mean_list.append(avg_train_mse_mean)


    # ----------------------------------------------------------
    # BUCLE DE VALIDACIÓN
    # ----------------------------------------------------------
    vae.eval()
    val_loss_total       = 0
    val_recon_loss_total = 0
    val_kl_loss_total    = 0
    val_ssim_accum       = 0.0
    val_mse_mean_accum   = 0.0

    val_phase_start = time.time()

    with torch.no_grad():
        for data, _ in val_loader:
            data = data.to(device)
            recon_batch, mu, logvar = vae(data)
            loss, recon_loss, kl_loss = loss_function(recon_batch, data, mu, logvar, beta)
            val_loss_total       += loss.item()
            val_recon_loss_total += recon_loss.item()
            val_kl_loss_total    += kl_loss.item()
            val_ssim_accum       += compute_ssim(recon_batch, data)
            val_mse_mean_accum   += compute_mse_mean(recon_batch, data)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    val_phase_time = time.time() - val_phase_start

    avg_val_loss       = val_loss_total       / len(val_loader)
    avg_val_recon_loss = val_recon_loss_total / len(val_loader)
    avg_val_kl_loss    = val_kl_loss_total    / len(val_loader)
    avg_val_ssim       = val_ssim_accum       / len(val_loader)
    avg_val_mse_mean   = val_mse_mean_accum   / len(val_loader)

    val_losses.append(avg_val_loss)
    val_recon_losses.append(avg_val_recon_loss)
    val_kl_losses.append(avg_val_kl_loss)
    val_ssim_list.append(avg_val_ssim)
    val_mse_mean_list.append(avg_val_mse_mean)

    # ----------------------------------------------------------
    # Coste computacional de la época
    # ----------------------------------------------------------
    epoch_time = time.time() - epoch_start
    epoch_times_sec.append(epoch_time)
    train_times_sec.append(train_phase_time)
    val_times_sec.append(val_phase_time)

    if device.type == "cuda":
        peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    else:
        peak_mem_mb = float('nan')  # no aplica en CPU
    peak_mem_mb_list.append(peak_mem_mb)

    print(f'Epoch {epoch} | '
          f'Train -> Loss: {avg_loss:.4f} | Recon: {avg_recon_loss:.4f} | KL: {avg_kl_loss:.4f} | '
          f'SSIM: {avg_train_ssim:.4f} | MSE_mean: {avg_train_mse_mean:.6f}')
    print(f'        | '
          f'Val   -> Loss: {avg_val_loss:.4f} | Recon: {avg_val_recon_loss:.4f} | KL: {avg_val_kl_loss:.4f} | '
          f'SSIM: {avg_val_ssim:.4f} | MSE_mean: {avg_val_mse_mean:.6f}')
    print(f'  -> Coste computacional: epoch_time={epoch_time:.2f}s '
          f'(train={train_phase_time:.2f}s, val={val_phase_time:.2f}s) | '
          f'peak_GPU_mem={peak_mem_mb:.1f} MiB')

    # Guardado del mejor modelo
    if avg_val_loss < best_val_loss:
        best_val_loss    = avg_val_loss
        patience_counter = 0
        os.makedirs("checkpoints", exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': vae.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_loss,
            'val_loss': avg_val_loss,
            'train_ssim': avg_train_ssim,
            'val_ssim': avg_val_ssim,
            'val_mse_mean': avg_val_mse_mean,
        }, "checkpoints/vae_best.pth")
        print(f"  -> Mejor modelo guardado (val_loss: {best_val_loss:.4f} | val_ssim: {avg_val_ssim:.4f} | "
              f"val_mse_mean: {avg_val_mse_mean:.6f})")
    else:
        patience_counter += 1
        print(f"  -> Sin mejora en validacion ({patience_counter}/{patience})")

    # Early stopping
    if patience_counter >= patience:
        print(f"\nEarly stopping en epoca {epoch}. Sin mejora durante {patience} epocas consecutivas.")
        break

    # Guardado periódico
    if epoch % save_interval == 0 or epoch == epochs:
        os.makedirs("checkpoints", exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': vae.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, f"checkpoints/vae_epoch_{epoch}.pth")
        print(f"Checkpoint guardado en checkpoints/vae_epoch_{epoch}.pth")

    # Muestra aleatoria del decoder
    with torch.no_grad():
        z = torch.randn(64, latent_dim).to(device)
        sample = vae.decoder(z)
        save_image(sample.cpu(), f'samples/sample_epoch_{epoch}.png', nrow=8, normalize=True)

    if epoch % sample_interval == 0 or epoch == epochs:
        generate_and_save_mosaic(vae.decoder, epoch)


total_training_time = time.time() - training_start_time

torch.save(vae.state_dict(), "vae_final.pth")


# ============================================================
# CURVAS DE APRENDIZAJE
# ============================================================
epochs_run = len(losses)
x_axis = range(1, epochs_run + 1)

plt.figure(figsize=(10, 5))
plt.plot(x_axis, losses,     'b-',  label='Train Total Loss')
plt.plot(x_axis, val_losses, 'b--', label='Val Total Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Total Loss: Train vs Val')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('curve_total_loss.png')
plt.close()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
ax1.plot(x_axis, recon_losses,     'g-',  label='Train Recon (MSE sum)')
ax1.plot(x_axis, val_recon_losses, 'g--', label='Val Recon (MSE sum)')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Reconstruction Loss (MSE sum)')
ax1.set_title('Reconstruction Loss: Train vs Val')
ax1.legend()
ax1.grid(True)
ax2.plot(x_axis, kl_losses,     'r-',  label='Train KL')
ax2.plot(x_axis, val_kl_losses, 'r--', label='Val KL')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('KL Divergence')
ax2.set_title('KL Divergence: Train vs Val')
ax2.legend()
ax2.grid(True)
plt.tight_layout()
plt.savefig('curve_recon_kl.png')
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(x_axis, train_ssim_list, 'm-',  label='Train SSIM')
plt.plot(x_axis, val_ssim_list,   'm--', label='Val SSIM')
plt.xlabel('Epoch')
plt.ylabel('SSIM')
plt.ylim(0, 1)
plt.title('SSIM: Train vs Val')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('curve_ssim.png')
plt.close()

# Curva MSE_mean (comparable entre preprocesados con distinto tamaño)
plt.figure(figsize=(10, 5))
plt.plot(x_axis, train_mse_mean_list, 'c-',  label='Train MSE mean')
plt.plot(x_axis, val_mse_mean_list,   'c--', label='Val MSE mean')
plt.xlabel('Epoch')
plt.ylabel('MSE mean (por pixel)')
plt.title(f'MSE mean: Train vs Val [{experiment_name}]')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('curve_mse_mean.png')
plt.close()

# Curva de coste computacional por época
fig, (axt, axm) = plt.subplots(1, 2, figsize=(15, 5))
axt.plot(x_axis, epoch_times_sec, 'k-',  label='Epoch total')
axt.plot(x_axis, train_times_sec, 'g--', label='Train phase')
axt.plot(x_axis, val_times_sec,   'r--', label='Val phase')
axt.set_xlabel('Epoch')
axt.set_ylabel('Tiempo (s)')
axt.set_title(f'Tiempo por época [{experiment_name}]')
axt.legend()
axt.grid(True)
axm.plot(x_axis, peak_mem_mb_list, 'b-', label='Peak GPU mem (MiB)')
axm.set_xlabel('Epoch')
axm.set_ylabel('MiB')
axm.set_title('Memoria pico de GPU por época')
axm.legend()
axm.grid(True)
plt.tight_layout()
plt.savefig('curve_computational_cost.png')
plt.close()

print("\nGraficas guardadas: curve_total_loss.png | curve_recon_kl.png | curve_ssim.png | "
      "curve_mse_mean.png | curve_computational_cost.png")


# Imágenes reconstruidas para inspección visual
vae.eval()
with torch.no_grad():
    imgs, _ = next(iter(train_loader))
    imgs    = imgs.to(device)[:10*10]
    mu, logvar = vae.encoder(imgs)
    z          = vae.reparameterize(mu, logvar)
    recons     = vae.decoder(z)
    comparison = torch.cat([imgs, recons])
    save_image(comparison.cpu(), 'sample_reconst.png', nrow=10, normalize=True)


# ============================================================
# EVALUACIÓN FINAL SOBRE TEST
# ============================================================
print("\n--- Evaluacion final sobre TEST ---")
best_checkpoint = torch.load("checkpoints/vae_best.pth", map_location=device)
vae.load_state_dict(best_checkpoint['model_state_dict'])
vae.eval()

test_loss_total       = 0
test_recon_loss_total = 0
test_kl_loss_total    = 0
test_ssim_accum       = 0.0
test_mse_mean_accum   = 0.0

test_eval_start = time.time()

with torch.no_grad():
    for data, _ in test_loader:
        data = data.to(device)
        recon_batch, mu, logvar = vae(data)
        loss, recon_loss, kl_loss = loss_function(recon_batch, data, mu, logvar, beta)
        test_loss_total       += loss.item()
        test_recon_loss_total += recon_loss.item()
        test_kl_loss_total    += kl_loss.item()
        test_ssim_accum       += compute_ssim(recon_batch, data)
        test_mse_mean_accum   += compute_mse_mean(recon_batch, data)

if device.type == "cuda":
    torch.cuda.synchronize(device)
test_eval_time = time.time() - test_eval_start

avg_test_loss       = test_loss_total       / len(test_loader)
avg_test_recon_loss = test_recon_loss_total / len(test_loader)
avg_test_kl_loss    = test_kl_loss_total    / len(test_loader)
avg_test_ssim       = test_ssim_accum       / len(test_loader)
avg_test_mse_mean   = test_mse_mean_accum   / len(test_loader)

print(f"Test Loss: {avg_test_loss:.4f} | Test Recon: {avg_test_recon_loss:.4f} | "
      f"Test KL: {avg_test_kl_loss:.4f} | Test SSIM: {avg_test_ssim:.4f} | "
      f"Test MSE_mean: {avg_test_mse_mean:.6f}")
print(f"(Modelo epoca {best_checkpoint['epoch']} | "
      f"val_loss={best_checkpoint['val_loss']:.4f} | val_ssim={best_checkpoint['val_ssim']:.4f})")
print(f"Tiempo de inferencia sobre test: {test_eval_time:.2f}s "
      f"({test_eval_time/len(test_dataset)*1000:.2f} ms/imagen)")


# Se guardan los vectores de medias del espacio latente (mu) de cada imagen junto con
# su etiqueta real para posteriormente runnear UMAP_tSNE.py
print("\n--- Extrayendo vectores latentes de todo el dataset ---")

# DataLoader sobre el dataset completo usado en este experimento
# (respeta la opción elegida: solo clase 5 o varias clases)
full_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=False,      # Sin shuffle para mantener correspondencia imagen-etiqueta
    num_workers=workers
)

all_mu     = []
all_labels = []

vae.eval()
with torch.no_grad():
    for imgs, lbls in full_loader:
        imgs = imgs.to(device)
        mu, logvar = vae.encoder(imgs)
        # Usamos mu (la media) en lugar de z muestreado porque es determinista:
        # la misma imagen siempre produce el mismo vector, lo que hace las
        # visualizaciones UMAP/t-SNE reproducibles sin fijar semilla de muestreo.
        all_mu.append(mu.cpu().numpy())
        all_labels.append(lbls.numpy())

all_mu     = np.concatenate(all_mu,     axis=0)  # (N, latent_dim)
all_labels = np.concatenate(all_labels, axis=0)  # (N,)

os.makedirs("latent", exist_ok=True)
np.save("latent/latent_vectors.npy", all_mu)
np.save("latent/latent_labels.npy",  all_labels)

print(f"Vectores latentes guardados en latent/")
print(f"  latent_vectors.npy -> shape: {all_mu.shape}")
print(f"  latent_labels.npy  -> shape: {all_labels.shape}")
print(f"  Etiquetas presentes: {np.unique(all_labels)}")


# ============================================================
# RESUMEN DEL EXPERIMENTO (JSON) + FILA EN CSV COMPARATIVO
# ============================================================
avg_epoch_time = float(np.mean(epoch_times_sec))
avg_peak_mem   = float(np.nanmean(peak_mem_mb_list)) if device.type == "cuda" else float('nan')

# Tamaño real de los feature maps tras las 4 conv (útil para depurar
# arquitecturas cuando cambias img_size)
encoder_final_size = vae.encoder.final_size

# Nº de parámetros del modelo (para comparar coste entre configuraciones)
n_params_total     = sum(p.numel() for p in vae.parameters())
n_params_trainable = sum(p.numel() for p in vae.parameters() if p.requires_grad)

summary = {

    # ------------------------------------------------------
    # IDENTIFICACIÓN Y REPRODUCIBILIDAD
    # ------------------------------------------------------
    "experiment_name": experiment_name,
    "timestamp": datetime.now().isoformat(timespec="seconds"),
    "random_seed": 42,  # el valor pasado a torch.manual_seed()
    "python_version": sys.version.split()[0],
    "torch_version": torch.__version__,
    "cuda_version": torch.version.cuda,
    "gpu_name": gpu_name,

    # ------------------------------------------------------
    # PREPROCESAMIENTO (lo que estás comparando ahora mismo)
    # ------------------------------------------------------
    "preprocessing_mode": preprocessing_mode,      # "resize" | "resize_centercrop"
    "img_size": img_size,
    "img_channels": img_channels,
    "color_inversion": True,                       # 1.0 - x, fijo en tus experimentos
    "encoder_final_feature_map_size": encoder_final_size,

    # ------------------------------------------------------
    # DATASET Y SPLIT
    # ------------------------------------------------------
    "dataset_path": data_path,
    "class_selection": "clase 5 unicamente",        # o "todas las clases" si usas OPCIÓN 2
    "n_total_samples": total,
    "n_train": len(train_dataset),
    "n_val": len(val_dataset),
    "n_test": len(test_dataset),
    "split_ratios": {"train": 0.85, "val": 0.10, "test": 0.05},
    "split_indices_files": {
        "train": "splits/train_indices.pt",
        "val": "splits/val_indices.pt",
        "test": "splits/test_indices.pt",
    },

    # ------------------------------------------------------
    # ARQUITECTURA E HIPERPARÁMETROS
    # ------------------------------------------------------
    "latent_dim": latent_dim,
    "batch_size": batch_size,
    "lr": lr,
    "optimizer": "Adam",
    "beta": beta,
    "kernel_size": 3,
    "n_conv_layers": 4,
    "n_params_total": n_params_total,
    "n_params_trainable": n_params_trainable,

    # ------------------------------------------------------
    # ENTRENAMIENTO / EARLY STOPPING
    # ------------------------------------------------------
    "epochs_max": epochs,
    "epochs_run": epochs_run,
    "early_stopping_patience": patience,
    "early_stopping_triggered": epochs_run < epochs,
    "best_epoch": best_checkpoint['epoch'],

    # ------------------------------------------------------
    # MÉTRICAS EN EL MEJOR MODELO (val, en el punto de guardado)
    # ------------------------------------------------------
    "best_val_loss": best_val_loss,
    "best_val_recon_loss_sum": best_checkpoint.get('val_recon_loss', None),
    "best_val_kl": best_checkpoint.get('val_kl', None),
    "best_val_ssim": best_checkpoint['val_ssim'],
    "best_val_mse_mean": best_checkpoint.get('val_mse_mean', None),

    # ------------------------------------------------------
    # MÉTRICAS FINALES DE TEST (con el mejor modelo cargado)
    # ------------------------------------------------------
    "test_loss": avg_test_loss,
    "test_recon_loss_sum": avg_test_recon_loss,
    "test_kl": avg_test_kl_loss,
    "test_ssim": avg_test_ssim,
    "test_mse_mean": avg_test_mse_mean,

    # ------------------------------------------------------
    # CURVAS: valores clave (no solo el mejor, para ver overfitting)
    # ------------------------------------------------------
    "final_train_loss": losses[-1],
    "final_val_loss": val_losses[-1],
    "final_train_ssim": train_ssim_list[-1],
    "final_val_ssim": val_ssim_list[-1],
    "train_val_loss_gap_final": losses[-1] - val_losses[-1],  # indicador de overfitting
    "min_val_loss_epoch_vs_last_epoch": {
        "best_epoch": best_checkpoint['epoch'],
        "last_epoch": epochs_run,
    },

    # ------------------------------------------------------
    # COSTE COMPUTACIONAL
    # ------------------------------------------------------
    "device": str(device),
    "total_training_time_sec": total_training_time,
    "avg_epoch_time_sec": avg_epoch_time,
    "min_epoch_time_sec": float(np.min(epoch_times_sec)),
    "max_epoch_time_sec": float(np.max(epoch_times_sec)),
    "avg_train_phase_time_sec": float(np.mean(train_times_sec)),
    "avg_val_phase_time_sec": float(np.mean(val_times_sec)),
    "test_inference_time_sec": test_eval_time,
    "test_inference_ms_per_image": test_eval_time / len(test_dataset) * 1000,
    "avg_peak_gpu_mem_mb": avg_peak_mem,
    "max_peak_gpu_mem_mb": float(np.nanmax(peak_mem_mb_list)) if device.type == "cuda" else float('nan'),
    "throughput_images_per_sec_train": (
        len(train_dataset) / float(np.mean(train_times_sec))
        if np.mean(train_times_sec) > 0 else None
    ),

    # ------------------------------------------------------
    # RUTAS A ARTEFACTOS GENERADOS (para trazabilidad en la memoria)
    # ------------------------------------------------------
    "artifacts": {
        "best_checkpoint": "checkpoints/vae_best.pth",
        "final_model": "vae_final.pth",
        "training_log": "training_log.txt",
        "latent_vectors": "latent/latent_vectors.npy",
        "latent_labels": "latent/latent_labels.npy",
        "curve_total_loss": "curve_total_loss.png",
        "curve_recon_kl": "curve_recon_kl.png",
        "curve_ssim": "curve_ssim.png",
        "sample_reconst": "sample_reconst.png",
    },
}

with open(f"experiment_summary_{experiment_name}.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(f"\nResumen guardado en experiment_summary_{experiment_name}.json")

# CSV comparativo acumulado. Se guarda un nivel por encima de la carpeta
# del experimento (ajusta esta ruta si tu estructura de carpetas es otra),
# así todos los runs con distinto preprocesado/tamaño caen en el mismo archivo.
comparison_csv_path = os.path.join("..", "experiments_comparison.csv")
file_exists = os.path.isfile(comparison_csv_path)
try:
    with open(comparison_csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(summary)
    print(f"Fila añadida a {comparison_csv_path}")
except Exception as e:
    print(f"No se pudo escribir en el CSV comparativo ({comparison_csv_path}): {e}")
# ============================================================


# Cerrar el archivo de log limpiamente
logger = sys.stdout
sys.stdout = logger.terminal
logger.log.close()


# Ejecutar automáticamente el script de visualización
import subprocess

print("\nEjecutando UMAP_tSNE.py...")
subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "UMAP_tSNE.py")], check=True)


# Como cargar el modelo guardado:
# checkpoint = torch.load("checkpoints/vae_best.pth", map_location=device)
# vae.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# print(f"Cargado desde epoca {checkpoint['epoch']}, val_loss={checkpoint['val_loss']:.4f}")
