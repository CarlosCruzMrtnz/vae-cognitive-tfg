# -*- coding: utf-8 -*-
"""
fid_cmmd_eval.py
=================
Calcula FID y CMMD (real vs reconstruido, real vs generado) sobre el
checkpoint ya entrenado (checkpoints/vae_best.pth) y el split de test
guardado por vae_training.py (splits/test_indices.pt).

Se ejecuta de forma independiente al entrenamiento (igual que
UMAP_tSNE.py), para que:
  - un fallo o dependencia extra (transformers/CLIP) no tire la ejecución
    completa del entrenamiento.
  - se pueda recalcular sobre un checkpoint ya entrenado sin reentrenar.
  - la VRAM del entrenamiento quede totalmente liberada antes de cargar
    Inception y CLIP (que consumen bastante memoria).

Uso:
    python fid_cmmd_eval.py --experiment_name resize_img384
    (o simplemente `python fid_cmmd_eval.py`, que intenta detectar el
    experiment_name a partir del único experiment_summary_*.json presente
    en el directorio actual)

Requisitos adicionales: pip install transformers scipy --break-system-packages
"""

import argparse
import csv
import glob
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import linalg
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder


# ============================================================
# CONFIGURACIÓN
# ============================================================
# NOTA: estos valores deben coincidir con los usados en vae_training.py
# para reconstruir exactamente el mismo dataset/test-split. Si cambias
# algo aquí sin cambiarlo también en vae_training.py, real_imgs dejará de
# corresponder al split de test real.
preprocessing_mode = "resize"          # "resize" | "resize_centercrop"
latent_dim   = 128
batch_size   = 128
img_channels = 3
img_size     = 384
workers      = 0

# NOTA: os.chdir NO se hace aquí porque este script se lanza mediante
# subprocess.run(..., check=True) desde dentro de vae_training.py, que
# hereda el cwd ya posicionado en la carpeta del experimento
# (./workspace/latent_dim/latent_dim128_cmmd_fid). Si vas a
# ejecutar este script de forma manual/suelta, descomenta la línea de abajo.
# os.chdir('./workspace/latent_dim/latent_dim128_cmmd_fid')

data_path = os.environ.get("TFG_DATA_DIR", "./datos/clock_shulman") #servidor

device = torch.device(os.environ.get("TFG_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu"))

fid_extraction_batch_size = 16  # Inception/CLIP a img_size=384 consumen bastante VRAM; bajar si hay OOM


# ============================================================
# ARQUITECTURA (idéntica a la de vae_training.py; se necesita para poder
# cargar el state_dict del checkpoint)
# ============================================================
class Encoder(nn.Module):
    def __init__(self, img_size=64, img_channels=3, latent_dim=128, kernel_size=3):
        super().__init__()
        self.img_size = img_size
        self.latent_dim = latent_dim
        self.kernel_size = kernel_size

        padding = (kernel_size - 1) // 2
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

        size = img_size
        for _ in range(4):
            size = (size + 2 * padding - kernel_size) // 2 + 1
        self.final_size = size

        self.fc_mu     = nn.Linear(256 * self.final_size * self.final_size, latent_dim)
        self.fc_logvar = nn.Linear(256 * self.final_size * self.final_size, latent_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
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
            size = (size + 2 * padding - kernel_size) // 2 + 1
        self.final_size = size

        self.fc = nn.Linear(latent_dim, 256 * self.final_size * self.final_size)

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size, stride=2, padding=padding, output_padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size, stride=2, padding=padding, output_padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size, stride=2, padding=padding, output_padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(32, img_channels, kernel_size, stride=2, padding=padding, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, z):
        x = self.fc(z).view(-1, 256, self.final_size, self.final_size)
        return self.deconv(x)


class VAE(nn.Module):
    def __init__(self, img_size=64, img_channels=3, latent_dim=128):
        super().__init__()
        self.encoder = Encoder(img_size, img_channels, latent_dim, 3)
        self.decoder = Decoder(img_size, img_channels, latent_dim, 3)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


# ============================================================
# MÉTRICAS FID (Fréchet Inception Distance) y CMMD (CLIP-MMD)
# ============================================================
# Basado en: Jayasumana et al., "Rethinking FID: Towards a Better Evaluation
# Metric for Image Generation" (CVPR 2024, Google Research).
#
# NOTA METODOLÓGICA (léela antes de interpretar los resultados en la memoria):
# ------------------------------------------------------------------------------
# - CMMD fue propuesto para evaluar modelos texto-imagen. No está claro que
#   sea la métrica más adecuada para modelos generativos NO supervisados como
#   un VAE (no hay condicionamiento por texto que CLIP pueda aprovechar).
#   Además, CLIP se entrenó con fotografías naturales + texto, no con dibujos
#   lineales tipo "reloj", así que sus embeddings pueden no ser muy
#   informativos sobre trazo/geometría. Trátalo como métrica complementaria a
#   SSIM/MSE, no como sustituto.
# - FID requiere idealmente >10-20k imágenes para ser estable (asunción de
#   Gaussianidad + estimación de covarianza en alta dimensión). Con clase 5
#   únicamente (muy por debajo de eso), interpreta el valor absoluto con
#   cautela y úsalo sobre todo de forma COMPARATIVA entre tus propios
#   experimentos (mismo nº de imágenes en ambos lados), no para compararte
#   con cifras de otros papers.
# - El estimador de CMMD implementado aquí es el de VARIANZA MÍNIMA / SESGADO
#   (el que usa realmente el código de referencia de Google Research), no el
#   estimador insesgado clásico de Gretton et al. que excluye la diagonal.
#   Es importante documentarlo así en la memoria si citas la fórmula.

class InceptionFeatureExtractor(nn.Module):
    """
    Envuelve Inception-v3 (preentrenada en ImageNet) para devolver el vector
    de 2048 dimensiones de la penúltima capa (antes de la FC de clasificación).
    Espera imágenes en [0, 1], shape (B, 3, H, W): se redimensionan
    internamente a 299x299 y se normalizan con las estadísticas de ImageNet.
    """

    def __init__(self):
        super().__init__()
        from torchvision.models import inception_v3, Inception_V3_Weights

        weights = Inception_V3_Weights.IMAGENET1K_V1
        net = inception_v3(weights=weights, aux_logits=True)
        net.fc = nn.Identity()
        net.eval()
        self.net = net

        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    @torch.no_grad()
    def forward(self, x):
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        feats = self.net(x)
        return feats  # (B, 2048)


class ClipFeatureExtractor(nn.Module):
    """
    Envuelve CLIP ViT-L/14@336px (checkpoint "openai/clip-vit-large-patch14-336"
    vía HuggingFace transformers) y devuelve el embedding de imagen normalizado
    (dim=768). Espera imágenes en [0, 1], shape (B, 3, H, W).
    """

    def __init__(self):
        super().__init__()
        from transformers import CLIPModel

        self.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336")
        self.model.eval()

        self.register_buffer(
            "mean",
            torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1),
        )

    @torch.no_grad()
    def forward(self, x):
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        x = F.interpolate(x, size=(336, 336), mode="bicubic", align_corners=False)
        x = (x - self.mean) / self.std
        feats = self.model.get_image_features(pixel_values=x)

        # FIX: dependiendo de la version de `transformers` instalada,
        # get_image_features() puede devolver directamente un tensor (B, dim)
        # o un objeto ModelOutput (p.ej. BaseModelOutputWithPooling) que
        # envuelve varios tensores. Se maneja de forma defensiva para no
        # depender de la version exacta de la libreria.
        if not torch.is_tensor(feats):
            if hasattr(feats, "image_embeds") and feats.image_embeds is not None:
                feats = feats.image_embeds
            elif hasattr(feats, "pooler_output") and feats.pooler_output is not None:
                feats = feats.pooler_output
            else:
                raise TypeError(
                    "get_image_features() devolvio un tipo inesperado "
                    f"({type(feats)}) del que no se pudo extraer un tensor. "
                    "Revisa la version de 'transformers' instalada "
                    "(pip show transformers) y ajusta la extraccion aqui."
                )

        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats


class TensorImageDataset(Dataset):
    """Permite pasar un tensor (N, C, H, W) directamente a extract_features."""

    def __init__(self, images_tensor):
        self.images = images_tensor

    def __len__(self):
        return self.images.size(0)

    def __getitem__(self, idx):
        return self.images[idx]


@torch.no_grad()
def extract_features(images_iterable, extractor, device, batch_size=32, max_images=None):
    extractor.eval()
    feats_list = []
    n_seen = 0

    def process_batch(batch):
        batch = batch.to(device)
        f = extractor(batch)
        return f.cpu().numpy()

    if isinstance(images_iterable, torch.Tensor):
        loader = DataLoader(
            TensorImageDataset(images_iterable), batch_size=batch_size, shuffle=False
        )
    else:
        loader = images_iterable

    for batch in loader:
        if isinstance(batch, (list, tuple)):
            imgs = batch[0]
        else:
            imgs = batch

        if max_images is not None and n_seen >= max_images:
            break
        if max_images is not None and n_seen + imgs.size(0) > max_images:
            imgs = imgs[: max_images - n_seen]

        feats_list.append(process_batch(imgs))
        n_seen += imgs.size(0)

    return np.concatenate(feats_list, axis=0)


def compute_fid_from_features(feats_real, feats_gen, eps=1e-6):
    mu_r, mu_g = feats_real.mean(axis=0), feats_gen.mean(axis=0)
    sigma_r = np.cov(feats_real, rowvar=False)
    sigma_g = np.cov(feats_gen, rowvar=False)

    diff = mu_r - mu_g

    covmean = linalg.sqrtm(sigma_r.dot(sigma_g))
    if isinstance(covmean, tuple):
        covmean = covmean[0]

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            max_imag = np.max(np.abs(covmean.imag))
            print(f"[FID] Aviso: parte imaginaria no despreciable ({max_imag:.4f}), se descarta.")
        covmean = covmean.real

    if not np.isfinite(covmean).all():
        offset = np.eye(sigma_r.shape[0]) * eps
        covmean = linalg.sqrtm((sigma_r + offset).dot(sigma_g + offset))
        if isinstance(covmean, tuple):
            covmean = covmean[0]
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma_r) + np.trace(sigma_g) - 2 * np.trace(covmean)
    return float(fid)


def compute_fid(real_images_or_loader, gen_images_or_loader, device,
                 batch_size=32, max_images=None):
    extractor = InceptionFeatureExtractor().to(device)
    feats_real = extract_features(real_images_or_loader, extractor, device,
                                   batch_size=batch_size, max_images=max_images)
    feats_gen = extract_features(gen_images_or_loader, extractor, device,
                                  batch_size=batch_size, max_images=max_images)
    del extractor
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return compute_fid_from_features(feats_real, feats_gen)


_CMMD_SIGMA = 10.0
_CMMD_SCALE = 1000.0


def compute_cmmd_from_features(feats_x, feats_y, sigma=_CMMD_SIGMA, scale=_CMMD_SCALE):
    x = torch.from_numpy(feats_x).double()
    y = torch.from_numpy(feats_y).double()

    x_sqnorms = torch.diag(torch.matmul(x, x.T))
    y_sqnorms = torch.diag(torch.matmul(y, y.T))

    gamma = 1.0 / (2 * sigma ** 2)

    k_xx = torch.mean(torch.exp(
        -gamma * (-2 * torch.matmul(x, x.T)
                  + x_sqnorms.unsqueeze(1) + x_sqnorms.unsqueeze(0))
    ))
    k_xy = torch.mean(torch.exp(
        -gamma * (-2 * torch.matmul(x, y.T)
                  + x_sqnorms.unsqueeze(1) + y_sqnorms.unsqueeze(0))
    ))
    k_yy = torch.mean(torch.exp(
        -gamma * (-2 * torch.matmul(y, y.T)
                  + y_sqnorms.unsqueeze(1) + y_sqnorms.unsqueeze(0))
    ))

    cmmd = scale * (k_xx + k_yy - 2 * k_xy)
    return float(cmmd.item())


def compute_cmmd(real_images_or_loader, gen_images_or_loader, device,
                  batch_size=32, max_images=None, sigma=_CMMD_SIGMA, scale=_CMMD_SCALE):
    extractor = ClipFeatureExtractor().to(device)
    feats_real = extract_features(real_images_or_loader, extractor, device,
                                   batch_size=batch_size, max_images=max_images)
    feats_gen = extract_features(gen_images_or_loader, extractor, device,
                                  batch_size=batch_size, max_images=max_images)
    del extractor
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return compute_cmmd_from_features(feats_real, feats_gen, sigma=sigma, scale=scale)


# ============================================================
# UTILIDADES PARA ACTUALIZAR JSON / CSV DEL EXPERIMENTO
# ============================================================
def detect_experiment_name():
    """Si no se pasa --experiment_name, intenta detectarlo a partir del
    único experiment_summary_*.json presente en el directorio actual."""
    candidates = glob.glob("experiment_summary_*.json")
    if len(candidates) == 1:
        name = candidates[0][len("experiment_summary_"):-len(".json")]
        return name
    raise RuntimeError(
        "No se pudo detectar experiment_name automaticamente "
        f"(se encontraron {len(candidates)} archivos experiment_summary_*.json). "
        "Pasa --experiment_name explicitamente."
    )


def update_json_summary(experiment_name, fid_metrics):
    json_path = f"experiment_summary_{experiment_name}.json"
    if not os.path.isfile(json_path):
        print(f"[Aviso] No se encontro {json_path}; no se actualiza el JSON.")
        return
    with open(json_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    summary.update(fid_metrics)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"JSON actualizado: {json_path}")


def update_csv_row(experiment_name, fid_metrics):
    """Actualiza (en vez de duplicar) la fila del CSV comparativo que
    corresponde a este experiment_name con los valores de FID/CMMD."""
    comparison_csv_path = os.path.join("..", "experiments_comparison.csv")
    if not os.path.isfile(comparison_csv_path):
        print(f"[Aviso] No se encontro {comparison_csv_path}; no se actualiza el CSV.")
        return

    with open(comparison_csv_path, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        fieldnames = list(rows[0].keys()) if rows else []

    # Amplia la cabecera si faltan columnas nuevas
    for key in fid_metrics.keys():
        if key not in fieldnames:
            fieldnames.append(key)

    # Se actualiza la ULTIMA fila que coincide con experiment_name (por si
    # se ha relanzado el mismo experimento varias veces)
    updated = False
    for row in reversed(rows):
        if row.get("experiment_name") == experiment_name:
            row.update({k: v for k, v in fid_metrics.items()})
            updated = True
            break

    if not updated:
        print(f"[Aviso] No se encontro ninguna fila con experiment_name="
              f"'{experiment_name}' en {comparison_csv_path}; no se actualiza el CSV.")
        return

    with open(comparison_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"CSV actualizado: {comparison_csv_path} (fila '{experiment_name}')")


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default=None,
                         help="Nombre del experimento (p.ej. resize_img384). "
                              "Si se omite, se intenta detectar automaticamente.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/vae_best.pth",
                         help="Ruta al checkpoint del VAE a evaluar.")
    args = parser.parse_args()

    experiment_name = args.experiment_name or detect_experiment_name()
    print(f"Evaluando FID/CMMD para experimento: {experiment_name}")
    print(f"Checkpoint: {args.checkpoint}")

    # --------------------------------------------------------
    # Reconstruccion del dataset y del split de TEST exactamente
    # igual que en vae_training.py
    # --------------------------------------------------------
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
    indices = [i for i, (_, label) in enumerate(full_dataset) if label == 5]
    dataset = torch.utils.data.Subset(full_dataset, indices)

    test_indices = torch.load("splits/test_indices.pt")
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)

    print(f"Test set reconstruido a partir de splits/test_indices.pt -> {len(test_dataset)} imagenes")

    # --------------------------------------------------------
    # Carga del modelo entrenado
    # --------------------------------------------------------
    vae = VAE(img_size=img_size, img_channels=img_channels, latent_dim=latent_dim).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    vae.load_state_dict(checkpoint["model_state_dict"])
    vae.eval()
    print(f"Modelo cargado (epoca {checkpoint.get('epoch', '?')})")

    # --------------------------------------------------------
    # Reconstrucciones (deterministas, decoder(mu)) y muestras generadas
    # desde el prior N(0, I)
    # --------------------------------------------------------
    real_imgs_list  = []
    recon_imgs_list = []
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            mu, logvar = vae.encoder(data)
            recon = vae.decoder(mu)
            real_imgs_list.append(data.cpu())
            recon_imgs_list.append(recon.cpu())

    real_imgs  = torch.cat(real_imgs_list,  dim=0)
    recon_imgs = torch.cat(recon_imgs_list, dim=0)

    with torch.no_grad():
        z_prior  = torch.randn(real_imgs.size(0), latent_dim).to(device)
        gen_imgs = vae.decoder(z_prior).cpu()

    # --------------------------------------------------------
    # FID / CMMD
    # --------------------------------------------------------
    print("\n--- Calculando FID y CMMD sobre TEST ---")
    fid_cmmd_start = time.time()

    fid_reconstruction  = compute_fid(real_imgs, recon_imgs, device, batch_size=fid_extraction_batch_size)
    cmmd_reconstruction = compute_cmmd(real_imgs, recon_imgs, device, batch_size=fid_extraction_batch_size)
    fid_generation       = compute_fid(real_imgs, gen_imgs, device, batch_size=fid_extraction_batch_size)
    cmmd_generation      = compute_cmmd(real_imgs, gen_imgs, device, batch_size=fid_extraction_batch_size)

    if device.type == "cuda":
        torch.cuda.empty_cache()

    fid_cmmd_time = time.time() - fid_cmmd_start

    print(f"FID  (real vs reconstruido) : {fid_reconstruction:.4f}")
    print(f"CMMD (real vs reconstruido) : {cmmd_reconstruction:.4f}")
    print(f"FID  (real vs generado)     : {fid_generation:.4f}")
    print(f"CMMD (real vs generado)     : {cmmd_generation:.4f}")
    print(f"Tiempo FID+CMMD: {fid_cmmd_time:.2f}s sobre {real_imgs.size(0)} imagenes")

    # --------------------------------------------------------
    # Actualizacion de JSON + CSV del experimento
    # --------------------------------------------------------
    fid_metrics = {
        "fid_reconstruction": fid_reconstruction,
        "cmmd_reconstruction": cmmd_reconstruction,
        "fid_generation": fid_generation,
        "cmmd_generation": cmmd_generation,
        "cmmd_sigma": _CMMD_SIGMA,
        "cmmd_scale": _CMMD_SCALE,
        "n_images_fid_cmmd": real_imgs.size(0),
        "fid_cmmd_time_sec": fid_cmmd_time,
    }

    update_json_summary(experiment_name, fid_metrics)
    update_csv_row(experiment_name, fid_metrics)


if __name__ == "__main__":
    main()
