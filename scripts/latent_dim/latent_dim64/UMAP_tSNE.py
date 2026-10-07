# -*- coding: utf-8 -*-
# vae_latent_viz.py
# ============================================================
# Script independiente para visualización del espacio latente
# mediante UMAP y t-SNE, con Silhouette score como métrica
# cuantitativa de separabilidad.
#
# Prerequisitos:
#   pip install umap-learn scikit-learn matplotlib numpy
#
# Uso:
#   1. Ejecuta primero vae_training.py para generar los archivos:
#        latent/latent_vectors.npy
#        latent/latent_labels.npy
#   2. Ejecuta este script desde el mismo directorio de trabajo.
#
# Funciona con cualquier combinación de puntuaciones (solo clase 5,
# o varias clases simultáneamente) sin modificar nada.
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import sys

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples
import umap

# ============================================================
# CONFIGURACIÓN — ajusta estas rutas si es necesario
# ============================================================
LATENT_DIR = os.environ.get("TFG_LATENT_DIR", "./latent")
OUTPUT_DIR = os.environ.get("TFG_VIZ_DIR", "./latent_viz")
VECTORS_PATH = os.path.join(LATENT_DIR, "latent_vectors.npy")
LABELS_PATH  = os.path.join(LATENT_DIR, "latent_labels.npy")

# Parámetros UMAP a explorar
UMAP_N_NEIGHBORS = [5, 20, 80, 320]
UMAP_MIN_DIST    = [0.01, 0.1, 0.5]

# Parámetros t-SNE a explorar
TSNE_PERPLEXITIES = [5, 15, 30, 50]
TSNE_N_ITER       = 1000

# Semilla para reproducibilidad
RANDOM_STATE = 42
# ============================================================


# Logger
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

os.makedirs(OUTPUT_DIR, exist_ok=True)
sys.stdout = Logger(os.path.join(OUTPUT_DIR, "latent_viz_log.txt"))


# ============================================================
# CARGA DE DATOS
# ============================================================
print("Cargando vectores latentes...")

if not os.path.exists(VECTORS_PATH) or not os.path.exists(LABELS_PATH):
    raise FileNotFoundError(
        f"No se encontraron los archivos en '{LATENT_DIR}/'. "
        "Ejecuta primero vae_training.py para generarlos."
    )

vectors = np.load(VECTORS_PATH)   # (N, latent_dim)
labels  = np.load(LABELS_PATH)    # (N,)

print(f"  Vectores: {vectors.shape}  |  Etiquetas: {labels.shape}")
print(f"  Puntuaciones presentes: {np.unique(labels)}")
unique_labels = np.unique(labels)
n_classes     = len(unique_labels)

cmap   = plt.get_cmap('tab10', max(n_classes, 2))
colors = {lbl: cmap(i) for i, lbl in enumerate(unique_labels)}


def scatter_plot(ax, embedding, labels, title, silhouette=None):
    """Dibuja un scatter plot 2D coloreado por etiqueta clínica."""
    for lbl in np.unique(labels):
        mask = labels == lbl
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=[colors[lbl]],
            label=f'Score {int(lbl)}',
            alpha=0.6,
            s=8,
            linewidths=0
        )
    # AÑADIDO: si se pasa el silhouette score, lo mostramos en el título
    if silhouette is not None:
        title += f'\nSilhouette: {silhouette:.4f}'
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('Dim 1')
    ax.set_ylabel('Dim 2')
    ax.legend(fontsize=7, markerscale=2, loc='best')
    ax.grid(True, alpha=0.3)


# ============================================================
# AÑADIDO: SILHOUETTE SCORE SOBRE EL ESPACIO LATENTE ORIGINAL
#
# El Silhouette score mide qué tan bien separados están los grupos
# en el espacio latente de alta dimensión (NO en la proyección 2D).
# Es la versión cuantitativa de lo que UMAP y t-SNE muestran visualmente.
#
# Rango [-1, 1]:
#   cercano a  1  → grupos compactos y bien separados entre sí
#   cercano a  0  → grupos solapados
#   negativo      → puntos mal asignados a sus grupos
#
# Se calcula sobre los vectores latentes originales con distancia
# euclídea, que es la métrica natural del espacio latente del VAE.
#
# NOTA: solo tiene sentido si hay más de una clase. Si solo hay
# puntuación 5, se omite automáticamente.
# ============================================================
print("\n=== SILHOUETTE SCORE (espacio latente original) ===")

if n_classes < 2:
    print("  Solo hay una clase presente. Silhouette score no aplicable.")
    print("  (Para obtener resultados significativos necesitas al menos")
    print("   dos puntuaciones distintas en el dataset.)")
    global_silhouette = None
else:
    # Silhouette global: un único número que resume la separabilidad
    global_silhouette = silhouette_score(vectors, labels, metric='euclidean')
    print(f"  Silhouette global: {global_silhouette:.4f}")

    # Silhouette por clase: cuánto contribuye cada puntuación a la separación
    # Un valor alto en una clase indica que sus vectores están bien agrupados
    # y lejos de las otras clases.
    sample_silhouettes = silhouette_samples(vectors, labels, metric='euclidean')
    print("\n  Silhouette medio por puntuación clínica:")
    for lbl in unique_labels:
        mask = labels == lbl
        mean_s = sample_silhouettes[mask].mean()
        n      = mask.sum()
        print(f"    Score {int(lbl):>2d} | N={n:>4d} | Silhouette medio: {mean_s:.4f}")

    # AÑADIDO: Gráfica de barras del Silhouette por clase
    # Permite ver de un vistazo qué puntuaciones están mejor separadas
    # en el espacio latente y cuáles se solapan con otras.
    fig_sil, ax_sil = plt.subplots(figsize=(8, 4))
    means_per_class = [sample_silhouettes[labels == lbl].mean() for lbl in unique_labels]
    bar_colors      = [colors[lbl] for lbl in unique_labels]
    bars = ax_sil.bar(
        [f'Score {int(l)}' for l in unique_labels],
        means_per_class,
        color=bar_colors,
        edgecolor='black',
        linewidth=0.5
    )
    ax_sil.axhline(y=global_silhouette, color='red', linestyle='--',
                   linewidth=1.5, label=f'Global: {global_silhouette:.4f}')
    ax_sil.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax_sil.set_xlabel('Puntuación clínica')
    ax_sil.set_ylabel('Silhouette medio')
    ax_sil.set_title('Silhouette score por puntuación clínica\n(espacio latente original)')
    ax_sil.legend()
    ax_sil.set_ylim(-1, 1)
    ax_sil.grid(True, alpha=0.3, axis='y')

    # Añadir valores sobre las barras
    for bar, val in zip(bars, means_per_class):
        ax_sil.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02 * np.sign(bar.get_height()),
            f'{val:.3f}',
            ha='center', va='bottom', fontsize=9
        )

    plt.tight_layout()
    sil_path = os.path.join(OUTPUT_DIR, 'silhouette_by_class.png')
    plt.savefig(sil_path, dpi=150)
    plt.close()
    print(f"\n  Gráfica guardada: {sil_path}")


# ============================================================
# PRE-PROCESADO PARA t-SNE
# ============================================================
pca_dims = min(50, vectors.shape[1])
print(f"\nAplicando PCA previo a {pca_dims} dimensiones para t-SNE...")
pca         = PCA(n_components=pca_dims, random_state=RANDOM_STATE)
vectors_pca = pca.fit_transform(vectors)
print(f"  Varianza explicada acumulada: {pca.explained_variance_ratio_.sum():.4f}")


# ============================================================
# UMAP — exploración de hiperparámetros
# ============================================================
print("\n=== UMAP ===")

n_rows = len(UMAP_N_NEIGHBORS)
n_cols = len(UMAP_MIN_DIST)

fig_umap, axes_umap = plt.subplots(
    n_rows, n_cols,
    figsize=(6 * n_cols, 5 * n_rows)
)
if n_rows == 1 and n_cols == 1:
    axes_umap = np.array([[axes_umap]])
elif n_rows == 1:
    axes_umap = axes_umap[np.newaxis, :]
elif n_cols == 1:
    axes_umap = axes_umap[:, np.newaxis]

# AÑADIDO: guardamos los embeddings UMAP para no recalcularlos
umap_embeddings = {}

for i, n_neighbors in enumerate(UMAP_N_NEIGHBORS):
    for j, min_dist in enumerate(UMAP_MIN_DIST):
        print(f"  UMAP: n_neighbors={n_neighbors}, min_dist={min_dist} ...", end=" ")

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            init='spectral',
            random_state=RANDOM_STATE
        )
        embedding = reducer.fit_transform(vectors)

        # AÑADIDO: guardamos embedding en disco para reutilizarlo
        key = f"umap_nn{n_neighbors}_md{min_dist}"
        umap_embeddings[key] = embedding

        # AÑADIDO: Silhouette sobre el embedding 2D de UMAP
        # Complementa el Silhouette del espacio latente original:
        # mide si la proyección 2D preserva la separabilidad.
        sil_2d = None
        if n_classes >= 2:
            sil_2d = silhouette_score(embedding, labels, metric='euclidean')

        scatter_plot(
            axes_umap[i, j],
            embedding,
            labels,
            f'UMAP  n_neighbors={n_neighbors}  min_dist={min_dist}',
            silhouette=sil_2d
        )
        msg = f"OK | Silhouette 2D: {sil_2d:.4f}" if sil_2d is not None else "OK"
        print(msg)

fig_umap.suptitle('UMAP — Exploración de hiperparámetros', fontsize=14, fontweight='bold')
plt.tight_layout()
umap_path = os.path.join(OUTPUT_DIR, 'umap_grid.png')
plt.savefig(umap_path, dpi=150)
plt.close()
print(f"  Guardado: {umap_path}")

# AÑADIDO: guardamos todos los embeddings UMAP en un único archivo
np.save(os.path.join(OUTPUT_DIR, 'umap_embeddings.npy'), umap_embeddings)
print(f"  Embeddings UMAP guardados: umap_embeddings.npy")


# ============================================================
# t-SNE — exploración de perplexity
# ============================================================
print("\n=== t-SNE ===")

n_tsne_cols = 2
n_tsne_rows = int(np.ceil(len(TSNE_PERPLEXITIES) / n_tsne_cols))

fig_tsne, axes_tsne = plt.subplots(
    n_tsne_rows, n_tsne_cols,
    figsize=(7 * n_tsne_cols, 6 * n_tsne_rows)
)
axes_tsne = np.array(axes_tsne).reshape(n_tsne_rows, n_tsne_cols)

# AÑADIDO: guardamos los embeddings t-SNE para no recalcularlos
tsne_embeddings = {}

for idx, perplexity in enumerate(TSNE_PERPLEXITIES):
    row = idx // n_tsne_cols
    col = idx % n_tsne_cols

    perp = min(perplexity, len(vectors) - 1)
    print(f"  t-SNE: perplexity={perp}, n_iter={TSNE_N_ITER} ...", end=" ")

    tsne = TSNE(
        n_components=2,
        perplexity=perp,
        max_iter=TSNE_N_ITER,
        method='barnes_hut',
        random_state=RANDOM_STATE,
        init='pca'
    )
    embedding = tsne.fit_transform(vectors_pca)

    # AÑADIDO: guardamos embedding
    key = f"tsne_perp{perp}"
    tsne_embeddings[key] = embedding

    # AÑADIDO: Silhouette sobre el embedding 2D de t-SNE
    sil_2d = None
    if n_classes >= 2:
        sil_2d = silhouette_score(embedding, labels, metric='euclidean')

    scatter_plot(
        axes_tsne[row, col],
        embedding,
        labels,
        f't-SNE  perplexity={perp}  n_iter={TSNE_N_ITER}',
        silhouette=sil_2d
    )
    msg = f"OK | Silhouette 2D: {sil_2d:.4f}" if sil_2d is not None else "OK"
    print(msg)

total_plots = n_tsne_rows * n_tsne_cols
for idx in range(len(TSNE_PERPLEXITIES), total_plots):
    row = idx // n_tsne_cols
    col = idx % n_tsne_cols
    axes_tsne[row, col].set_visible(False)

fig_tsne.suptitle('t-SNE — Exploración de perplexity', fontsize=14, fontweight='bold')
plt.tight_layout()
tsne_path = os.path.join(OUTPUT_DIR, 'tsne_grid.png')
plt.savefig(tsne_path, dpi=150)
plt.close()
print(f"  Guardado: {tsne_path}")

# AÑADIDO: guardamos todos los embeddings t-SNE
np.save(os.path.join(OUTPUT_DIR, 'tsne_embeddings.npy'), tsne_embeddings)
print(f"  Embeddings t-SNE guardados: tsne_embeddings.npy")


# ============================================================
# FIGURA COMPARATIVA: mejor UMAP vs mejor t-SNE
# ============================================================
print("\n=== Figura comparativa (UMAP vs t-SNE) ===")

best_n_neighbors = UMAP_N_NEIGHBORS[len(UMAP_N_NEIGHBORS) // 2]
best_min_dist    = UMAP_MIN_DIST[len(UMAP_MIN_DIST) // 2]
best_perplexity  = min(TSNE_PERPLEXITIES[len(TSNE_PERPLEXITIES) // 2], len(vectors) - 1)

# Reutilizamos los embeddings ya calculados
umap_key = f"umap_nn{best_n_neighbors}_md{best_min_dist}"
tsne_key = f"tsne_perp{best_perplexity}"
umap_embedding = umap_embeddings[umap_key]
tsne_embedding = tsne_embeddings[tsne_key]
print(f"  Reutilizando embeddings ya calculados (sin recalcular)")

sil_umap_2d = silhouette_score(umap_embedding, labels) if n_classes >= 2 else None
sil_tsne_2d = silhouette_score(tsne_embedding, labels) if n_classes >= 2 else None

fig_comp, (ax_u, ax_t) = plt.subplots(1, 2, figsize=(14, 6))
scatter_plot(ax_u, umap_embedding, labels,
             f'UMAP  (n_neighbors={best_n_neighbors}, min_dist={best_min_dist})',
             silhouette=sil_umap_2d)
scatter_plot(ax_t, tsne_embedding, labels,
             f't-SNE  (perplexity={best_perplexity}, n_iter={TSNE_N_ITER})',
             silhouette=sil_tsne_2d)

# AÑADIDO: si hay Silhouette global, lo añadimos al suptitle
suptitle = 'Espacio latente — UMAP vs t-SNE'
if global_silhouette is not None:
    suptitle += f'\nSilhouette espacio latente original: {global_silhouette:.4f}'
fig_comp.suptitle(suptitle, fontsize=13, fontweight='bold')

plt.tight_layout()
comp_path = os.path.join(OUTPUT_DIR, 'latent_comparison.png')
plt.savefig(comp_path, dpi=150)
plt.close()
print(f"  Guardado: {comp_path}")


# ============================================================
# RESUMEN FINAL EN LOG
# ============================================================
print("\n=== Resumen final ===")
if global_silhouette is not None:
    print(f"Silhouette espacio latente original: {global_silhouette:.4f}")
    print("  > 0.5  → separación clara entre grupos")
    print("  0.2-0.5 → separación moderada")
    print("  < 0.2  → grupos muy solapados")
print(f"\nResultados guardados en '{OUTPUT_DIR}/':")
print(f"  silhouette_by_class.png  -> Silhouette por puntuación clínica")
print(f"  umap_grid.png            -> cuadricula UMAP")
print(f"  tsne_grid.png            -> cuadricula t-SNE")
print(f"  latent_comparison.png    -> figura comparativa final")
print(f"  umap_embeddings.npy      -> embeddings UMAP guardados")
print(f"  tsne_embeddings.npy      -> embeddings t-SNE guardados")

sys.stdout.log.close()
