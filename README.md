# vae-cognitive-tfg

Código de las simulaciones del TFG de Ingeniería de Tecnologías de Telecomunicaciones de **Carlos Cruz Martínez**, Universidad de Málaga:

> Diseño y evaluación de un autocodificador variacional para el análisis de dibujos del test del reloj.

Se estudia la reconstrucción de imágenes con un VAE convolucional mediante barridos de resolución, regularización y dimensión latente. La configuración seleccionada dentro de los experimentos realizados utiliza resolución 384, beta 7 y dimensión latente 128.

## Organización

```text
scripts/
├── Preprocesamiento/     # Resolución y recorte
├── beta/                 # Peso de regularización
└── latent_dim/           # Dimensión latente y ejecución posterior
README.md
.gitignore
```

Cada configuración conserva su `codigo.py` y su análisis `UMAP_tSNE.py`. La carpeta `latent_dim128_cmmd_fid` contiene la ejecución posterior del modelo seleccionado y `fid_cmmd_eval.py`. Los análisis utilizan salidas de un entrenamiento previo.

## Instalación

Python 3.12. Crear un entorno virtual y activarlo (`.venv\Scripts\activate` en Windows o `source .venv/bin/activate` en Linux):

```bash
python -m venv .venv
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install "numpy>=1.26,<3" "scipy>=1.11,<2" "matplotlib>=3.8,<4" "Pillow>=10,<13" pytorch-msssim==1.0.0 "scikit-learn>=1.5,<2" "umap-learn>=0.5.7,<0.6" "transformers>=4.45,<5"
```

La instalación anterior utiliza CPU. Para GPU, instalar las versiones correspondientes de PyTorch y torchvision desde [PyTorch](https://pytorch.org/get-started/previous-versions/). El dispositivo se puede elegir mediante `TFG_DEVICE`; por defecto se utiliza la primera GPU disponible o CPU. FID/CMMD requieren descargar modelos preentrenados.

## Datos y ejecución

Los datos, resultados, imágenes y pesos no se publican. Es necesario obtener acceso autorizado. `RUTA_DATOS` debe contener las seis carpetas originales: `0_no_clock`, `1_severe_vis`, `2_mod_vis_xhands`, `3_hands_vis_errors`, `4_minor_VIS_errors` y `5_perfect_clock`. El entrenamiento selecciona la puntuación 5.

Desde la raíz del repositorio, en PowerShell (Windows), configura rutas absolutas y crea una carpeta nueva para cada ejecución:

```powershell
$env:TFG_DATA_DIR = (Resolve-Path "RUTA_DATOS").Path
New-Item -ItemType Directory -Path "results/modelo" -ErrorAction Stop
$env:TFG_RUN_DIR = (Resolve-Path "results/modelo").Path
$env:TFG_LATENT_DIR = Join-Path $env:TFG_RUN_DIR "latent"
$env:TFG_VIZ_DIR = Join-Path $env:TFG_RUN_DIR "latent_viz"
python scripts/latent_dim/latent_dim128_cmmd_fid/codigo.py
```

En Linux, define las mismas variables con `export` y crea previamente la carpeta de salida. Para otro experimento, cambia la ruta de `codigo.py`, por ejemplo a `scripts/beta/beta7/codigo.py`, y utiliza otra carpeta de salida. No reutilices una carpeta con resultados anteriores.

Ejecuta los experimentos secuencialmente: escriben un CSV comparativo en el directorio padre de las salidas. El entrenamiento guarda registros, particiones, pesos, reconstrucciones y representaciones latentes. Los scripts de análisis necesitan esas salidas. Para repetirlos por separado, conserva las variables anteriores y ejecútalos desde la carpeta de la ejecución utilizando la ruta absoluta del script. `UMAP_tSNE.py` genera las proyecciones; `fid_cmmd_eval.py`, disponible en la ejecución posterior, calcula las métricas complementarias.

## Alcance

Es un estudio experimental de reconstrucción, no un sistema de diagnóstico clínico validado. La auditoría detectó duplicados que cruzan particiones al aplicar los índices guardados al orden actual de los archivos; falta un manifiesto histórico para verificar esa correspondencia. Evaluar la generalización requiere agrupar duplicados/participantes y repetir los experimentos. Los scripts conservan el procedimiento histórico, no corrigen automáticamente sus particiones.

Las métricas históricas se promedian por lote. La fila R256 del CSV/Excel y el JSON conservado corresponden a ejecuciones diferentes; las tablas reproducen la serie histórica sin mezclarlas. MiB significa 2²⁰ bytes, aunque algunas claves antiguas del JSON terminen en `_mb`.

Durante la revisión se comprobaron la instalación CPU y el funcionamiento básico del modelo con herramientas auxiliares conservadas fuera de este repositorio. No se ha repetido el entrenamiento completo ni toda la evaluación posterior en ese entorno. La reproducción exacta requiere datos autorizados, particiones y condiciones de ejecución originales.
