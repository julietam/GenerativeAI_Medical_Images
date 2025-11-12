# Guía de Instalación Completa

Esta guía te ayudará a configurar el entorno para ejecutar todos los notebooks y demos del repositorio.

---

## 📋 Requisitos del Sistema

### Hardware Mínimo
- **RAM**: 16 GB (32 GB recomendado)
- **GPU**: NVIDIA GPU con 6+ GB VRAM (RTX 3060 o superior recomendado)
  - Para demos: GPU opcional, CPU suficiente
  - Para entrenamiento: GPU **requerida**
- **Disco**: 50 GB libres (para datasets y modelos)

### Software
- **OS**: Linux (Ubuntu 20.04+), macOS, o Windows 10/11
- **Python**: 3.8, 3.9, o 3.10
- **CUDA**: 11.8 o 12.1 (si usas GPU NVIDIA)

---

## 🚀 Instalación Rápida (Conda - Recomendado)

### Paso 1: Instalar Miniconda

Si no tienes Conda instalado:

**Linux/macOS:**
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# Sigue las instrucciones en pantalla
```

**Windows:**
Descarga desde: https://docs.conda.io/en/latest/miniconda.html

### Paso 2: Crear Entorno

```bash
# Clonar repositorio
git clone https://github.com/tu-usuario/medical-image-generation.git
cd medical-image-generation

# Crear entorno conda
conda env create -f environment.yml

# Activar entorno
conda activate medgen
```

### Paso 3: Verificar Instalación

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import monai; print(f'MONAI: {monai.__version__}')"
python -c "import medigan; print(f'medigan: {medigan.__version__}')"
```

**Output esperado:**
```
PyTorch: 2.0.0
CUDA available: True
MONAI: 1.3.0
medigan: 0.2.0
```

---

## 🐍 Instalación Alternativa (pip + virtualenv)

### Para Linux/macOS:

```bash
# Crear virtualenv
python3 -m venv venv
source venv/bin/activate

# Actualizar pip
pip install --upgrade pip

# Instalar PyTorch (con CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Instalar dependencias
pip install -r requirements.txt
```

### Para Windows:

```cmd
# Crear virtualenv
python -m venv venv
venv\Scripts\activate

# Actualizar pip
python -m pip install --upgrade pip

# Instalar PyTorch (con CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Instalar dependencias
pip install -r requirements.txt
```

---

## 🎮 Configuración de GPU

### Verificar CUDA

```bash
nvidia-smi
```

Deberías ver información de tu GPU y versión de CUDA.

### Seleccionar GPU Específica

Si tienes múltiples GPUs:

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Usar primera GPU
```

O desde terminal:
```bash
export CUDA_VISIBLE_DEVICES=0
```

### Sin GPU (Solo CPU)

Si no tienes GPU, puedes usar CPU (será más lento):

```bash
# En environment.yml, comentar:
# - cudatoolkit=11.8

# O en requirements.txt, instalar PyTorch CPU:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

---

## 📦 Descargar Datasets

### Opción 1: Script Automático (Datasets pequeños)

```bash
cd data/
bash download_datasets.sh
cd ..
```

Esto descarga ~2 GB de datasets de muestra.

### Opción 2: Datasets Completos (Manual)

#### fastMRI (MRI Reconstruction)
1. Registrarse en https://fastmri.org/
2. Aceptar términos de uso
3. Descargar `knee_singlecoil_train` o `brain_multicoil_train`
4. Extraer en `data/fastmri/`

```bash
mkdir -p data/fastmri
# Copiar archivos descargados aquí
```

#### BraTS (Brain Tumor Segmentation)
1. Registrarse en https://www.med.upenn.edu/cbica/brats2023/
2. Descargar training data
3. Extraer en `data/brats/`

```bash
mkdir -p data/brats
# Copiar archivos aquí
```

#### ChestX-ray14
1. Ir a https://nihcc.app.box.com/v/ChestXray-NIHCC
2. Descargar `images_001.tar.gz` a `images_012.tar.gz` (47 GB)
3. Descargar `Data_Entry_2017_v2020.csv`
4. Extraer en `data/chest_xray/`

```bash
mkdir -p data/chest_xray/images
# Extraer todos los .tar.gz en images/
```

#### IXI Dataset (Brain MRI)
1. Ir a https://brain-development.org/ixi-dataset/
2. Descargar T1, T2, PD sequences
3. Extraer en `data/ixi/`

```bash
mkdir -p data/ixi/{T1,T2,PD}
```

---

## 🔧 Configuración de Jupyter

### Instalar Kernel

```bash
conda activate medgen
python -m ipykernel install --user --name medgen --display-name "Python (medgen)"
```

### Iniciar Jupyter

```bash
jupyter notebook notebooks/
```

O con JupyterLab:
```bash
jupyter lab notebooks/
```

### Extensiones Útiles

```bash
# JupyterLab extensions
pip install jupyterlab-lsp
pip install jupyter-resource-usage

# Widgets para visualización
jupyter nbextension enable --py widgetsnbextension
```

---

## 🧪 Testing de Instalación

Ejecuta este script para verificar que todo funciona:

```python
# test_installation.py

import sys
print(f"Python: {sys.version}")

# Core libraries
import torch
print(f"✅ PyTorch {torch.__version__}")
print(f"   CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

import torchvision
print(f"✅ torchvision {torchvision.__version__}")

# Medical imaging
import monai
print(f"✅ MONAI {monai.__version__}")

import medigan
print(f"✅ medigan {medigan.__version__}")

import nibabel
print(f"✅ nibabel {nibabel.__version__}")

import pydicom
print(f"✅ pydicom {pydicom.__version__}")

import SimpleITK as sitk
print(f"✅ SimpleITK {sitk.Version.VersionString()}")

# Scientific
import numpy as np
print(f"✅ numpy {np.__version__}")

import scipy
print(f"✅ scipy {scipy.__version__}")

import pandas as pd
print(f"✅ pandas {pd.__version__}")

# Visualization
import matplotlib
print(f"✅ matplotlib {matplotlib.__version__}")

# Metrics
try:
    import pytorch_fid
    print(f"✅ pytorch-fid installed")
except:
    print(f"⚠️  pytorch-fid not installed (optional)")

print("\n🎉 Todas las dependencias principales están instaladas!")
```

Ejecutar:
```bash
python test_installation.py
```

---

## ⚠️ Problemas Comunes

### Error: "CUDA out of memory"

**Solución:**
- Reducir `batch_size` en notebooks
- Usar `torch.cuda.empty_cache()` entre entrenamientos
- Cerrar otros programas que usen GPU

```python
# Al inicio del notebook
import torch
torch.cuda.empty_cache()

# Reducir batch size
batch_size = 8  # En lugar de 32
```

### Error: "No module named 'monai'"

**Solución:**
```bash
conda activate medgen
pip install monai
```

### Error: "FileNotFoundError" en datasets

**Solución:**
- Verificar que descargaste los datasets
- Revisar rutas en notebooks (cambiar `data/` si es necesario)

```python
# Verificar rutas
import os
print(os.listdir('data/'))
```

### Error: PyTorch no detecta GPU

**Solución:**
```bash
# Verificar CUDA
nvidia-smi

# Reinstalar PyTorch con CUDA
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### medigan: "Model not found"

**Solución:**
```python
# Primera vez, instalar dependencias del modelo
from medigan import Generators
generators = Generators()

generators.generate(
    model_id="00008_C-DCGAN_MMG_MASSES",
    num_samples=1,
    install_dependencies=True  # Importante!
)
```

---

## 🐳 Docker (Opcional)

Para un entorno completamente reproducible:

### Dockerfile

```dockerfile
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Expose Jupyter port
EXPOSE 8888

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root", "--no-browser"]
```

### Build y Run

```bash
# Build
docker build -t medgen:latest .

# Run
docker run -it --gpus all -p 8888:8888 -v $(pwd):/workspace medgen:latest
```

---

## 📊 Configuración de Weights & Biases (Opcional)

Para tracking de experimentos:

```bash
# Instalar
pip install wandb

# Login
wandb login
# Pegar tu API key desde https://wandb.ai/settings
```

Usar en notebooks:
```python
import wandb

wandb.init(
    project="medical-image-generation",
    config={
        "model": "DCGAN",
        "dataset": "ChestXray",
        "batch_size": 32
    }
)

# Loggear métricas
wandb.log({"loss_G": loss_g, "loss_D": loss_d})
```

---

## 🔗 Recursos Adicionales

- **MONAI Tutorials**: https://github.com/Project-MONAI/tutorials
- **PyTorch Tutorials**: https://pytorch.org/tutorials/
- **fastMRI Tutorial**: https://github.com/facebookresearch/fastMRI
- **medigan Examples**: https://github.com/RichardObi/medigan/tree/main/examples

---

## 📧 Soporte

Si tienes problemas:
1. Revisa [Issues](https://github.com/tu-usuario/medical-image-generation/issues)
2. Crea un nuevo issue con detalles del error
3. Contacta al instructor: tu.email@universidad.edu

---

**Última actualización**: Noviembre 2025