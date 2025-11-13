# Generación y Validación de Imágenes Médicas: Retos, Riesgos y Oportunidades

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

> Repositorio educativo para la plática sobre modelos generativos en imagenología médica  
> **Audiencia**: Estudiantes de Maestría en Ciencias e Ingeniería de la Computación

---

## 📋 Contenido del Repositorio

```
medical-image-generation/
├── README.md                          # Este archivo
├── slides/                            # Presentación
│   ├── slides.pdf
│   └── figures/
├── notebooks/                         # Jupyter notebooks
│   ├── 01_introduction.ipynb         # Explorando datasets médicos
│   ├── 02_gans_basics.ipynb          # GANs desde cero
│   ├── 03_pix2pix_mri.ipynb         # MRI T1→T2 con Pix2Pix
│   ├── 04_cyclegan_ct_mri.ipynb     # MRI↔CT con CycleGAN
│   ├── 05_diffusion_xray.ipynb      # Chest X-ray con Diffusion
│   └── 06_medigan_demo.ipynb        # Demo con modelos pre-entrenados
├── src/                               # Código fuente
│   ├── models/
│   │   ├── dcgan.py
│   │   ├── pix2pix.py
│   │   ├── cyclegan.py
│   │   └── diffusion.py
│   ├── data/
│   │   ├── datasets.py
│   │   └── preprocessing.py
│   ├── utils/
│   │   ├── metrics.py                # FID, IS, SSIM, PSNR
│   │   ├── visualization.py
│   │   └── medical_utils.py          # DICOM, NIfTI handling
│   └── train.py
├── data/                              # Datasets (gitignore, solo scripts)
│   ├── download_datasets.sh
│   └── README.md                      # Instrucciones de descarga
├── results/                           # Resultados generados
│   └── .gitkeep
├── docs/                              # Documentación adicional
│   ├── PLAN_PLATICA.md               # Plan completo de la plática
│   ├── REFERENCIAS.md                # Papers y recursos
│   └── SETUP.md                      # Guía de instalación detallada
├── requirements.txt                   # Dependencias Python
├── environment.yml                    # Conda environment
└── LICENSE

```

---

## 🚀 Inicio Rápido

### 1. Clonar el Repositorio

```bash
git clone https://github.com/tu-usuario/medical-image-generation.git
cd medical-image-generation
```

### 2. Configurar Entorno

#### Opción A: Conda (Recomendado)

```bash
conda env create -f environment.yml
conda activate medgen
```

#### Opción B: pip + virtualenv

```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Descargar/preparar datasets abiertos (MRI/CT)

Ver guías en `data/README.md`. Resumen rápido (Windows PowerShell):

- OpenNeuro ds005533 (T1w/T2w) → pares para Pix2Pix:
  Opción A (GitHub Actions – recomendado):
  - En la pestaña Actions del repo, ejecuta el workflow "Fetch BIDS pairs" (workflow_dispatch)
    con dataset=ds005533, version=1.0.0, max_pairs=3. Esto descargará T1w/T2w y subirá PNGs a `data/paired_mri/`.
  
  Opción B (local):
  1) Instala OpenNeuro CLI (requiere Node.js):
     npm install -g openneuro-cli
  2) Descarga solo anat/T1w y T2w:
     powershell -ExecutionPolicy Bypass -File scripts/download_openneuro_ds005533.ps1
  3) Convierte 3 pares a PNG 256x256:
     python scripts/prepare_bids_pairs.py --ds_root data/openneuro/ds005533 --out_root data/paired_mri --max_pairs 3

- IXI Brain MRI (T1/T2 → pares para Pix2Pix):

```powershell
python -m src.data.datasets ixi_pairs --ixi_root "C:\\path\\to\\IXI" --out_root data/paired_mri --img_size 256 --take_slices 1
```

- Chest CT (e.g., MosMedData → dominio B para CycleGAN):

```powershell
python -m src.data.datasets ct_slices --ct_root "C:\\path\\to\\CT_NIfTI" --out_root data/unpaired_ct --img_size 256 --take_slices 3
```

Luego ajusta las rutas en los notebooks 03 y 04 si usas una estructura distinta.

### 4. Ejecutar Notebooks

```powershell
jupyter notebook
```

Recomendamos seguir el orden:
1. `01_introduction_mri.ipynb` - MONAI: carga y preprocesamiento IXI (3D → 2D)
2. `02_gans_basics.ipynb` - Entrenar DCGAN desde cero
3. `03_pix2pix_mri.ipynb` - T1→T2 con Pix2Pix (usa `data/paired_mri`)
4. `04_cyclegan_ct_mri.ipynb` - MRI↔CT con CycleGAN (usa `data/unpaired_*`)
5. `05_diffusion_xray.ipynb` - Diffusion (DDPM)

---

## 📚 Estructura de la Plática

### 1. Introducción (8 min)
- Retos del flujo clínico: Diagnostic, Treatment, Prognosis
- Motivaciones computacionales

### 2. Modelos Generativos (12 min)
- **GANs**: DCGAN, Pix2Pix, CycleGAN, StyleGAN
- **Diffusion Models**: DDPM, Latent Diffusion, Medfusion
- **Comparación**: Fidelidad vs. Diversidad

### 3. Estado del Arte (10 min)
- Papers clave 2024-2025
- Medfusion: Diffusion supera GANs en diversidad

### 4. Modalidades (12 min)
- **Brain MRI**: T1→T2, Compressed sensing, Synthetic-CT
- **CT**: Low-dose denoising, Artifact reduction
- **Chest X-ray**: Data augmentation, Super-resolution

### 5. Demo Práctica (10 min)
- Notebooks interactivos con medigan y MONAI
- Generación multi-modalidad

### 6. Riesgos (8 min)
- Hallucinations, Mode collapse
- Validación clínica insuficiente
- Bias y reproducibilidad

---

## 🧪 Demos y Scripts

### Gradio: Pix2Pix MRI T1→T2
- Lanza una UI mínima para cargar una T1 (PNG/JPG) y generar T2.
- Requiere un checkpoint del generador (por ejemplo, entrenado con `scripts/train_pix2pix.py`).

```powershell
# Ejemplo
python apps/pix2pix_demo.py --checkpoint ckpts/pix2pix/G_best.pt --device cpu --port 7860
```

### Entrenamiento por script (Pix2Pix)
- Configurable vía YAML en `configs/pix2pix.yaml`.

```powershell
python scripts/train_pix2pix.py --config configs/pix2pix.yaml
```

Salida: checkpoints en `ckpts/pix2pix/` y métricas en consola.

---

## 🛠️ Herramientas Utilizadas

### Frameworks Principales

- **[MONAI](https://monai.io/)**: Framework PyTorch para medical imaging
- **[medigan](https://github.com/RichardObi/medigan)**: 21+ modelos pre-entrenados
- **[TorchIO](https://github.com/fepegar/torchio)**: Preprocesamiento 3D/4D
- **PyTorch 2.0+**: Deep learning framework

### Datasets Públicos

- **[fastMRI](https://fastmri.org/)**: MRI reconstruction challenge
- **[BraTS](https://www.med.upenn.edu/cbica/brats/)**: Brain tumor segmentation
- **[ChestX-ray14](https://nihcc.app.box.com/v/ChestXray-NIHCC)**: 112K chest X-rays
- **[IXI Dataset](https://brain-development.org/ixi-dataset/)**: Brain MRI multi-modal

---

## 📓 Notebooks Detallados

### `01_introduction.ipynb`
**Objetivo**: Familiarizarse con datos médicos  
**Contenido**:
- Cargar imágenes DICOM y NIfTI
- Visualización 3D de MRI y CT
- Estadísticas de datasets médicos
- Desafíos: Desbalance de clases, tamaño reducido

**Duración estimada**: 20 min

---

### `02_gans_basics.ipynb`
**Objetivo**: Implementar DCGAN desde cero  
**Contenido**:
- Arquitectura Generator y Discriminator
- Training loop con adversarial loss
- Generar chest X-rays sintéticos
- Métricas: FID, IS
- Detectar mode collapse

**Duración estimada**: 45 min

**Código ejemplo**:
```python
# Generator architecture
generator = nn.Sequential(
    nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0),
    nn.BatchNorm2d(512),
    nn.ReLU(True),
    # ... más capas
)

# Training loop
for epoch in range(num_epochs):
    for real_images, _ in dataloader:
        # Train Discriminator
        loss_D = train_discriminator(real_images, generator, discriminator)
        
        # Train Generator
        loss_G = train_generator(generator, discriminator)
```

---

### `03_pix2pix_mri.ipynb`
**Objetivo**: Síntesis T1→T2 en Brain MRI  
**Contenido**:
- Cargar pares aligned de IXI dataset
- Implementar Pix2Pix (U-Net + PatchGAN)
- L1 + adversarial loss
- Evaluar con SSIM, PSNR
- Comparar con ground truth

**Duración estimada**: 60 min

**Dataset**: IXI Brain MRI (T1, T2, PD)

---

### `04_cyclegan_ct_mri.ipynb`
**Objetivo**: Traducción unpaired MRI↔CT  
**Contenido**:
- CycleGAN architecture (2 generators, 2 discriminators)
- Cycle consistency loss
- Synthetic-CT generation
- Evaluar MAE en Hounsfield Units (HU)

**Duración estimada**: 60 min

**Aplicación**: Radioterapia planning sin CT real

---

### `05_diffusion_xray.ipynb`
**Objetivo**: Generar chest X-rays con Diffusion Models  
**Contenido**:
- DDPM forward/reverse process
- Entrenar denoising U-Net
- Conditional generation (por patología)
- Comparar diversidad vs. GANs (Precision-Recall)

**Duración estimada**: 90 min

**Dataset**: ChestX-ray14 subset

---

### `06_medigan_demo.ipynb`
**Objetivo**: Demo rápida con modelos pre-entrenados  
**Contenido**:
- Instalar medigan
- Listar 21+ modelos disponibles
- Generar:
  - Mamografías (C-DCGAN)
  - Chest X-rays (DCGAN)
  - Brain MRI (si disponible)
- Visualizar resultados
- Explorar latent space

**Duración estimada**: 15 min

**Ventaja**: Sin entrenamiento, resultados inmediatos

---

## 🎯 Ejercicios Prácticos

### Ejercicio 1: Data Augmentation para Clasificación
**Objetivo**: Mejorar clasificador de neumotórax con datos sintéticos

**Pasos**:
1. Entrenar clasificador baseline (ResNet-18) con datos reales (N=500)
2. Generar 1000 X-rays sintéticos con DCGAN
3. Re-entrenar con datos reales + sintéticos
4. Comparar accuracy, precision, recall

**Pregunta**: ¿Cuántos datos sintéticos son óptimos? (0%, 50%, 100%, 200%)

---

### Ejercicio 2: Evaluación de Calidad
**Objetivo**: Implementar métricas de evaluación

**Tareas**:
- Calcular FID entre imágenes reales y sintéticas
- Implementar Precision-Recall para GANs
- Evaluar SSIM/PSNR para reconstrucción
- Comparar DCGAN vs. StyleGAN vs. Diffusion

---

### Ejercicio 3: Detección de Hallucinations
**Objetivo**: Identificar estructuras anatómicas falsas

**Método**:
- Generar 100 brain MRIs sintéticos
- Usar segmentador pre-entrenado (FreeSurfer)
- Detectar estructuras anatómicamente imposibles
- Filtrar imágenes con hallucinations

---

## 🧪 Validación y Métricas (TorchMetrics)

Este repositorio usa TorchMetrics para evaluar la calidad de los modelos generativos:

- DCGAN (02_gans_basics.ipynb): FrechetInceptionDistance (FID) e InceptionScore (IS)
- Pix2Pix (03_pix2pix_mri.ipynb): StructuralSimilarityIndexMeasure (SSIM) y PeakSignalNoiseRatio (PSNR) contra ground truth
- CycleGAN (04_cyclegan_ct_mri.ipynb): SSIM y PSNR sobre la consistencia de ciclo (A→B→A y B→A→B)
- Diffusion (05_diffusion_xray.ipynb): FID entre muestras generadas y el set de validación

Salida y guardado de resultados:
- Las figuras y grids se guardan automáticamente en `outputs/<modelo>/`:
  - `outputs/dcgan/metrics.png`, `outputs/dcgan/samples.png`
  - `outputs/pix2pix/metrics.png`, `outputs/pix2pix/val_grid.png`
  - `outputs/cyclegan/metrics.png`, `outputs/cyclegan/a_b_a.png`, `outputs/cyclegan/b_a_b.png`
  - `outputs/diffusion/metrics.png`, `outputs/diffusion/samples.png`

Instalación rápida de métricas:

```bash
pip install torchmetrics
```

Nota: Para FID/IS, TorchMetrics descarga/usa un Inception por defecto. Las imágenes se re-escalan a [0,1] en el notebook antes de evaluar.

---

## 📊 Resultados Esperados

Al completar este repositorio, los estudiantes podrán:

✅ **Implementar** GANs y Diffusion Models desde cero  
✅ **Entrenar** modelos para MRI, CT, y X-ray synthesis  
✅ **Evaluar** calidad con FID, IS, SSIM, y métricas clínicas  
✅ **Detectar** problemas como mode collapse y hallucinations  
✅ **Aplicar** modelos pre-entrenados con medigan  
✅ **Entender** trade-offs: Fidelidad vs. Diversidad vs. Velocidad  

---

## 📖 Referencias Principales

### Papers Fundamentales

**GANs**:
- Goodfellow et al. (2014). "Generative Adversarial Networks". NeurIPS.
- Radford et al. (2016). "Unsupervised Representation Learning with DCGANs". ICLR.
- Isola et al. (2017). "Image-to-Image Translation with Conditional GANs". CVPR.
- Zhu et al. (2017). "Unpaired Image-to-Image Translation using CycleGANs". ICCV.
- Karras et al. (2019). "A Style-Based Generator Architecture for GANs". CVPR.

**Diffusion Models**:
- Ho et al. (2020). "Denoising Diffusion Probabilistic Models". NeurIPS.
- Rombach et al. (2022). "High-Resolution Image Synthesis with Latent Diffusion". CVPR.
- Friedrich et al. (2023). "Medfusion: Latent DDPMs vs GANs for Medical Imaging". Scientific Reports.

**Medical Imaging Reviews**:
- Oulmalme et al. (2025). "Systematic Review of Generative AI for Medical Image Enhancement".
- Ibrahim et al. (2025). "Generative AI for Synthetic Data Across Multiple Modalities".

Ver lista completa en [`docs/REFERENCIAS.md`](docs/REFERENCIAS.md)

---

## 🤝 Contribuciones

¡Las contribuciones son bienvenidas! Si encuentras bugs o tienes sugerencias:

1. Fork el repositorio
2. Crea una branch (`git checkout -b feature/nueva-funcionalidad`)
3. Commit cambios (`git commit -m 'Agregar nueva funcionalidad'`)
4. Push a la branch (`git push origin feature/nueva-funcionalidad`)
5. Abre un Pull Request

---

## 📧 Contacto

- **Instructor**: Maria Julieta Mateos
- **Email**: julieta8a3@gmail.com


---

## 📄 Licencia

Este proyecto está bajo la licencia MIT. Ver [`LICENSE`](LICENSE) para más detalles.

---

## 🙏 Agradecimientos

- **MONAI Consortium** por el framework
- **medigan** team por modelos pre-entrenados
- **PyTorch** community
- Datasets públicos: fastMRI, BraTS, ChestX-ray14, IXI

---

## 🔗 Enlaces Útiles

- [Documentación MONAI](https://docs.monai.io/)
- [medigan GitHub](https://github.com/RichardObi/medigan)
- [fastMRI Challenge](https://fastmri.org/)
- [Grand Challenges](https://grand-challenge.org/)
- [Papers with Code - Medical Imaging](https://paperswithcode.com/area/medical)

---

**Última actualización**: Noviembre 2025
