#!/bin/bash

# Script para descargar datasets médicos públicos
# Uso: bash download_datasets.sh

echo "=========================================="
echo "Descargando Datasets Médicos Públicos"
echo "=========================================="

# Crear directorios
mkdir -p brain_mri chest_xray ct_samples

echo ""
echo "[1/4] Descargando Brain MRI samples (IXI Dataset subset)..."
echo "Este dataset contiene T1, T2, y PD-weighted MRI de cerebros sanos"

# IXI Dataset - Subset pequeño para demos
# Nota: Dataset completo disponible en https://brain-development.org/ixi-dataset/
# Aquí descargamos solo algunos ejemplos para práctica

gdown --fuzzy "https://drive.google.com/uc?id=SAMPLE_IXI_ID" -O brain_mri/
# Alternativa: wget desde servidor público si tienes uno configurado

echo ""
echo "[2/4] Descargando Chest X-ray samples..."
echo "Subset de ChestX-ray14 con ~500 imágenes balanceadas"

# Opción 1: Kaggle (requiere configurar API key en ~/.kaggle/kaggle.json)
# kaggle datasets download -d nih-chest-xrays/sample -p chest_xray/

# Opción 2: Google Drive (ejemplo)
# gdown --fuzzy "https://drive.google.com/uc?id=SAMPLE_XRAY_ID" -O chest_xray/

# Opción 3: wget desde repositorio público
# wget -O chest_xray/chest_xray_samples.zip "URL_TO_SAMPLES"

echo ""
echo "[3/4] Descargando synthetic CT samples para demos..."
echo "Ejemplos pre-generados para validación rápida"

# Ejemplos sintéticos para comparación
# gdown --fuzzy "https://drive.google.com/uc?id=SYNTHETIC_CT_ID" -O ct_samples/

echo ""
echo "[4/4] Descomprimiendo archivos..."

# Descomprimir si es necesario
if [ -f brain_mri/*.zip ]; then
    unzip -q brain_mri/*.zip -d brain_mri/
    rm brain_mri/*.zip
fi

if [ -f chest_xray/*.zip ]; then
    unzip -q chest_xray/*.zip -d chest_xray/
    rm chest_xray/*.zip
fi

if [ -f ct_samples/*.zip ]; then
    unzip -q ct_samples/*.zip -d ct_samples/
    rm ct_samples/*.zip
fi

echo ""
echo "=========================================="
echo "✅ Descarga completada!"
echo "=========================================="
echo ""
echo "Estructura de datos:"
tree -L 2 -d .

echo ""
echo "📊 Estadísticas de datasets:"
echo "Brain MRI: $(find brain_mri -name "*.nii*" | wc -l) archivos NIfTI"
echo "Chest X-ray: $(find chest_xray -name "*.png" -o -name "*.jpg" | wc -l) imágenes"
echo "CT samples: $(find ct_samples -name "*.nii*" | wc -l) archivos"

echo ""
echo "🚀 Para datasets completos, consulta:"
echo "  - fastMRI: https://fastmri.org/"
echo "  - BraTS: https://www.med.upenn.edu/cbica/brats/"
echo "  - ChestX-ray14: https://nihcc.app.box.com/v/ChestXray-NIHCC"
echo "  - IXI: https://brain-development.org/ixi-dataset/"
echo ""
echo "⚠️  IMPORTANTE: Algunos datasets requieren registro y aceptación de términos"