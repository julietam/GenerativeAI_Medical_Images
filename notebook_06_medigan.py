# Demo Rápida: Generación de Imágenes Médicas con medigan
# Notebook 06: Modelos Pre-entrenados

"""
Este notebook demuestra cómo usar medigan para generar imágenes médicas
sintéticas sin necesidad de entrenar modelos desde cero.

Tiempo estimado: 15 minutos
"""

# ============================================================================
# 1. INSTALACIÓN Y SETUP
# ============================================================================

# Instalar medigan (descomentar si es necesario)
# !pip install medigan

import medigan
from medigan import Generators
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display, HTML

print(f"medigan version: {medigan.__version__}")

# ============================================================================
# 2. EXPLORAR MODELOS DISPONIBLES
# ============================================================================

# Inicializar generadores
generators = Generators()

print("=" * 70)
print("MODELOS PRE-ENTRENADOS DISPONIBLES EN MEDIGAN")
print("=" * 70)

# Listar todos los modelos
all_models = generators.list_models()
print(f"\n📊 Total de modelos disponibles: {len(all_models)}\n")

# Mostrar primeros 5 modelos como ejemplo
for i, model_id in enumerate(list(all_models.keys())[:5]):
    model_info = all_models[model_id]
    print(f"{i+1}. Model ID: {model_id}")
    print(f"   Modalidad: {model_info.get('modality', 'N/A')}")
    print(f"   Tipo: {model_info.get('model_type', 'N/A')}")
    print(f"   Descripción: {model_info.get('description', 'N/A')[:100]}...")
    print()

# ============================================================================
# 3. BUSCAR MODELOS POR MODALIDAD
# ============================================================================

print("=" * 70)
print("BUSCAR MODELOS POR MODALIDAD")
print("=" * 70)

# Buscar modelos de mamografía
mammography_models = generators.search_models("mammography")
print(f"\n🔍 Modelos de Mamografía encontrados: {len(mammography_models)}")
for model_id in mammography_models:
    print(f"  - {model_id}")

# Buscar modelos de rayos X
xray_models = generators.search_models("xray")
print(f"\n🔍 Modelos de Chest X-ray encontrados: {len(xray_models)}")
for model_id in xray_models:
    print(f"  - {model_id}")

# ============================================================================
# 4. GENERAR MAMOGRAFÍAS CON MASAS (C-DCGAN)
# ============================================================================

print("\n" + "=" * 70)
print("GENERANDO MAMOGRAFÍAS CON MASAS")
print("=" * 70)

# Model ID para mamografías con masas
model_id_mammo = "00008_C-DCGAN_MMG_MASSES"

print(f"\nUsando modelo: {model_id_mammo}")
print("Generando 8 mamografías sintéticas...")

# Generar imágenes
try:
    # Generate images
    mammo_images = generators.generate(
        model_id=model_id_mammo,
        num_samples=8,
        output_path="./results/mammography/",
        save_images=True
    )
    
    # Visualizar
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Mamografías Sintéticas (C-DCGAN)', fontsize=16, fontweight='bold')
    
    for i, ax in enumerate(axes.flat):
        if i < len(mammo_images):
            ax.imshow(mammo_images[i], cmap='gray')
            ax.set_title(f'Muestra {i+1}')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('./results/mammography_grid.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✅ Generación exitosa!")
    print(f"📁 Imágenes guardadas en: ./results/mammography/")

except Exception as e:
    print(f"❌ Error: {e}")
    print("Asegúrate de tener instaladas todas las dependencias del modelo")

# ============================================================================
# 5. GENERAR CHEST X-RAYS (DCGAN)
# ============================================================================

print("\n" + "=" * 70)
print("GENERANDO CHEST X-RAYS")
print("=" * 70)

# Model ID para chest X-rays
model_id_xray = "00009_DCGAN_XRAY_LUNG_NODULES"

print(f"\nUsando modelo: {model_id_xray}")
print("Generando 8 chest X-rays con nódulos pulmonares...")

try:
    xray_images = generators.generate(
        model_id=model_id_xray,
        num_samples=8,
        output_path="./results/chest_xray/",
        save_images=True
    )
    
    # Visualizar
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Chest X-rays Sintéticos (DCGAN)', fontsize=16, fontweight='bold')
    
    for i, ax in enumerate(axes.flat):
        if i < len(xray_images):
            ax.imshow(xray_images[i], cmap='gray')
            ax.set_title(f'Muestra {i+1}')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('./results/xray_grid.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✅ Generación exitosa!")
    print(f"📁 Imágenes guardadas en: ./results/chest_xray/")

except Exception as e:
    print(f"❌ Error: {e}")

# ============================================================================
# 6. TRADUCCIÓN DE DENSIDAD MAMARIA (CycleGAN)
# ============================================================================

print("\n" + "=" * 70)
print("TRADUCCIÓN DE DENSIDAD MAMARIA")
print("=" * 70)

# Model ID para traducción de densidad
model_id_density = "00003_CYCLEGAN_MMG_DENSITY_FULL"

print(f"\nUsando modelo: {model_id_density}")
print("Generando traducción: Baja densidad → Alta densidad...")

try:
    density_images = generators.generate(
        model_id=model_id_density,
        num_samples=4,
        output_path="./results/density_translation/",
        save_images=True
    )
    
    # Visualizar comparación
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle('Traducción de Densidad Mamaria (CycleGAN)', 
                 fontsize=16, fontweight='bold')
    
    for i, ax in enumerate(axes.flat):
        if i < len(density_images):
            ax.imshow(density_images[i], cmap='gray')
            ax.set_title(f'Traducción {i+1}')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('./results/density_translation_grid.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✅ Generación exitosa!")
    print(f"📁 Imágenes guardadas en: ./results/density_translation/")

except Exception as e:
    print(f"❌ Error: {e}")

# ============================================================================
# 7. EXPLORACIÓN DEL ESPACIO LATENTE
# ============================================================================

print("\n" + "=" * 70)
print("EXPLORACIÓN DEL ESPACIO LATENTE")
print("=" * 70)

"""
Algunos modelos permiten controlar el espacio latente para generar
variaciones controladas de las imágenes.
"""

print("\nGenerando interpolación en espacio latente...")
print("(Esto crea una transición suave entre dos imágenes sintéticas)")

# Ejemplo conceptual de latent space walk
# En medigan, algunos modelos exponen el generador directamente

try:
    # Generar dos puntos aleatorios en espacio latente
    z1 = np.random.randn(1, 100)  # Latent vector 1
    z2 = np.random.randn(1, 100)  # Latent vector 2
    
    # Interpolar entre z1 y z2
    steps = 8
    latent_interp = []
    for alpha in np.linspace(0, 1, steps):
        z_interp = (1 - alpha) * z1 + alpha * z2
        latent_interp.append(z_interp)
    
    print(f"✅ Interpolación creada con {steps} pasos")
    print("Nota: Para visualizar, necesitas acceso directo al generador del modelo")
    
except Exception as e:
    print(f"⚠️  La interpolación requiere acceso al generador: {e}")

# ============================================================================
# 8. MÉTRICAS DE CALIDAD (Conceptual)
# ============================================================================

print("\n" + "=" * 70)
print("EVALUACIÓN DE CALIDAD")
print("=" * 70)

print("""
Para evaluar la calidad de imágenes sintéticas, típicamente se usan:

1. **FID (Fréchet Inception Distance)**
   - Mide distancia entre distribuciones real vs. sintética
   - Menor es mejor (FID < 50 es bueno, FID < 20 es excelente)
   
2. **IS (Inception Score)**
   - Evalúa diversidad y calidad
   - Mayor es mejor (IS > 3 es típico para medical images)
   
3. **SSIM (Structural Similarity)**
   - Para reconstrucción o traducción
   - Rango [0, 1], cercano a 1 es mejor
   
4. **Evaluación por expertos**
   - Radiólogos clasifican real vs. sintético
   - Idealmente: 50% accuracy (indistinguible)

En notebooks posteriores implementaremos estas métricas.
""")

# ============================================================================
# 9. RESUMEN Y PRÓXIMOS PASOS
# ============================================================================

print("\n" + "=" * 70)
print("RESUMEN DEL DEMO")
print("=" * 70)

print("""
✅ Has aprendido a:
   - Instalar y usar medigan
   - Listar modelos pre-entrenados disponibles
   - Generar mamografías sintéticas (C-DCGAN)
   - Generar chest X-rays (DCGAN)
   - Realizar traducción de imágenes (CycleGAN)
   - Explorar el espacio latente (conceptual)

📚 Próximos pasos:
   1. Notebook 02: Implementar DCGAN desde cero
   2. Notebook 03: Entrenar Pix2Pix para MRI T1→T2
   3. Notebook 04: Implementar CycleGAN para MRI↔CT
   4. Notebook 05: Entrenar Diffusion Models

🔗 Recursos:
   - medigan docs: https://github.com/RichardObi/medigan
   - MONAI: https://monai.io/
   - Papers: Ver docs/REFERENCIAS.md
""")

print("\n" + "=" * 70)
print("FIN DEL DEMO")
print("=" * 70)