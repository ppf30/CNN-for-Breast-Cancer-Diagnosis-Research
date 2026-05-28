# CNN for Breast Cancer Diagnosis Research — ResNet Branch

> Investigación sobre detección de cáncer de mama mediante redes neuronales convolucionales aplicadas a mamografías digitales. Esta rama implementa una arquitectura **ResNet-18** con fine-tuning parcial y una cabeza de embeddings personalizada para clasificación binaria (benigno / maligno).

---

## Índice

- [Descripción del proyecto](../../../../../../Downloads/README.md#descripción-del-proyecto)
- [Arquitectura del modelo](../../../../../../Downloads/README.md#arquitectura-del-modelo)
- [Estructura del repositorio](../../../../../../Downloads/README.md#estructura-del-repositorio)
- [Dataset](../../../../../../Downloads/README.md#dataset)
- [Instalación](../../../../../../Downloads/README.md#instalación)
- [Uso](../../../../../../Downloads/README.md#uso)
- [Pipeline completo](../../../../../../Downloads/README.md#pipeline-completo)
- [Resultados y visualización](../../../../../../Downloads/README.md#resultados-y-visualización)
- [Configuración](../../../../../../Downloads/README.md#configuración)
- [Licencia](../../../../../../Downloads/README.md#licencia)

---

## Descripción del proyecto

Este proyecto explora el uso de redes neuronales convolucionales profundas para asistir en el diagnóstico de cáncer de mama a partir de mamografías digitales del dataset **DMID** (Digital Mammography Dataset for Breast Cancer Diagnosis Research).

La rama `ResNet` implementa una arquitectura basada en **ResNet-18 pre-entrenada** (ImageNet), adaptada para:

- Clasificación binaria: **benigno (0) / maligno (1)**
- Extracción de **embeddings de baja dimensión** (64D) por región de interés (ROI)
- Evaluación mediante **AUC-ROC** como métrica principal
- Manejo de **desbalance de clases** mediante pesos en la función de pérdida

---

## Arquitectura del modelo

`ResNet18Embedding` (`src/models/resnet.py`)

```
ResNet-18 backbone (pre-trained, ImageNet)
    └── Capas congeladas hasta layer4
    └── layer4 → fine-tuning habilitado
    └── Average Pool → [batch, 512, 1, 1]

Embedding Head:
    Flatten → [batch, 512]
    Linear(512, 256) + BatchNorm1D + ReLU + Dropout(0.3)
    Linear(256, 64)  → embedding [batch, 64]

Classifier Head:
    Dropout(0.3)
    Linear(64, 2)    → logits [batch, 2]
```

**Decisiones de diseño:**

- Fine-tuning parcial desde `layer4` para balancear adaptación y generalización
- BatchNorm en la cabeza de embedding para estabilizar el entrenamiento
- Dropout (0.3) en dos puntos para regularización
- Pesos de clase inversamente proporcionales a la frecuencia para compensar el desbalance benigno/maligno

---

## Estructura del repositorio

```
CNN-for-Breast-Cancer-Diagnosis-Research/
├── src/
│   ├── main.py                        # Punto de entrada principal (entrenamiento + embeddings)
│   ├── train.py                       # Bucle de entrenamiento con AUC y guardado del mejor modelo
│   ├── visualize_training.py          # Dashboard de métricas post-entrenamiento
│   ├── models/
│   │   └── resnet.py                  # Definición de ResNet18Embedding
│   ├── data/
│   │   └── dataset.py                 # MammographyROIDataset (PyTorch Dataset)
│   └── utils/
│       ├── extract_embeddings.py      # Extracción y agregación de embeddings por imagen
│       ├── generate_roi_splits.py     # Preprocesado: extracción de ROIs desde el dataset
│       └── roi.py                     # Función de recorte y redimensionado de ROI
├── best_resnet18_embedding.pth        # Pesos del mejor modelo (por AUC de validación)
├── training_history.pkl               # Historial de métricas para visualización
├── test_embeddings.pkl                # Embeddings extraídos del conjunto de test
├── requirements.txt
├── .env                               # PYTHONPATH configurado a ./src
└── LICENSE
```

---

## Dataset

El proyecto utiliza el dataset **DMID** (Digital Mammography Dataset for Breast Cancer Diagnosis Research).

**Descarga:** [https://figshare.com/articles/dataset/DMID/24522883](https://figshare.com/articles/dataset/b_Digital_mammography_Dataset_for_Breast_Cancer_Diagnosis_Research_DMID_b_DMID_rar/24522883)

Una vez descargado, la estructura esperada es:

```
dataset/
├── Metadata.xlsx               # Coordenadas y etiquetas de cada lesión
├── split_masks/
│   ├── train/                  # Imágenes .tif de entrenamiento
│   ├── val/                    # Imágenes .tif de validación
│   └── test/                   # Imágenes .tif de test
└── isolated_tumors/            # Generado por generate_roi_splits.py
    ├── train/
    ├── val/
    └── test/
```

El archivo `Metadata.xlsx` debe contener las columnas: `IMG_REF`, `CLASS` (B/M), `X_COORD`, `Y_COORD`, `RADIUS`.

---

## Instalación

**Requisitos previos:** Python 3.10+, pip, (opcional) CUDA para entrenamiento con GPU.

```bash
# 1. Clonar el repositorio
git clone https://github.com/cmc186/CNN-for-Breast-Cancer-Diagnosis-Research.git
cd CNN-for-Breast-Cancer-Diagnosis-Research
git checkout ResNet

# 2. Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Configurar PYTHONPATH
# Ya incluido en .env; o bien ejecutar manualmente:
export PYTHONPATH=./src          # Linux/macOS
# set PYTHONPATH=./src           # Windows
```

---

## Uso

### 1. Preprocesado: extracción de ROIs

Antes del entrenamiento, extraer las regiones de interés (ROIs) del dataset:

```bash
cd src/utils
python generate_roi_splits.py
```

Esto lee `Metadata.xlsx`, localiza cada imagen `.tif` en `split_masks/`, recorta la ROI centrada en la lesión y guarda el resultado en `dataset/isolated_tumors/{train,val,test}/`.

El nombre de cada fichero generado sigue el formato: `{IMG_REF}_{idx}_{label}.tif` donde `label` es `0` (benigno) o `1` (maligno).

### 2. Entrenamiento

```bash
cd src
python main.py
```

El script ejecuta:

- Carga de los tres splits (train / val / test)
- Inicialización de `ResNet18Embedding` con fine-tuning desde `layer4`
- 50 épocas de entrenamiento con `AdamW` + `CosineAnnealingLR`
- Guardado automático de los mejores pesos (`best_resnet18_embedding.pth`) según AUC de validación
- Extracción de embeddings del conjunto de test → `test_embeddings.pkl`
- Guardado del historial de métricas → `training_history.pkl`

**Hiperparámetros por defecto:**

| Parámetro       | Valor   |
|-----------------|---------|
| Batch size      | 32      |
| Epochs          | 50      |
| Learning rate   | 1e-4    |
| Embedding dim   | 64      |
| Optimizer       | AdamW   |
| Scheduler       | CosineAnnealingLR |
| Weight decay    | 1e-4    |

### 3. Visualización del entrenamiento

```bash
cd src
python visualize_training.py
```

Genera `training_dashboard.png` con dos gráficas: train loss por época (con media móvil de 5 épocas) y AUC de validación, marcando el mejor epoch.

---

## Pipeline completo

```
DMID dataset (.tif + Metadata.xlsx)
        │
        ▼
generate_roi_splits.py   ← extrae ROIs centradas en cada lesión
        │
        ▼
isolated_tumors/{train,val,test}/
        │
        ▼
main.py
  ├── MammographyROIDataset    ← carga, normaliza y etiqueta cada ROI
  ├── ResNet18Embedding        ← backbone ResNet-18 + cabeza embedding
  ├── train_model()            ← entrenamiento con AUC-guided model saving
  ├── training_history.pkl     ← historial para visualización
  └── extract_embeddings()     ← embeddings agregados por imagen completa
        │
        ▼
test_embeddings.pkl      ← representaciones 64D listas para análisis
        │
        ▼
visualize_training.py    ← dashboard de métricas
```

---

## Resultados y visualización

El entrenamiento genera:

- `best_resnet18_embedding.pth`: pesos del modelo con mayor AUC en validación.
- `training_history.pkl`: diccionario con claves `train_loss` y `val_auc` por época.
- `test_embeddings.pkl`: tupla `(ids, embeddings)` — array `[N, 64]` de embeddings agregados por imagen de test.
- `training_dashboard.png`: gráfica de loss y AUC generada por `visualize_training.py`.

---

## Configuración

El archivo `.env` en la raíz define:

```
PYTHONPATH=./src
```

Esto permite importar módulos como `from models.resnet import ResNet18Embedding` directamente desde cualquier script dentro de `src/`.

---

## Licencia

Distribuido bajo los términos especificados en el archivo [LICENSE](../../../../../../Downloads/LICENSE).
