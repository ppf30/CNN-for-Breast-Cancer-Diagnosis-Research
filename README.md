# CNN for Breast Cancer Diagnosis

> Pipeline de segmentación con variantes UNet para la detección del cáncer de mama.

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Status](https://img.shields.io/badge/estado-investigación-orange?style=flat-square)

---

## Variantes del modelo

| # | Modelo | Descripción |
|---|--------|-------------|
| 1 | **UNet clásico** | Arquitectura encoder–decoder con skip connections simétricas |
| 2 | **Attention UNet** | Incorpora mecanismo de atención sobre las skip connections |

---

## Componentes

-  **Flujo de entrenamiento y test** — pipeline completo end-to-end
- **Generación de embeddings** — extracción de representaciones latentes
- **Implementación de modelos** — UNet clásico y Attention UNet
- **Visualizaciones** — métricas, máscaras de segmentación y análisis

---

## Estructura del repositorio

```
├── unet/
|    models/
│         ├── unet.py
│         └── attention_unet.py
├── model_related/
│   ├── train.py
│   └── embddings.py
|   └── test.py
├── visualizations/
│   └── test_visualizations
|   └── loss_curves
|   └── gt_masked
├── notebooks/
│   └── attention_unet
|   └── unet
├── general_utils/
│   └── general
|   └── metrics
|   └── splits
└── data/
    ├── emb/
    ├── masks/
    └── splits/



```

---

##  Instalación

```bash
git clone https://github.com/tu-usuario/CNN-for-Breast-Cancer-Diagnosis-Research.git
cd CNN-for-Breast-Cancer-Diagnosis-Research
pip install -r requirements.txt
```

---

##  Uso rápido

```bash
# Entrenamiento
python main_unet.py  --model attention_unet --epochs 50 --mode train

# Evaluación
python main_unet.py --modelname selected_model --mode test

# Generar embeddings
python main_unet.py --modelname selected_model --mode emb

```

---

## 📄 Referencia

Este repositorio forma parte de un trabajo de investigación sobre el uso de redes neuronales convolucionales para la detección del cáncer de mama mediante segmentación semántica.
