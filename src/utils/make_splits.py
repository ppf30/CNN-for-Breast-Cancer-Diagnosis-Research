from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import defaultdict

def make_splits(images_dir: str, val_size=0.15, test_size=0.15, seed=42):
    """
    Divide un conjunto de ROIs (.tif) en train/val/test evitando fuga de información.

    El split se realiza a nivel de imagen/paciente (IMGxxx), no a nivel de ROI,
    de modo que todos los tumores de una misma imagen quedan en la misma partición.
    La estratificación se hace usando una etiqueta por imagen: 1 si algún tumor
    es maligno, 0 en caso contrario.

    Args:
        images_dir (str): Carpeta con los archivos .tif.
        val_size (float): Proporción para validación.
        test_size (float): Proporción para test.
        seed (int): Semilla para reproducibilidad.

    Returns:
        train_files, val_files, test_files (list[Path]): Listas de rutas a ROIs.
    """
    all_files = list(Path(images_dir).glob("*.tif"))

    # agrupar ROIs por imagen (IMGxxx)
    groups = defaultdict(list)  # diccionario: clave->id img | valor-> lista de tumores pertenecientes a la img
    for f in all_files:
        img_id = Path(f).stem.split('_')[0] #nos quedamos con el nombre del archivo previo al primer _ (el id de la imagen)
        groups[img_id].append(f)

    img_ids = list(groups.keys())

    # etiqueta por imagen: si algún ROI es maligno → 1
    img_labels = []
    for img_id in img_ids:
        labs = [int(Path(f).stem[-1]) for f in groups[img_id]]
        img_labels.append(int(max(labs)))

    # split por imagen
    train_val_ids, test_ids, y_train_val, _ = train_test_split(
        img_ids, img_labels,
        test_size=test_size, stratify=img_labels, random_state=seed
    )

    val_size_adj = val_size / (1 - test_size)

    train_ids, val_ids = train_test_split(
        train_val_ids,
        test_size=val_size_adj,
        stratify=y_train_val,
        random_state=seed
    )

    # expandir a ROIs
    train_files = [f for i in train_ids for f in groups[i]]
    val_files   = [f for i in val_ids   for f in groups[i]]
    test_files  = [f for i in test_ids  for f in groups[i]]

    print(f"ROIs -> Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}")
    print(f"Imgs -> Train: {len(train_ids)} | Val: {len(val_ids)} | Test: {len(test_ids)}")

    return train_files, val_files, test_files