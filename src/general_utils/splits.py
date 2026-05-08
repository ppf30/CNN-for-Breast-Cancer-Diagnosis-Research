import numpy as np
import os
import shutil

def splits_masks(images_dir: str, val_size: float = 0.15, test_size: float = 0.15) -> tuple[list, list, list]:
    """
    Splits masks into train, validation and tests sets.

    Args:
    - images_dir(str): Directory containing the masks images
    - val_size(float): Proportion of the dataset to be used as validation set, defaults 
    to 0.15
    - test_size(float): Proportion of the dataset to be used as test set, defaults
    to 0.15

    Returns:
    - tuple[list, list, list]: Lists of file names for train, validation and test sets
    that can be used to split the TIFF Images
    """

    files = os.listdir(images_dir)
    img_ids = np.array([f for f in files])


    lim_train = int(len(img_ids) * (1 - val_size - test_size))
    lim_val = int(len(img_ids) * (1 - test_size))

    # Calculate splits indices
    train = img_ids[:lim_train]
    val = img_ids[lim_train:lim_val]
    test = img_ids[lim_val:]

    # Create directories for the splits
    split_dir = os.path.join('./dataset', "split_masks")

    train_dir = os.path.join(split_dir, "train")
    val_dir = os.path.join(split_dir, "val")
    test_dir = os.path.join(split_dir, "test")

    try:
        os.makedirs(train_dir)
        os.makedirs(val_dir)
        os.makedirs(test_dir)

        # Copy files to the respective folders
        for file in train:
            src = os.path.join(images_dir, file)
            dst = os.path.join(train_dir, file)
            shutil.copy(src, dst)

        for file in val:
            src = os.path.join(images_dir, file)
            dst = os.path.join(val_dir, file)
            shutil.copy(src, dst)

        for file in test:
            src = os.path.join(images_dir, file)
            dst = os.path.join(test_dir, file)
            shutil.copy(src, dst)

    except FileExistsError:
        return train, val, test, True

    else:
        return train, val, test, False



def splits_tiff(base_dir:str, images_dir:str,  train:list, val:list, test:list):
    """
    Synchronizes TIFF image splits based on the existing partition of the masks

    Args:
    - base_dir(str): The base dir for images
    - images_dir(str): Directory containing the TIFF images
    - train(list): List of file names for the training set from masks split
    - val(list): List of file names for the validation set from masks split
    - test(list): List of file names for the test set from masks split

    """
    # Create directories for the splits
    split_dir = os.path.join(base_dir, "split_tiff")
    train_dir = os.path.join(split_dir, "train")
    val_dir = os.path.join(split_dir, "val")
    test_dir = os.path.join(split_dir, "test")


    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    
    # Copy files to the respective folders
    for file in train:
        src = os.path.join(images_dir, file)
        dst = os.path.join(train_dir, file)
        shutil.copy(src, dst)

    for file in val:
        src = os.path.join(images_dir, file)
        dst = os.path.join(val_dir, file)
        shutil.copy(src, dst)

    for file in test:
        src = os.path.join(images_dir, file)
        dst = os.path.join(test_dir, file)
        shutil.copy(src, dst)