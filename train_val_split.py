from pathlib import Path
from typing import Tuple
import numpy as np
import cv2
from sklearn.model_selection import train_test_split


def load_dataset(dataset_dir_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    x, y = [], []
    for i, class_dir in enumerate(sorted(dataset_dir_path.iterdir())):
        for file in class_dir.iterdir():
            # print(file)
            # img_file = cv2.imread(str(file))
            x.append(file)
            y.append(i)

    return np.asarray(x), np.asarray(y)


X, y = load_dataset(Path('data/split_dataset/train'))

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(X_train, y_train)

for idx_label, image_path in enumerate(X_train):
    img = cv2.imread(str(image_path))
    output_folder = "data/data_to_train/train/" + str("{:02d}".format(y_train[idx_label]))
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    output_path = output_folder + "/" + image_path.stem + ".jpeg"
    cv2.imwrite(output_path, img)

for idx_label, image_path in enumerate(X_val):
    img = cv2.imread(str(image_path))
    output_folder = "data/data_to_train/val/" + str("{:02d}".format(y_val[idx_label]))
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    output_path = output_folder + "/" + image_path.stem + ".jpeg"
    cv2.imwrite(output_path, img)