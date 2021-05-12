import cv2
import numpy as np

from pathlib import Path
from typing import Tuple
from sklearn.model_selection import train_test_split


def load_dataset(dataset_dir_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    x, y = [], []
    ihh = 0
    for i, class_dir in enumerate(sorted(dataset_dir_path.iterdir())):
        for file in sorted(class_dir.iterdir()):
            _, idx = str(file).split("_")
            x.append(file)
            y.append(int(idx[:-4])+ihh)
        ihh += 1000

    return np.asarray(x), np.asarray(y)


X, y = load_dataset(Path('data/NewSet'))
# print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(len(X_train))
print(len(X_test))
print()

for idx_label, image_path in enumerate(X_train):
    img = cv2.imread(str(image_path))
    if img is None:
        print(idx_label)
    output_folder = "data/split_dataset/split_dataset_sequential/train/" + str("{:04d}".format(y_train[idx_label]))
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    output_path = output_folder + "/" + str("{:04d}".format(y_train[idx_label])) + image_path.stem + ".jpeg"
    cv2.imwrite(output_path, img)

for idx_label, image_path in enumerate(X_test):
    img = cv2.imread(str(image_path))
    output_folder = "data/split_dataset/split_dataset_sequential/test/" + str("{:04d}".format(y_test[idx_label]))
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    output_path = output_folder + "/" + str("{:04d}".format(y_test[idx_label])) + image_path.stem + ".jpeg"
    cv2.imwrite(output_path, img)
