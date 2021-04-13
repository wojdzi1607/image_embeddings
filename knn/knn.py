import cv2
import json
import faiss
import random
import numpy as np
import tensorflow as tf
import pyarrow.parquet as pq

from pathlib import Path
from efficientnet.tfkeras import EfficientNetB0
from image_embeddings.inference.inference import read_tfrecord

from dataclasses import dataclass
from IPython.display import Image, display
from ipywidgets import widgets, HBox, VBox


def read_embeddings(path):
    emb = pq.read_table(path).to_pandas()
    id_to_name = {k: v.decode("utf-8") for k, v in enumerate(list(emb["image_name"]))}
    name_to_id = {v: k for k, v in id_to_name.items()}
    embgood = np.stack(emb["embedding"].to_numpy())
    return [id_to_name, name_to_id, embgood]


def embeddings_to_numpy(input_path, output_path):
    emb = pq.read_table(input_path).to_pandas()
    Path(output_path).mkdir(parents=True, exist_ok=True)
    id_name = [{"id": k, "name": v.decode("utf-8")} for k, v in enumerate(list(emb["image_name"]))]
    json.dump(id_name, open(output_path + "/id_name.json", "w"))
    emb = np.stack(emb["embedding"].to_numpy())
    np.save(open(output_path + "/embedding.npy", "wb"), emb)


def add_to_embeddings_no_tf(image, embeddings_path):
    emb = pq.read_table(embeddings_path).to_pandas()
    # model = EfficientNetB0(weights="imagenet", include_top=False, pooling="avg")
    model = tf.keras.models.load_model('models/no_top_model.hdf5')
    image = image[None, :, :, :] * (1 / 255)
    pred = model.predict(image, verbose=1)
    emb = emb.append({'image_name': b'query', 'embedding': np.array(pred[0])}, ignore_index=True)
    id_to_name = {k: v.decode("utf-8") for k, v in enumerate(list(emb["image_name"]))}
    name_to_id = {v: k for k, v in id_to_name.items()}
    embgood = np.stack(emb["embedding"].to_numpy())
    return [id_to_name, name_to_id, embgood]


def build_index(emb):
    d = emb.shape[1]
    xb = emb
    index = faiss.IndexFlatIP(d)
    # index = faiss.IndexLSH(d, 8)
    index.add(xb)
    return index


def random_search(path):
    [id_to_name, name_to_id, embeddings] = read_embeddings(path)
    index = build_index(embeddings)
    p = random.randint(0, len(id_to_name) - 1)
    print(id_to_name[p])
    results = search(index, id_to_name, embeddings[p])
    for e in results:
        print(f"{e[0]:.2f} {e[1]}")


def search(index, id_to_name, emb, k=5):
    D, I = index.search(np.expand_dims(emb, 0), k)  # actual search
    return list(zip(D[0], [id_to_name[x] for x in I[0]]))


def display_picture(image_path, image_name):
    # display(Image(filename=f"{image_path}/{image_name}.jpeg"))
    # img = cv2.imread(f"{image_path}/{image_name}.jpeg")
    img = cv2.imread(image_path)
    cv2.imshow(f"Query: {image_name}", img)
    cv2.waitKey()


def display_results(image_path, results):

    # hbox = HBox(
    #     [
    #         VBox(
    #             [
    #                 widgets.Label(f"{distance:.2f} {image_name}"),
    #                 widgets.Image(value=open(f"{image_path}/{image_name}.jpeg", "rb").read()),
    #             ]
    #         )
    #         for distance, image_name in results
    #     ]
    # )
    # display(hbox)

    for distance, image_name in results:
        if image_name != "query":
            img = cv2.imread(f"{image_path}/{image_name}.jpeg")
            cv2.imshow(f"{distance:.2f} {image_name}", img)
    cv2.waitKey()
