import cv2
import json
import faiss
import random
import numpy as np
import pyarrow.parquet as pq

from pathlib import Path
from efficientnet.tfkeras import EfficientNetB0
import tensorflow as tf
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
    model = tf.keras.models.load_model('models/final_model.hdf5')
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
    # CPU
    # index = faiss.IndexFlatIP(d)
    index = faiss.IndexFlatL2(d)
    # index = faiss.IndexLSH(d, 2*d)
    # index = faiss.IndexHNSWFlat(d, 32)

    # GPU
    # res = faiss.StandardGpuResources()
    # flat_config = faiss.GpuIndexFlatConfig()
    # flat_config.device = 0
    # index = faiss.GpuIndexFlatL2(res, d, flat_config)
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


def search(index, id_to_name, emb, k=6):
    D, I = index.search(np.expand_dims(emb, 0), k)  # actual search
    return dict(zip([id_to_name[x] for x in I[0]], D[0]))


def display_picture(image_path, image_name):
    # display(Image(filename=f"{image_path}/{image_name}.jpeg"))
    # img = cv2.imread(f"{image_path}/{image_name}.jpeg")
    img = cv2.imread(image_path)
    cv2.imshow(f"Query: {image_name}", img)
    cv2.waitKey()


def display_results(q_path, res_path, results):
    cv2.namedWindow('Test Results', )
    q_img = cv2.imread(str(q_path))

    images = []
    distances = ['Query']
    sections = [str(q_path.stem)[0:2]]
    for image_name, distance in results.items():
        img = cv2.imread(f"{res_path}/{str(image_name)[0:2]}/{image_name}.jpeg")
        images.append(img)
        sections.append(str(image_name)[0:2])
        distances.append(distance)
    imgs = np.concatenate(images, axis=1)
    f_imgs = np.concatenate((q_img, imgs), axis=1)
    b, g, r = cv2.split(f_imgs)
    black = np.zeros((30, f_imgs.shape[1], 3), dtype = b.dtype)
    f_imgs = np.concatenate((f_imgs, black), axis=0)

    font = cv2.FONT_HERSHEY_SIMPLEX
    x = 25
    y = 210
    fontScale = 1
    color_default = (255, 255, 255)
    color_green = (0, 255, 0)
    color_red = (0, 0, 255)

    lineType = 1
    f_imgs = cv2.resize(f_imgs, (0, 0), fx=1.5, fy=1.5)
    for i, string in enumerate(distances):
        color = color_default
        if str(sections[0]) == str(sections[i]) and i > 0: color = color_green
        elif str(sections[0]) != str(sections[i]) and i > 0: color = color_red

        if string != 'Query': string = np.round(string, 2)
        text = str('[' + str(sections[i]) + ']: ' + str(string))
        f_imgs = cv2.putText(f_imgs, text,
                             (x, y),
                             font,
                             fontScale,
                             color,
                             lineType)
        x += 241

    cv2.imshow('Test Results', f_imgs)
    # cv2.imwrite(f'film/img{str(q_path.stem)[-2:]}.png', f_imgs)
    cv2.waitKey()
