from knn import *
from inference import *

import cv2
import time

from pathlib import Path


path_images = "data/split_dataset/train"
path_tfrecords = "data/tfrecords"
path_embeddings = "data/embeddings"

inference.write_tfrecord(image_folder=path_images, output_folder=path_tfrecords, num_shards=10)
inference.run_inference(tfrecords_folder=path_tfrecords, output_folder=path_embeddings, batch_size=32)

inputPath = Path("data/split_dataset/test")
inputFiles = inputPath.glob("**/*.jpeg")

acc = 0
n_acc = 0
error = 0
start0 = time.time()

for path_to_q_img in inputFiles:
    n_acc += 1
    s_path_to_q_img = str(path_to_q_img)
    image = cv2.imread(s_path_to_q_img)

    start = time.time()

    [id_to_name, name_to_id, embeddings] = knn.add_to_embeddings_no_tf(image, path_embeddings)

    # print(embeddings)
    index = knn.build_index(embeddings)

    p = len(embeddings) - 1
    # print('Query: ' + id_to_name[p])
    # knn.display_picture(str(s_path_to_q_img), id_to_name[p])
    results = knn.search(index, id_to_name, embeddings[p])
    if str(results[0][1]) != "query":
        top_knn = str(results[0][1])[0:2]
    else:
        top_knn = str(results[1][1])[0:2]
    # print("top knn: ", top_knn)
    # knn.display_results(path_images, results)
    # print(path_to_q_img)
    # print(str(path_to_q_img)[24:26])

    end = time.time()
    print(f"Time for {str(path_to_q_img)}: {end - start}")
    q = str(path_to_q_img)[24:26]
    p = top_knn
    if q == p:
        acc += 1
print("Accuracy: ", acc)
print("Total Accuracy: ", acc / n_acc)
end0 = time.time()
print(f"TOTAL TIME: {end0 - start0}")

with open('logs/emb_test3.txt', 'a') as file:
    file.write(str("Accuracy: " + str(acc) + ", final acc: " + str(acc / n_acc) + '\n'))
