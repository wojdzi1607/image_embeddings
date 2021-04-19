from knn import *
from inference import *

import cv2
import time
import random

from pathlib import Path

random.seed(42)

path_images = "data/split_dataset/train"
path_tfrecords = "data/tfrecords"
path_embeddings = "data/embeddings"

# inference.write_tfrecord(image_folder=path_images, output_folder=path_tfrecords, num_shards=10)
# inference.run_inference(tfrecords_folder=path_tfrecords, output_folder=path_embeddings, batch_size=32)

inputPath = Path("data/split_dataset/test")
inputFiles = inputPath.glob("**/*.jpeg")

acc = 0
n_acc = 0
error = 0
start0 = time.time()

paths = []
for path_to_q_img in inputFiles:
    paths.append(path_to_q_img)
random.shuffle(paths)

for path_to_q_img in paths:
    print(f"query test: {path_to_q_img.stem}")
    n_acc += 1
    s_path_to_q_img = str(path_to_q_img)
    image = cv2.imread(s_path_to_q_img)

    start = time.time()

    [id_to_name, name_to_id, embeddings] = knn.add_to_embeddings_no_tf(image, path_embeddings)

    # Build index
    index = knn.build_index(embeddings)
    p = len(embeddings) - 1
    results = knn.search(index, id_to_name, embeddings[p])
    x = results.pop('query', None)
    if x is None: results.pop(list(results)[-1], None)

    top_knn = next(iter(results))

    end = time.time()
    print(f"Time for {str(path_to_q_img)}: {end - start}")

    # Display results
    knn.display_results(path_to_q_img, path_images, results)

    # Calculate accuracy
    q = str(path_to_q_img.stem)[0:2]
    p = str(top_knn)[0:2]
    if q == p:
        acc += 1
print("Accuracy: ", acc)
print("Total Accuracy: ", acc / n_acc)
end0 = time.time()
print(f"TOTAL TIME: {end0 - start0}")

with open('logs/emb_test10.txt', 'a') as file:
    file.write(str("Accuracy: " + str(acc) + ", final acc: " + str(acc / n_acc) + ", time: " + str({end0 - start0}) + '\n'))
