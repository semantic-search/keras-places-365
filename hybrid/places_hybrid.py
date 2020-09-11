import os
from urllib.request import urlopen
import numpy as np
from PIL import Image
from cv2 import resize
from vgg16_hybrid_places_1365 import VGG16_Hybrid_1365

LABELS_URL = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_hybrid1365.txt'

LABELS = np.array(urlopen(LABELS_URL).read().splitlines())
model = VGG16_Hybrid_1365()
# Redis initialize


def predict(file_name, doc=False):
    image = Image.open(file_name)
    image = np.array(image, dtype=np.uint8)
    image = resize(image, (224, 224))
    # image = preprocess_input(image.astype(np.float32))
    image = np.expand_dims(image, 0)
    output = model.predict(image)
    output = np.squeeze(output)
    new_labels = []
    top5 = output.argsort()[-5:][::-1]
    labels = LABELS[top5]
    scores = output[top5]
    for vals in labels:
        decoded_string_array = vals.decode('UTF-8')
        string_array_without_quotes = decoded_string_array.replace(',', '')
        array_without_quotes = string_array_without_quotes.split(" ")
        array_without_quotes.pop()
        array_without_quotes.pop(0)
        new_labels.append(array_without_quotes)

    scores = [float(np_float) for np_float in scores]

    if doc:
        response_dict = {
            "objects": new_labels,
            "score": scores
        }
        os.remove(file_name)
        return response_dict
    else:
        response_dict = {
            "file_name": file_name,
            "objects": new_labels,
            "score": scores,
            "is_doc_type": False
        }
        os.remove(file_name)
        return response_dict
