from urllib.request import urlopen
import numpy as np
from PIL import Image
from cv2 import resize
from vgg16_places_365 import VGG16_Places365
import redis
from kafka import KafkaConsumer
from kafka import KafkaProducer
from json import loads
import base64
import os
import uuid
import json
from dotenv import load_dotenv
from base64 import decodestring

load_dotenv()
KAFKA_HOSTNAME = os.getenv("KAFKA_HOSTNAME")
KAFKA_PORT = os.getenv("KAFKA_PORT")
REDIS_HOSTNAME = os.getenv("REDIS_HOSTNAME")
REDIS_PORT = os.getenv("REDIS_PORT")
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
# kafka prerequisites
RECEIVE_TOPIC = 'KERAS_BASE'
SEND_TOPIC_FULL = "IMAGE_RESULTS"
SEND_TOPIC_TEXT = "TEXT"
print("kafka : " + KAFKA_HOSTNAME + ':' + KAFKA_PORT)

LABELS_URL = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
LABELS = np.array(urlopen(LABELS_URL).read().splitlines())
model = VGG16_Places365()
# Redis initialize
r = redis.StrictRedis(host=REDIS_HOSTNAME, port=REDIS_PORT,
                      password=REDIS_PASSWORD, ssl=True)
consumer_easyocr = KafkaConsumer(
    RECEIVE_TOPIC,
    bootstrap_servers=[KAFKA_HOSTNAME + ':' + KAFKA_PORT],
    auto_offset_reset="earliest",
    enable_auto_commit=True,
    group_id="my-group",
    value_deserializer=lambda x: loads(x.decode("utf-8")),
)
producer = KafkaProducer(
    bootstrap_servers=[KAFKA_HOSTNAME + ':' + KAFKA_PORT],
    value_serializer=lambda x: json.dumps(x).encode("utf-8"),
)


def predict(file_name,image_id):
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
        array_with_id = decoded_string_array.split(" ")
        array_with_id.pop()
        new_labels.append(array_with_id[0])

    scores = [float(np_float) for np_float in scores]
    response_dict = {
            "labels": new_labels,
            "scores": scores,
            "image_id": image_id,

    }
    return response_dict

if __name__ == "__main__":
    print("shit jere")
    for message in consumer_easyocr:
        print('xxx--- inside open images consumer---xxx')
        print(KAFKA_HOSTNAME + ':' + KAFKA_PORT)

        message = message.value
        print("MESSAGE RECEIVED consumer_densecap: ")
        image_id = message['image_id']
        # data = message['data']
        data = message['data']
        r.set(RECEIVE_TOPIC, image_id)
        file_name = str(uuid.uuid4()) + ".jpg"
        with open(file_name, "wb") as fh:
            fh.write(base64.b64decode(data.encode("ascii")))

        full_res = predict(file_name, image_id)
        text_res = {
            "image_id": full_res["image_id"],
            "captions": full_res["labels"]
        }
        producer.send(SEND_TOPIC_FULL, value=json.dumps(full_res))
        producer.send(SEND_TOPIC_TEXT, value=json.dumps(text_res))

        producer.flush()
