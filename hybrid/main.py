import base64
import json
import uuid
from db_models.mongo_setup import global_init
from db_models.models.cache_model import Cache
import init
from places-hybrid import predict
import globals


if __name__ == "__main__":
    for message in init.consumer_obj:
        global_init()
        message = message.value
        db_key = str(message)
        db_object = Cache.objects.get(pk=db_key)
        file_name = db_object.file_name
        init.redis_obj.set(globals.RECEIVE_TOPIC, file_name)


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
