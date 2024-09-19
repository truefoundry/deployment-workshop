import os

import numpy as np
import tensorflow as tf
from fastapi import FastAPI
from predict import load_model, predict_fn
from pydantic import BaseModel

model_path = os.path.join(os.environ.get("MODEL_DOWNLOAD_PATH", "."), "mnist_model.h5")
model = load_model(model_path)


class ImageUrl(BaseModel):
    url: str = "https://github.com/truefoundry/deployment-workshop/blob/main/mnist-classifaction/deploy_model/sample_images/1.jpg?raw=true"


def load_image(img_url: str) -> np.ndarray:
    # generate random name for the image
    rand_num = np.random.randint(10000)
    img_path = tf.keras.utils.get_file(f"image{rand_num}.jpg", img_url)
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(28, 28))
    img_arr = tf.keras.preprocessing.image.img_to_array(img)
    return img_arr


app = FastAPI(docs_url="/", root_path=os.getenv("TFY_SERVICE_ROOT_PATH"))


@app.post("/predict")
async def predict(body: ImageUrl):
    img_arr = load_image(body.url)
    prediction = predict_fn(model, img_arr)
    return {"prediction": prediction}
