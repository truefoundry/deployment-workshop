import os

import numpy as np
import tensorflow as tf
from fastapi import FastAPI
import numpy as np
import tensorflow as tf
from pydantic import BaseModel


def load_model(model_path: str) -> tf.keras.Model:
    # Load the trained model
    model = tf.keras.models.load_model(model_path)
    return model


def predict_fn(model, img_arr: np.ndarray) -> str:
    # Preprocess the image before passing it to the model
    img_arr = tf.expand_dims(img_arr, 0)
    img_arr = img_arr[:, :, :, 0]  # Keep only the first channel (grayscale)

    # Make predictions
    predictions = model.predict(img_arr)
    predicted_label = tf.argmax(predictions[0]).numpy()

    return str(predicted_label)


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
