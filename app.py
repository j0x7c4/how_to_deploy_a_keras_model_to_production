from flask import Flask, request, jsonify
import os

from PIL import Image
import numpy as np
import io
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from model.load_mobilenet_nima import *

# initialize our flask app
app = Flask(__name__)
# global vars for easy reusability
global model, graph
# initialize these variables
model, graph = init()


def prepare_image(image, target):
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize the input image and preprocess it
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    # return the processed image
    return image


@app.route('/')
def index():
    return jsonify({"name": "lilyBot"})


@app.route('/predict', methods=['POST'])
def predict():
    # calculate mean score for AVA dataset
    def mean_score(scores):
        si = np.arange(1, 11, 1)
        mean = np.sum(scores * si)
        return mean

    # calculate standard deviation of scores for AVA dataset
    def std_score(scores):
        si = np.arange(1, 11, 1)
        mean = mean_score(scores)
        std = np.sqrt(np.sum(((si - mean) ** 2) * scores))
        return std

    data = {"success": False}
    # ensure an image was properly uploaded to our endpoint
    if request.method == "POST":
        if request.files.get("image"):
            # read the image in PIL format
            image = request.files["image"].read()
            image = Image.open(io.BytesIO(image))
            image = prepare_image(image, target=(224, 224))
            with graph.as_default():
                preds = model.predict(image)
                mean = mean_score(preds)
                std = std_score(preds)
            data["success"] = True
            data["pred"] = {"mean": mean, "std": std}
    return jsonify(data)


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    # app.run(debug=True)
    app.run(host='0.0.0.0', port=port)