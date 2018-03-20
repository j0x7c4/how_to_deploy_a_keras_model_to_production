from flask import Flask, request, jsonify
import os
from urllib import unquote_plus
from PIL import Image
import numpy as np
import io
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import preprocess_input
from model.load_mobilenet_nima import *
import urllib, cStringIO
import json
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
    image = preprocess_input(image, mode='tf')

    # return the processed image
    return image


@app.route('/')
def index():
    print "hello"
    return jsonify({"name": "lilyBot"})


@app.route('/predict', methods=['POST', 'GET'])
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
        # print "*****",request.data
        body = json.loads(request.data)
        img_url = body.get("img")
    elif request.method == "GET":
        # print "******", request.args
        img_url = unquote_plus(request.args.get('img'))
    if img_url:
        # read the image in PIL format
        image = Image.open(io.BytesIO(urllib.urlopen(img_url).read()))
        image = prepare_image(image, target=(224, 224))
        with graph.as_default():
            preds = model.predict(image, batch_size=1, verbose=0)[0]
            mean = mean_score(preds)
            std = std_score(preds)
        data["success"] = True
        data["pred"] = {"mean": mean, "std": std}
    return jsonify(data)


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    # app.run(debug=True)
    app.run(host='0.0.0.0', port=port)