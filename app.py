from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import h5py
import pickle
import os

from dnn_functions import predict

train = h5py.File('datasets/train_catvnoncat.h5', 'r')
classes = np.array(train["list_classes"][:])
parameters = pickle.load(open("parameters.pkl", "rb"))

app = Flask(__name__)
app.config["IMAGE_UPLOAD"] = "static/images"


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        image = request.files["image"]
        if image.filename == "":
            print("File name is invalid")
            return redirect(request.url)
        filename = secure_filename(image.filename)
        image.save(os.path.join(app.config['IMAGE_UPLOAD'], filename))

        img = Image.open("static/images/" + filename)
        print(filename)
        img = np.array(img.resize((64, 64))).reshape(1, -1).T
        class_prediction = predict(img, parameters)
        print(class_prediction)
        catvnoncat = classes[int(np.squeeze(class_prediction))].decode('utf-8')
        print(catvnoncat)
        return render_template("index.html", uploaded_image=filename, result=catvnoncat)

    return render_template("index.html")


@app.route("/display-image/<filename>")
def display_image(filename):
    return redirect(url_for('static', filename='images/' + filename), code=301)


if __name__ == "__main__":
    app.run(debug=True)
