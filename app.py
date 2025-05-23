from flask import Flask, request, render_template
import numpy as np
import tensorflow as tf
import os
from PIL import Image
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename

IMG_HEIGHT, IMG_WIDTH = 256, 256
UPLOAD_FOLDER = 'static/uploads'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load your trained model
model = tf.keras.models.load_model("model_for_lane.keras", compile=False)

def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict_mask(image_path):
    image = preprocess_image(image_path)
    pred_mask = model.predict(image)[0]
    pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255
    pred_mask = np.squeeze(pred_mask)

    # Save the mask using matplotlib
    mask_img_path = os.path.join(app.config['UPLOAD_FOLDER'], f"mask_{os.path.basename(image_path)}")
    plt.imsave(mask_img_path, pred_mask, cmap='gray')
    return mask_img_path

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            mask_path = predict_mask(file_path)

            return render_template('index.html', uploaded=True, img_path=file_path, mask_path=mask_path)
    return render_template('index.html', uploaded=False)

if __name__ == '__main__':
    app.run(debug=True)
