from flask import Flask, render_template, request, redirect
import os
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Set paths
UPLOAD_FOLDER = 'data'
MODEL_PATH = 'models/signature_model.keras'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Ensure required folders exist
os.makedirs(os.path.join(UPLOAD_FOLDER, 'genuine'), exist_ok=True)
os.makedirs(os.path.join(UPLOAD_FOLDER, 'forged'), exist_ok=True)
os.makedirs('models', exist_ok=True)

# Load or initialize the model
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
else:
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload_and_train', methods=['POST'])
def upload_and_train():
    if 'file' not in request.files or 'signature_type' not in request.form:
        return redirect(request.url)

    file = request.files['file']
    signature_type = request.form['signature_type']

    if file and allowed_file(file.filename):
        # Save the uploaded file in the respective folder
        label_folder = os.path.join(UPLOAD_FOLDER, signature_type)
        filepath = os.path.join(label_folder, file.filename)
        file.save(filepath)

        # Retrain the model
        train_model()

        return render_template('index.html', train_result="Model retrained successfully!")

    return redirect(request.url)

@app.route('/verify_signature', methods=['POST'])
def verify_signature():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file and allowed_file(file.filename):
        # Save the uploaded file
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        # Preprocess the image
        img = image.load_img(filepath, target_size=(150, 150))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Predict
        prediction = model.predict(img_array)
        result = "Genuine Signature" if prediction[0][0] >= 0.4 else "Forged Signature"

        return render_template('index.html', result=result)

    return redirect(request.url)

def train_model():
    # Create data generators for training
    train_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(
        UPLOAD_FOLDER,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary'
    )

    # Train the model
    model.fit(train_generator, epochs=50)

    # Save the updated model
    model.save(MODEL_PATH)

if __name__ == '__main__':
    app.run(debug=True)
