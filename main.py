# app.py
from flask import Flask, render_template, request, send_file
from flask_uploads import UploadSet, configure_uploads, IMAGES
from PIL import Image
import numpy as np
import tensorflow as tf
import subprocess

app = Flask(__name__)

# Configuração para upload de arquivos
photos = UploadSet("photos", IMAGES)
app.config["UPLOADED_PHOTOS_DEST"] = "uploads"
configure_uploads(app, photos)

# Carregar modelo GAN treinado (substitua pelo seu modelo treinado)
generator = tf.keras.models.load_model('seu_modelo_gan.h5')

def install_dependencies():
    try:
        subprocess.check_call(['pip', 'install', '-r', 'requirements.txt'])
        print("Dependências instaladas com sucesso.")
    except subprocess.CalledProcessError:
        print("Erro na instalação das dependências.")

def generate_image():
    noise = np.random.normal(size=(1, 100))
    generated_image = generator.predict(noise)[0]
    generated_image = ((generated_image + 1) * 127.5).astype(np.uint8)
    img = Image.fromarray(generated_image)
    img_path = 'static/generated_image.png'
    img.save(img_path)
    return img_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
**⬤**