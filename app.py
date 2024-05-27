import os
import pickle
from flask import Flask, request, render_template, redirect, url_for
from signal_processing import preprocess_signal, wavelet_transform,\
      plot_signal, load_signal
from engine import classify_image, load_model, blood_pressure_estimation

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
classifier_dict = load_model('model/efficientnet-medium.pt')
model_name = 'PPG classifier'

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        original_signal, ground_truth, sid = load_signal(filepath)

        # Визуализация оригинального сигнала
        plot_signal(original_signal, 'static/original_signal.png')

        # Предобработка сигнала
        cleaned_signal = preprocess_signal(original_signal)
        plot_signal(cleaned_signal, 'static/cleaned_signal.png')

        # Преобразование в вейвлет картинку
        wavelet_transform(cleaned_signal)

        # Классификация изображения
        classification_result = classify_image(classifier_dict,
                                               'static/wavelet_image.png')
        classification_result['model_name'] = model_name
        classification_result['target'] = 'Diabetes' if ground_truth else 'Healthy'

        bp_estimation_result = blood_pressure_estimation(sid)

        return render_template('index.html',
                               original_signal=True,
                               cleaned_signal=True,
                               wavelet_image=True,
                               classification=classification_result,
                               blood_pressure=bp_estimation_result)

    return redirect(url_for('index'))

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)