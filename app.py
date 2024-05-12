import os
import pickle
from flask import Flask, request, render_template, redirect, url_for
from signal_processing import preprocess_signal, wavelet_transform, classify_image, plot_signal, load_model, load_signal

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
#model = load_model('model/model.pt')
model_name = 'Pretrained PPG Classifier (Swin transformer)'

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        original_signal = load_signal(filepath)

        # Визуализация оригинального сигнала
        plot_signal(original_signal, 'static/original_signal.png')

        # Предобработка сигнала
        cleaned_signal = preprocess_signal(original_signal)
        plot_signal(cleaned_signal, 'static/cleaned_signal.png')

        # Преобразование в вейвлет картинку
        wavelet_transform(cleaned_signal)

        # Классификация изображения
        classification_result = classify_image(0, 'static/wavelet_image.png') # model instead of 0
        classification_result['model_name'] = model_name

        return render_template('index.html',
                               original_signal=True,
                               cleaned_signal=True,
                               wavelet_image=True,
                               classification=classification_result)

    return redirect(url_for('index'))

if __name__ == '__main__':
    # os.makedirs('uploads', exist_ok=True)
    # app.run(debug=True)
    port = int(os.environ.get('PORT', 5000))  # Получаем порт из переменной окружения
    app.run(host='0.0.0.0', port=port, debug=True)