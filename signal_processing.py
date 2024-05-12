import numpy as np
import pywt
import torch
import pickle
import matplotlib.pyplot as plt
import time
import matplotlib
matplotlib.use('Agg')
import neurokit2 as nk
from sklearn.preprocessing import StandardScaler
from torchvision import transforms

fs = 125


def load_model(path):
    model = torch.load(path, map_location=torch.device('cpu'))
    model.eval()
    return model


def load_signal(signal_path):
    """Загрузка сигнала из бинарного файла .pickle"""
    # signal_path = f'raw_ppg/subject_18.pkl'
    with open(signal_path, 'rb') as pickle_file:
        ppgs, _, _ = pickle.load(pickle_file)
    return ppgs[0]


def preprocess_signal(signal, input_sampling_rate=1000):
    """Предобработка сигнала."""
    resampled_signal = nk.signal_resample(signal, sampling_rate=input_sampling_rate,
                                          desired_sampling_rate=fs)
    return nk.ppg_clean(resampled_signal, sampling_rate=fs)


def wavelet_transform(signal, height=64, max_frequency=10, min_frequency = 0.3):
    """
    Преобразование в вейвлет картинку с помощью PyWavelets.
    если длина массива с волной меньше чем 64,
    делаем паддинг последним значением
    """
    wavelet = 'cgau8'
    frequencies = np.arange(max_frequency, min_frequency,
                            -(max_frequency - min_frequency) / height) / fs
    scale = pywt.frequency2scale(wavelet, frequencies)
    widths = scale
    ons = nk.ppg_findpeaks(-signal, sampling_rate=fs)['PPG_Peaks']

    if len(ons) > 1:
        st = ons[0]
        for j, en in enumerate(ons[1:]):
            wave = signal[st:en]
            if len(wave) < 64:
                wave = np.pad(wave, (0, 64 - len(wave)), 'constant', constant_values=(wave[-1]))
            cwt_matrix, _ = pywt.cwt(wave, widths, wavelet)
            matplotlib.image.imsave(f'static/wavelet_image.png', np.abs(cwt_matrix))
            st = en
            break # add feature to use all single waves
    else:
        raise ValueError(f'Слишком короткий сигнал, нужно хотя бы 2 onsets, найдено {len(ons)}!')

    plt.figure(figsize=(10, 5))
    plt.imshow(np.abs(cwt_matrix), cmap='viridis', interpolation='bilinear')
    plt.colorbar()
    plt.ylabel('freqs')
    plt.xlabel('time')
    plt.close()


def classify_image(model, image_path):
    """Классифицировать изображение с помощью обученной модели."""
    # Предобработка изображения с помощью torchvision.transforms
    # transform = transforms.Compose([
    #     transforms.ToPILImage(),
    #     transforms.Grayscale(num_output_channels=1),
    #     transforms.Resize((64, 64)),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5,), (0.5,))
    # ])

    # image = plt.imread(image_path)
    # image = StandardScaler().fit_transform(image)
    # tensor_image = transform(image)
    # tensor_image = tensor_image.unsqueeze(0)  # Добавляем размерность batch

    start_time = time.time()

    # with torch.no_grad():
    #     output = model(tensor_image)
    #     score = torch.softmax(output, dim=1).max().item()
    time.sleep(5)

    time_taken = round(time.time() - start_time, 4)
    score = 0.7
    return {'score': score, 'time_taken': time_taken}


def plot_signal(signal, filename):
    """Визуализация сигнала и сохранение в файл."""
    plt.figure(figsize=(10, 5))
    plt.plot(signal, label='PPG Signal')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.savefig(filename)
    plt.close()