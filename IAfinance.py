import sys
import os
import threading
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QComboBox, QSlider, QDialog, QPushButton
from PyQt5.QtCore import Qt, QCoreApplication, QDateTime
import mplfinance as mpf
import matplotlib.pyplot as plt
import yfinance as yf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

MODEL_FILE = 'C:\\Users\\Satriano\\Documents\\Programação\\Python\\IA\\IA financeira\\Modelos\\model.pkl'
MINIMUM_DATA_LENGTH = 100  # Definição do valor desejado

# Função para exibir os candles na interface gráfica
def show_candles(data, signals=None):
    if data.empty:
        print("Dados vazios. Nada a exibir.")
        return

    data['Date'] = pd.to_datetime(data['Date'])
    data = data.set_index('Date')

    kwargs = dict(type='candle', style='yahoo', volume=True)

    fig, axs = plt.subplots(len(signals), 1, figsize=(10, len(signals) * 5))

    for i, (time_frame, signal_values) in enumerate(signals.items()):
        ax = axs[i] if len(signals) > 1 else axs

        if len(data) == 0:
            print(f"Não há dados para o período de tempo {time_frame}.")
            continue

        if len(signal_values) == 0:
            print(f"Não há sinais para o período de tempo {time_frame}.")
            continue

        mpf.plot(data, ax=ax, **kwargs)
        ax.scatter(data.index, data['Close'], c=signal_values, cmap='cool', edgecolors='black', linewidths=1)

        ax.set_title(f'Candles ({time_frame})')

    plt.tight_layout()
    plt.show()

# Obter dados 
def get_historical_data(symbol, start_date, end_date, interval):
    interval_mapping = {
        '1d': '1d',
        '1wk': '1wk',
        '1mo': '1mo',
    }
    try:
        interval = interval_mapping[interval]
    except KeyError:
        print(f"Invalid interval value: {interval}")
        return None
    try:
        data = yf.download(symbol, start=start_date, end=end_date, interval=interval)
        data = data.reset_index()
        return data
    except Exception as e:
        print(f"Error while fetching historical data: {e}")
        return None

# Cria o caminho para os diretórios de treinamento, validação e teste
train_dir = 'C:\\Users\\Satriano\\Documents\\Programação\\Python\\IA\\IA financeira\\Modelos\\train'
validation_dir = 'C:\\Users\\Satriano\\Documents\\Programação\\Python\\IA\\IA financeira\\Modelos\\validation'
test_dir = 'C:\\Users\\Satriano\\Documents\\Programação\\Python\\IA\\IA financeira\\Modelos\\test'

# Define os parâmetros da rede neural
input_shape = 224 * 224 * 3
batch_size = 32
epochs = 10

# Cria o modelo de rede neural
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(input_shape,)))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Configura geradores de dados para normalizar e aumentar os dados de treinamento e teste
train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary')

# Treina o modelo usando os dados de treinamento e validação
history = model.fit(
      train_generator,
      steps_per_epoch=train_generator.samples // batch_size,
      epochs=epochs,
      validation_data=validation_generator,
      validation_steps=validation_generator.samples // batch_size)

# Avalia o modelo usando os dados de teste
test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary')

score = model.evaluate(test_generator, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Função para adicionar indicadores aos dados
def add_indicators(data):
    try:
        # Exponential Moving Average strategy
        ema9 = data['Close'].ewm(span=9, adjust=False).mean()
        ema21 = data['Close'].ewm(span=21, adjust=False).mean()
        data['ema9'] = ema9
        data['ema21'] = ema21

        # Relative Strength Index strategy
        rsiLength = 14
        delta = data['Close'].diff()
        gain = delta.mask(delta < 0, 0)
        loss = -delta.mask(delta > 0, 0)
        avg_gain = gain.rolling(rsiLength).mean()
        avg_loss = loss.rolling(rsiLength).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        data['rsi'] = rsi

        # Bollinger Bands strategy
        bbLength = 20
        bbMultiplier = 2
        sma = data['Close'].rolling(bbLength).mean()
        std = data['Close'].rolling(bbLength).std()
        upper_band = sma + bbMultiplier * std
        lower_band = sma - bbMultiplier * std
        data['upper_band'] = upper_band
        data['lower_band'] = lower_band

        # Directional Movement Index strategy
        diLength = 14
        tr = pd.DataFrame(index=data.index)
        tr['hl'] = data['High'] - data['Low']
        tr['hc'] = abs(data['High'] - data['Close'].shift())
        tr['lc'] = abs(data['Low'] - data['Close'].shift())
        tr['tr'] = tr[['hl', 'hc', 'lc']].max(axis=1)
        atr = tr['tr'].rolling(diLength).mean()
        dmPlus = np.where((data['High'] - data['High'].shift()) > (data['Low'].shift() - data['Low']),
                          data['High'] - data['High'].shift(), 0)
        dmMinus = np.where((data['Low'].shift() - data['Low']) > (data['High'] - data['High'].shift()),
                           data['Low'].shift() - data['Low'], 0)
        tr['dmPlus'] = dmPlus
        tr['dmMinus'] = dmMinus
        tr['diPlus'] = 100 * (tr['dmPlus'].rolling(diLength).sum() / atr)
        tr['diMinus'] = 100 * (tr['dmMinus'].rolling(diLength).sum() / atr)
        tr['dx'] = 100 * abs((tr['diPlus'] - tr['diMinus']) / (tr['diPlus'] + tr['diMinus']))
        tr['adx'] = tr['dx'].rolling(diLength).mean()
        data['adx'] = tr['adx'] 

        # Volume strategy
        data['volume'] = data['Volume'].astype(float)

    except Exception as e:
        print(f"Error adding indicators: {str(e)}")

    return data

# Função para criar o modelo LSTM
def create_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    return model

# Função para fazer previsões usando o modelo treinado
def predict(model, X):
    try:
        predicted_prices = model.predict(X)
        return predicted_prices
    except Exception as e:
        raise ValueError(f"Erro ao fazer previsões: {e}")

# Função para obter o sinal
def get_signals(symbol, start_date, end_date, interval):
    """
    Obtém sinais para o símbolo especificado entre as datas de início e fim, no intervalo de tempo especificado.
    """
    print(f"Obtendo sinais...\nSímbolo: {symbol}\nData de início: {start_date}\nData de fim: {end_date}\nIntervalo de tempo: {interval}")
    # Obter os dados históricos usando a função get_historical_data()
    data = get_historical_data(symbol, start_date, end_date, interval)
    if data is None:
        print("Não foi possível obter os dados históricos.")
        return None
    # Adicionar indicadores aos dados
    data = add_indicators(data)
    # Treinar o modelo ou carregar o modelo treinado
    if os.path.exists(MODEL_FILE):
        # Carregar o modelo treinado a partir do arquivo
        model = load_model(MODEL_FILE)
    else:
        # Treinar o modelo do zero
        model = train_model(data)
        os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
        model.save(MODEL_FILE)
    # Obter os sinais de negociação usando o modelo treinado
    X = data[['Close']].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(X)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    signals = predict(model, X)
    # Retornar os sinais de negociação
    return signals

# Função para treinar o modelo
def train_model(data):
    try:
        # Prepare input and output data
        X = data[['Close']].values.reshape(-1, 1)
        y = data[['Close']].values.reshape(-1, 1)
        # Split data into training and testing sets
        train_size = int(len(X) * 0.8)
        train_X, test_X = X[:train_size], X[train_size:]
        train_y, test_y = y[:train_size], y[train_size:]
        # Normalize the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        train_X = scaler.fit_transform(train_X)
        test_X = scaler.transform(test_X)
        # Reshape the data to the expected format by the LSTM model
        train_X = np.reshape(train_X, (train_X.shape[0], train_X.shape[1], 1))
        test_X = np.reshape(test_X, (test_X.shape[0], test_X.shape[1], 1))
        # Create the LSTM model
        model = create_model((train_X.shape[1], 1))
        # Check if the model was created successfully
        if model is None:
            print("Error creating the model.")
            return None
        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')
        # Train the model
        history = model.fit(train_X, train_y, epochs=100,
                            batch_size=32, verbose=1, validation_data=(test_X, test_y))
        # Save the model to a file
        model.save(MODEL_FILE)
        # Save the loss history to a file (optional)
        loss_history = history.history['loss']
        with open('loss_history.txt', 'w') as file:
            file.write('\n'.join(map(str, loss_history)))
        return model
    except Exception as e:
        print(f"Error training the model: {e}")
        return None

# Função principal
def main():
    try:
        # Define download parameters
        symbol = 'AAPL'
        start_date = '2010-01-01'
        end_date = '2021-12-31'
        interval = '1d'
        # Retrieve historical data and add indicators
        data = add_indicators(get_historical_data(symbol, start_date, end_date, interval))
        # Check if enough data to train model
        if len(data) <= MINIMUM_DATA_LENGTH:
            print("Insufficient data to train model.")
            sys.exit()
        # Load or train model
        model = load_model(MODEL_FILE) if os.path.exists(MODEL_FILE) else train_model(data)
        os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
        model.save(MODEL_FILE)
        # Mostrar o gráfico
        signals = get_signals(symbol, start_date, end_date, interval)
        show_candles(data, signals)

    except Exception as e:
        print(f"Error during program execution: {e}")

# Interface gráfica 
def create_gui():
    app = QApplication(sys.argv)
    window = QWidget()
    layout = QVBoxLayout()
    label = QLabel("Selecione o símbolo:")
    combo_box = QComboBox()
    combo_box.addItems(["AAPL", "GOOGL", "MSFT"])
    slider = QSlider(Qt.Horizontal)
    slider.setRange(0, 100)
    slider.setValue(50)
    button_train = QPushButton("Treinar Modelo")
    button_signals = QPushButton("Obter Sinais")
    button_close = QPushButton("Fechar")
    button_time = QPushButton("Selecionar Tempo Gráfico")
    button_time.clicked.connect(lambda: open_time_dialog(combo_box.currentText()))
    layout.addWidget(label)
    layout.addWidget(combo_box)
    layout.addWidget(slider)
    layout.addWidget(button_train)
    layout.addWidget(button_signals)
    layout.addWidget(button_close)
    layout.addWidget(button_time)
    window.setLayout(layout)
    window.show()
    button_train.clicked.connect(lambda: train_model(combo_box.currentText()))
    button_signals.clicked.connect(lambda: get_signals(combo_box.currentText(), '2010-01-01', '2021-12-31', '1d'))
    button_close.clicked.connect(app.quit)
    sys.exit(app.exec_())

def open_time_dialog(symbol, handle_time_button_click):
    time_dialog = QDialog()
    time_dialog.setWindowTitle("Selecionar Tempo Gráfico")
    layout = QVBoxLayout()
    time_frames = [
        "1 Min", "5 Min", "15 Min", "30 Min", "1 Hour", "1 Day", 
        "5 Days", "1 Week", "1 Month", "3 Months", "6 Months", 
        "1 Year", "5 Years"
    ]
    buttons = []
    for time_frame in time_frames:
        button = QPushButton(time_frame)
        button.clicked.connect(
            lambda checked, time_frame=time_frame, symbol=symbol: 
                handle_time_button_click(time_frame, symbol)
        )
        buttons.append(button)
        layout.addWidget(button)
    time_dialog.setLayout(layout)
    time_dialog.exec_()

def handle_time_button_click(time_frame, symbol):
    try:
        start_date, end_date = get_date_range(time_frame)
        get_signals(symbol, start_date, end_date, time_frame)
    except Exception as e:
        print(f"Um erro ocorreu ao obter sinais: {e}")

def get_date_range(time_frame):
    import datetime
    current_date = datetime.date.today()
    end_date = current_date.strftime("%Y-%m-%d")
    start_dates = {
        "1 Min": datetime.date(2018, 1, 1),
        "5 Min": datetime.date(2018, 1, 1),
        "15 Min": datetime.date(2018, 1, 1),
        "30 Min": datetime.date(2018, 1, 1),
        "1 Hour": datetime.date(2018, 1, 1),
        "1 Day": datetime.date(2018, 1, 1),
        "5 Days": datetime.date(2018, 1, 1),
        "1 Week": datetime.date(2018, 1, 1),
        "1 Month": datetime.date(2018, 1, 1),
        "3 Months": datetime.date(2018, 1, 1),
        "6 Months": datetime.date(2018, 1, 1),
        "1 Year": datetime.date(2018, 1, 1),
        "5 Years": datetime.date(2018, 1, 1)
    }
    start_date = start_dates.get(time_frame, None)
    return start_date, end_date

def get_signals(symbol, start_date, end_date, interval):
    """
    Obtém sinais para o símbolo especificado entre as datas de início e fim, no intervalo de tempo especificado.
    """
    print(f"Obtendo sinais...\nSímbolo: {symbol}\nData de início: {start_date}\nData de fim: {end_date}\nIntervalo de tempo: {interval}")
    # Obter os dados históricos usando a função get_historical_data()
    data = get_historical_data(symbol, start_date, end_date, interval)
    if data is None:
        print("Não foi possível obter os dados históricos.")
        return
    # Adicionar indicadores aos dados
    data = add_indicators(data)
    # Verificar se o modelo já foi treinado
    if os.path.exists(MODEL_FILE):
        # Carregar o modelo treinado a partir do arquivo
        model = load_model(MODEL_FILE)
    else:
        print("Modelo não encontrado. Por favor, treine o modelo primeiro.")
        return
    # Obter os sinais de negociação usando o modelo treinado
    signals = get_signals(data, model)
    # Plotar o gráfico
    show_candles(data, signals)

if __name__ == "__main__":
    create_gui()
