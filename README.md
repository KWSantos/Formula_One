# Análise de Dados e Previsão de Estratégia para a Fórmula 1 🏎️

Este repositório contém um projeto completo de análise de dados focado no automobilismo da Fórmula 1. Utilizando a API **FastF1**, foram coletados, processados e analisados dados de corrida do **Grande Prêmio da Itália (Monza)** entre os anos de **2018 e 2024**.

O objetivo final é treinar um modelo de **Rede Neural Recorrente (LSTM)** capaz de prever os tempos de volta, um componente essencial para determinar a estratégia de pneus mais eficaz para a corrida.

## 🎯 Objetivo do Projeto

A estratégia de pneus e a previsão de ritmo de corrida são fatores críticos para o sucesso na Fórmula 1. Este projeto busca responder à seguinte pergunta:

> É possível, com base em dados históricos de voltas, compostos de pneus, condições climáticas e telemetria, treinar um modelo de série temporal para prever com acurácia o desempenho de um piloto em voltas futuras?

## 🛠️ Tecnologias Utilizadas

* **Linguagem:** Python 3.x
* **Coleta de Dados:** [FastF1](https://theoehrly.github.io/Fast-F1/)
* **Análise e Manipulação de Dados:** Pandas, NumPy
* **Machine Learning:** TensorFlow, Keras, Scikit-learn
* **Visualização de Dados:** Matplotlib, Seaborn 
* **Ambiente:** Jupyter Notebook

## 📊 Processo de Dados

A qualidade do modelo depende diretamente da qualidade dos dados. O processo foi dividido em quatro etapas principais:

### 1. Coleta de Dados via FastF1
Os dados foram coletados para o Grande Prêmio da Itália de 2018 a 2024. Um loop foi criado para buscar todas as sessões relevantes (Treinos Livres, Qualificação, Corrida e Sprint, quando aplicável), tratando exceções para anos ou sessões sem dados.

```python
import fastf1
import pandas as pd

# Listas para armazenar os objetos de sessão
sessions_monza_race = []
#...
weather_data = []

gp_name = 'Italian Grand Prix'

for year in range(2018, 2025):
    print(f"\n--- Processando ano: {year} ---")
    # Lógica para tratar o formato de Fim de Semana com Sprint (2021)
    if year == 2021:
        sessions_to_load = {'Race': ..., 'Sprint': ..., 'Qualifying': ...}
    else:
        sessions_to_load = {'Race': ..., 'Qualifying': ..., 'FP1': ..., 'FP2': ..., 'FP3': ...}

    for session_name, session_list in sessions_to_load.items():
        try:
            session = fastf1.get_session(year, gp_name, session_name)
            session.load() # Carrega os dados da sessão (incluindo telemetria)
            weather_data.append(session.weather_data)
            session_list.append(session)
            print(f"  -> Sucesso: '{session_name}' de {year} carregado.")
        except Exception as e:
            print(f"  -> Falha: Não foi possível carregar '{session_name}' de {year}.")
```

### 2. Consolidação e Junção de Dados
Após a coleta, os dados de voltas (`laps`) e meteorológicos (`weather_data`) de todas as sessões e anos foram consolidados em dois DataFrames principais. Em seguida, foi utilizada a função `pandas.merge_asof` para enriquecer os dados de cada volta com as informações meteorológicas mais próximas no tempo.

```python
df_laps = pd.concat(all_laps, ignore_index=True)
df_weather = pd.concat(all_weather, ignore_index=True)

# Ordena por tempo para a junção
df_laps = df_laps.sort_values(by='LapStartTime').reset_index(drop=True)
df_weather = df_weather.sort_values(by='Time').reset_index(drop=True)

# Junção 'asof' para encontrar o dado de clima mais próximo para cada volta
df_merged = pd.merge_asof(
    left=df_laps,
    right=df_weather,
    left_on='LapStartTime',
    right_on='Time',
    by='Year',
    direction='nearest'
)
```

### 3. Feature Engineering e Limpeza
Esta é a etapa mais crítica. Novas features foram criadas e os dados foram limpos para garantir que apenas informações relevantes e precisas fossem usadas no modelo:
* **`FuelLoad`**: Uma estimativa da carga de combustível foi criada, assumindo um consumo de 1.6 kg por volta a partir de uma carga inicial de 110 kg.
* **Codificação Categórica**: `Compound` (composto do pneu) e `Driver` foram transformados em features numéricas usando One-Hot Encoding (`pd.get_dummies`).
* **Limpeza de Dados**:
    * Voltas sem `LapTime` foram removidas.
    * Apenas voltas marcadas como `IsAccurate == True` pelo FastF1 foram mantidas.
    * `LapTime` foi convertido para segundos (`LapTimeSeconds`) e outliers (tempos > 200s) foram descartados.
    * Dezenas de colunas irrelevantes ou com dados redundantes foram removidas para simplificar o modelo.

```python
# Feature Engineering: Carga de Combustível
df_merged['FuelLoad'] = 110 - (df_merged['LapNumber'] * 1.6)

# One-Hot Encoding
df_merged = pd.get_dummies(df_merged, columns=['Compound', 'Driver'], prefix=['Compound', 'Driver'])

# Limpeza e seleção de voltas válidas
df_clean = df_merged[df_merged['IsAccurate'] == True].copy()
df_clean['LapTimeSeconds'] = df_clean['LapTime'].dt.total_seconds()
df_clean.dropna(subset=['LapTimeSeconds'], inplace=True)
df_clean = df_clean[df_clean['LapTimeSeconds'] < 200]

# Remoção de colunas desnecessárias
features_to_exclude = ['Time_x', 'LapStartTime', 'Team', 'Position', ...]
df_final = df_clean.drop(columns=features_to_exclude, errors='ignore')
```

### 4. Salvando o Dataset Processado
O DataFrame final, limpo e processado, foi salvo em um arquivo CSV para facilitar o reuso e a etapa de modelagem.
```python
df_final.to_csv('dados_monza_2018_2024.csv', index=False)
```

## 🧠 O Modelo LSTM

Um modelo LSTM (Long Short-Term Memory) foi escolhido por sua eficácia em aprender com dados sequenciais, como uma série temporal de voltas de corrida.

### 1. Preparação para Séries Temporais
* **Divisão Temporal**: Os dados foram divididos em treino (`anos < 2024`) e teste (`ano == 2024`), garantindo que o modelo seja testado em dados que ele nunca viu, simulando um cenário de previsão real.
* **Normalização**: Todas as features foram normalizadas entre 0 e 1 usando `MinMaxScaler` para otimizar o processo de treinamento da rede neural.
* **Criação de Sequências**: Os dados foram transformados em sequências. Para cada ponto, o modelo recebe uma sequência de `10` voltas anteriores para prever o tempo da próxima volta. Esse processo é feito ano a ano para não misturar dados de GPs diferentes em uma mesma sequência.

```python
from sklearn.preprocessing import MinMaxScaler

# Divisão temporal dos dados
cut_off_year = 2024
train_df = df_final[df_final['Year'] < cut_off_year]
test_df = df_final[df_final['Year'] == cut_off_year]

# Função para criar sequências de dados
def create_sequences(X_data, y_data, sequence_length=10):
    X_sequences, y_sequences = [], []
    for i in range(len(X_data) - sequence_length):
        X_sequences.append(X_data[i:(i + sequence_length)])
        y_sequences.append(y_data[i + sequence_length])
    return np.array(X_sequences), np.array(y_sequences)

# Normalização e criação das sequências para treino e teste
scaler = MinMaxScaler()
```

### 2. Arquitetura do Modelo
O modelo foi construído com a API Keras do TensorFlow, contendo duas camadas LSTM com regularização `Dropout` para evitar overfitting e uma camada `Dense` de saída para a predição final.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

sequence_length = X_train.shape[1]
n_features = X_train.shape[2]

model = Sequential(name="Modelo_F1_LSTM")
model.add(LSTM(units=128, activation='tanh', return_sequences=True, input_shape=(sequence_length, n_features)))
model.add(Dropout(0.2))
model.add(LSTM(units=64, activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(units=1, activation='linear')) # Ativação linear para regressão
```
**Sumário do Modelo:**
```
Model: "Modelo_F1_LSTM"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 lstm (LSTM)                 (None, 10, 128)           ...
 dropout (Dropout)           (None, 10, 128)           0
 lstm_1 (LSTM)               (None, 64)                ...
 dropout_1 (Dropout)         (None, 64)                0
 dense (Dense)               (None, 1)                 65
=================================================================
Total params: ...
Trainable params: ...
Non-trainable params: 0
_________________________________________________________________
```

### 3. Compilação e Treinamento
O modelo foi compilado com o otimizador `Adam` e a função de perda `mean_squared_error`, adequada para problemas de regressão. Callbacks como `EarlyStopping` e `ReduceLROnPlateau` foram usados para um treinamento mais robusto e eficiente.

```python
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_absolute_error'])

# Callbacks para otimizar o treino
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

# Treinamento do modelo
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    shuffle=False, # Importante para séries temporais
    callbacks=[early_stopping, reduce_lr]
)
```

## 📈 Resultados

```python
# Código para gerar o gráfico de perda
import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='Perda no Treino')
plt.plot(history.history['val_loss'], label='Perda na Validação')
plt.title('Curva de Aprendizado do Modelo')
plt.xlabel('Épocas')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.show()
```

## 💡 Próximos Passos e Melhorias

* [ ] Expandir a análise para outros circuitos para criar um modelo mais generalista.
* [ ] Testar outras arquiteturas de modelos (ex: GRU, Transformers for Time Series).
* [ ] Incorporar mais dados de telemetria (RPM, velocidade, acionamento de freio/acelerador).
* [ ] Construir uma interface simples com Streamlit ou Flask para fazer previsões interativas.

## ✒️ Autor

**[Kauê Santos]**

* [LinkedIn](https://www.linkedin.com/in/kauê-santos-0a381b25a)
* [GitHub](https://github.com/KWSantos)
