# Importación de librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
import xgboost as xgb
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from scipy import stats
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import warnings
import plotly.express as px
import plotly.graph_objects as go
from pmdarima import auto_arima
from tbats import TBATS
from itertools import product
warnings.filterwarnings('ignore')

# Constante para almacenar resultados de modelos
RESULTADOS_MODELOS = {}

# Función para cargar y preprocesar los datos
def cargar_datos(file_path, sep=';', fecha_col='Fecha', dayfirst=True):
    """
    Carga y preprocesa los datos desde el archivo CSV
    """
    # Cargar los datos con el formato de fecha correcto
    df = pd.read_csv(
        file_path, 
        sep=sep,
        parse_dates=[fecha_col],
        dayfirst=dayfirst
    )
    
    # Configurar columna de fecha como índice
    df.set_index(fecha_col, inplace=True)
    
    return df

# Función para visualizar serie de tiempo
def visualizar_serie(serie, titulo="Serie Temporal", rolling_window=None):
    """
    Visualiza una serie temporal con o sin promedio móvil
    """
    plt.figure(figsize=(12, 6))
    plt.plot(serie, label='Datos Originales')
    
    if rolling_window:
        rolling_mean = serie.rolling(window=rolling_window).mean()
        plt.plot(rolling_mean, label=f'Promedio Móvil ({rolling_window} períodos)', color='red')
    
    plt.title(titulo)
    plt.xlabel("Fecha")
    plt.ylabel("Valor")
    plt.legend()
    plt.grid()
    plt.show()

# Función para comprobar estacionariedad
def comprobar_estacionariedad(serie):
    """
    Realiza pruebas de estacionariedad ADF
    """
    # Prueba de Dickey-Fuller Aumentada (ADF)
    adf_result = adfuller(serie.dropna())
    
    # Imprimir resultados
    print("Prueba de Dickey-Fuller Aumentada (ADF):")
    print(f'ADF Statistic: {adf_result[0]:.4f}')
    print(f'p-value: {adf_result[1]:.4f}')
    for key, value in adf_result[4].items():
        print(f'\t{key}: {value:.4f}')

    # Interpretación
    if adf_result[1] < 0.05:
        print("\nLa serie es estacionaria según la prueba ADF (se rechaza la hipótesis nula).")
    else:
        print("\nLa serie NO es estacionaria según la prueba ADF (no se rechaza la hipótesis nula).")

# Función para diferenciación de la serie
def diferenciar_serie(serie, d=1, D=0, s=12):
    """
    Realiza diferenciación de una serie temporal
    d: orden de diferenciación no estacional
    D: orden de diferenciación estacional
    s: período estacional
    """
    serie_dif = serie.copy()
    
    # Diferenciación no estacional
    for _ in range(d):
        serie_dif = serie_dif.diff().dropna()
        
    # Diferenciación estacional
    for _ in range(D):
        serie_dif = serie_dif.diff(s).dropna()
    
    return serie_dif

# Función para descomponer serie temporal
def descomponer_serie(serie, modelo='additive', periodo=12):
    """
    Descompone una serie temporal en tendencia, estacionalidad y residuo
    """
    # Descomposición por seasonal_decompose
    descomposicion = seasonal_decompose(serie, model=modelo, period=periodo)
    
    # Graficar descomposición
    plt.figure(figsize=(14, 10))
    
    # Serie original
    plt.subplot(411)
    plt.plot(serie, label='Original')
    plt.legend(loc='upper left')
    plt.title('Descomposición de la Serie Temporal')
    
    # Tendencia
    plt.subplot(412)
    plt.plot(descomposicion.trend, label='Tendencia')
    plt.legend(loc='upper left')
    
    # Estacionalidad
    plt.subplot(413)
    plt.plot(descomposicion.seasonal, label='Estacionalidad')
    plt.legend(loc='upper left')
    
    # Residuo
    plt.subplot(414)
    plt.plot(descomposicion.resid, label='Residuo')
    plt.legend(loc='upper left')
    
    plt.tight_layout()
    plt.show()
    
    # Descomposición por STL
    stl = STL(serie, period=periodo)
    resultado_stl = stl.fit()
    
    # Graficar descomposición STL
    plt.figure(figsize=(14, 10))
    
    # Serie original
    plt.subplot(411)
    plt.plot(serie, label='Original')
    plt.legend(loc='upper left')
    plt.title('Descomposición STL de la Serie Temporal')
    
    # Tendencia
    plt.subplot(412)
    plt.plot(resultado_stl.trend, label='Tendencia (STL)')
    plt.legend(loc='upper left')
    
    # Estacionalidad
    plt.subplot(413)
    plt.plot(resultado_stl.seasonal, label='Estacionalidad (STL)')
    plt.legend(loc='upper left')
    
    # Residuo
    plt.subplot(414)
    plt.plot(resultado_stl.resid, label='Residuo (STL)')
    plt.legend(loc='upper left')
    
    plt.tight_layout()
    plt.show()
    
    return descomposicion, resultado_stl

# Función para crear conjuntos de entrenamiento y prueba
def split_train_test(serie, test_size=0.2):
    """
    Divide la serie temporal en conjuntos de entrenamiento y prueba
    """
    train_size = int(len(serie) * (1 - test_size))
    train, test = serie[:train_size], serie[train_size:]
    return train, test

# Función para evaluar y mostrar resultados del modelo
def evaluar_modelo(y_true, y_pred, nombre_modelo):
    """
    Evalúa un modelo calculando diferentes métricas y visualizando los resultados
    """
    # Calcular métricas
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    # Calcular MAPE con manejo de división por cero
    # Si hay valores cero en y_true, se reemplazan por un valor pequeño para evitar división por cero
    y_true_safe = y_true.copy()
    y_true_safe[y_true_safe == 0] = 1e-10
    mape = mean_absolute_percentage_error(y_true_safe, y_pred) * 100
    
    # Mostrar métricas
    print(f"Resultados del modelo {nombre_modelo}:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"MAPE: {mape:.2f}%")
    
    # Guardar resultados en el diccionario global
    RESULTADOS_MODELOS[nombre_modelo] = {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'y_pred': y_pred
    }
    
    # Visualizar predicciones vs valores reales
    plt.figure(figsize=(12, 6))
    plt.plot(y_true.index, y_true, label='Real', color='blue')
    plt.plot(y_true.index, y_pred, label='Predicción', color='red')
    plt.title(f'Valores Reales vs Predicciones - {nombre_modelo}')
    plt.xlabel('Fecha')
    plt.ylabel('Valor')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Visualizar residuales
    residuales = y_true - y_pred
    plt.figure(figsize=(12, 6))
    plt.plot(y_true.index, residuales, color='green')
    plt.axhline(y=0, color='red', linestyle='--')
    plt.title(f'Residuales - {nombre_modelo}')
    plt.xlabel('Fecha')
    plt.ylabel('Residuales')
    plt.grid(True)
    plt.show()
    
    # Histograma de residuales
    plt.figure(figsize=(10, 6))
    sns.histplot(residuales, kde=True)
    plt.title(f'Distribución de Residuales - {nombre_modelo}')
    plt.xlabel('Residuales')
    plt.ylabel('Frecuencia')
    plt.grid(True)
    plt.show()
    
    return rmse, mae, mape

# Función para el modelo de promedio móvil simple
def modelo_promedio_movil(train, test, window=3):
    """
    Implementa un modelo de promedio móvil simple
    """
    # Concatenar series para hacer pronóstico sobre el período de prueba
    serie_completa = pd.concat([train, test])
    
    # Calcular promedio móvil
    y_pred = serie_completa.rolling(window=window).mean().iloc[-len(test):]
    
    # Evaluar modelo
    return evaluar_modelo(test, y_pred, f'Promedio Móvil (ventana={window})')

# Función para modelo de Holt-Winters
def modelo_holt_winters(train, test, seasonal_periods=12, trend='add', seasonal='add'):
    """
    Implementa un modelo de suavizado exponencial Holt-Winters
    """
    # Entrenar modelo
    model = ExponentialSmoothing(
        train,
        trend=trend,
        seasonal=seasonal,
        seasonal_periods=seasonal_periods
    ).fit()
    
    # Hacer predicciones
    y_pred = model.forecast(len(test))
    
    # Evaluar modelo
    return evaluar_modelo(test, y_pred, f'Holt-Winters ({trend}, {seasonal})')

# Función para modelo ARIMA
def modelo_arima(train, test, order=(1,1,1)):
    """
    Implementa un modelo ARIMA
    """
    # Entrenar modelo
    model = ARIMA(train, order=order).fit()
    
    # Hacer predicciones
    y_pred = model.forecast(len(test))
    
    # Evaluar modelo
    return evaluar_modelo(test, y_pred, f'ARIMA{order}')

# Función para modelo SARIMA
def modelo_sarima(train, test, order=(1,1,1), seasonal_order=(0,0,0,0)):
    """
    Implementa un modelo SARIMA
    """
    # Entrenar modelo
    model = SARIMAX(
        train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    ).fit(disp=False)
    
    # Hacer predicciones
    y_pred = model.forecast(len(test))
    
    # Evaluar modelo
    return evaluar_modelo(test, y_pred, f'SARIMA{order}x{seasonal_order}')

# Función para encontrar el mejor modelo SARIMA
def encontrar_mejor_sarima(train, test, p_range=range(0, 3), d_range=range(0, 2),
                          q_range=range(0, 3), P_range=range(0, 2),
                          D_range=range(0, 2), Q_range=range(0, 2), s=12):
    """
    Busca el mejor modelo SARIMA mediante grid search
    """
    best_aic = float('inf')
    best_order = None
    best_seasonal_order = None
    
    total_combinations = len(p_range) * len(d_range) * len(q_range) * \
                         len(P_range) * len(D_range) * len(Q_range)
    
    print(f"Total de combinaciones a probar: {total_combinations}")
    
    counter = 0
    for p, d, q, P, D, Q in product(p_range, d_range, q_range, P_range, D_range, Q_range):
        counter += 1
        if counter % 10 == 0:
            print(f"Probando combinación {counter}/{total_combinations}")
        
        try:
            model = SARIMAX(
                train,
                order=(p, d, q),
                seasonal_order=(P, D, Q, s),
                enforce_stationarity=False,
                enforce_invertibility=False
            ).fit(disp=False)
            
            if model.aic < best_aic:
                best_aic = model.aic
                best_order = (p, d, q)
                best_seasonal_order = (P, D, Q, s)
                print(f"Nuevo mejor modelo encontrado: SARIMA{best_order}x{best_seasonal_order}, AIC: {best_aic:.2f}")
        except:
            continue
    
    print(f"Mejor modelo SARIMA: {best_order}x{best_seasonal_order}, AIC: {best_aic:.2f}")
    
    # Entrenar y evaluar el mejor modelo encontrado
    best_model = SARIMAX(
        train,
        order=best_order,
        seasonal_order=best_seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    ).fit(disp=False)
    
    y_pred = best_model.forecast(len(test))
    
    # Evaluar modelo
    return evaluar_modelo(test, y_pred, f'SARIMA{best_order}x{best_seasonal_order}'), best_model

# Función para modelo auto_arima
def modelo_auto_arima(train, test, seasonal=True, m=12):
    """
    Implementa un modelo auto_arima que selecciona automáticamente los mejores parámetros
    """
    # Entrenar modelo
    model = auto_arima(
        train,
        seasonal=seasonal,
        m=m,
        start_p=0, start_q=0,
        max_p=5, max_q=5,
        start_P=0, start_Q=0,
        max_P=2, max_Q=2,
        d=None, D=None,
        trace=True,
        error_action='ignore',
        suppress_warnings=True,
        stepwise=True
    )
    
    # Imprimir el modelo seleccionado
    print(f"Mejor modelo auto_arima seleccionado: {model.order}x{model.seasonal_order}")
    
    # Hacer predicciones
    y_pred = model.predict(n_periods=len(test))
    
    # Evaluar modelo
    return evaluar_modelo(test, pd.Series(y_pred, index=test.index), f'auto_arima {model.order}x{model.seasonal_order}'), model

# Función para modelo Prophet
def modelo_prophet(train, test, yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False):
    """
    Implementa un modelo Prophet
    """
    # Preparar datos para Prophet
    df_train = pd.DataFrame({'ds': train.index, 'y': train.values})
    
    # Configurar y entrenar modelo
    model = Prophet(
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=weekly_seasonality,
        daily_seasonality=daily_seasonality
    )
    model.fit(df_train)
    
    # Crear dataframe para predicciones
    future = pd.DataFrame({'ds': test.index})
    
    # Hacer predicciones
    forecast = model.predict(future)
    y_pred = pd.Series(forecast['yhat'].values, index=test.index)
    
    # Evaluar modelo
    return evaluar_modelo(test, y_pred, 'Prophet'), model

# Función para modelo LSTM
def modelo_lstm(train, test, lookback=12, epochs=100, batch_size=32, neurons=50):
    """
    Implementa un modelo LSTM para predicción de series temporales
    """
    # Preparar datos
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train.values.reshape(-1, 1))
    
    # Crear secuencias
    X_train, y_train = [], []
    for i in range(lookback, len(train_scaled)):
        X_train.append(train_scaled[i-lookback:i, 0])
        y_train.append(train_scaled[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    
    # Reshape para modelo LSTM [muestras, pasos de tiempo, características]
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    
    # Construir modelo
    model = Sequential()
    model.add(LSTM(neurons, return_sequences=True, input_shape=(lookback, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(neurons))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Entrenar modelo
    early_stop = EarlyStopping(monitor='val_loss', patience=10)
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1
    )
    
    # Preparar datos para predicción
    # Usamos los últimos 'lookback' valores del train para predecir el inicio del test
    inputs = pd.concat([train.iloc[-lookback:], test]).values.reshape(-1, 1)
    inputs = scaler.transform(inputs)
    
    X_test = []
    for i in range(lookback, lookback + len(test)):
        X_test.append(inputs[i-lookback:i, 0])
    X_test = np.array(X_test)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    # Hacer predicciones
    y_pred_scaled = model.predict(X_test)
    y_pred = scaler.inverse_transform(y_pred_scaled)
    y_pred = pd.Series(y_pred.flatten(), index=test.index)
    
    # Evaluar modelo
    return evaluar_modelo(test, y_pred, 'LSTM'), model, history

# Función para modelo XGBoost
def modelo_xgboost(train, test, lookback=12, max_depth=5, n_estimators=100, learning_rate=0.1):
    """
    Implementa un modelo XGBoost para predicción de series temporales
    """
    # Preparar datos
    X_train, y_train = [], []
    for i in range(lookback, len(train)):
        X_train.append(train.iloc[i-lookback:i].values)
        y_train.append(train.iloc[i])
    X_train, y_train = np.array(X_train), np.array(y_train)
    
    # Entrenar modelo
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        max_depth=max_depth,
        n_estimators=n_estimators,
        learning_rate=learning_rate
    )
    model.fit(X_train, y_train)
    
    # Preparar datos para predicción
    X_test = []
    for i in range(lookback, lookback + len(test)):
        values = pd.concat([train.iloc[-lookback:], test]).iloc[i-lookback:i].values
        X_test.append(values)
    X_test = np.array(X_test)
    
    # Hacer predicciones
    y_pred = model.predict(X_test)
    y_pred = pd.Series(y_pred, index=test.index)
    
    # Evaluar modelo
    return evaluar_modelo(test, y_pred, 'XGBoost'), model

# Función para modelo TBATS (Trigonometric Seasonal, Box-Cox, ARMA residuals, Trend, and Seasonal components)
def modelo_tbats(train, test, seasonal_periods=None):
    """
    Implementa un modelo TBATS para predicción de series temporales con múltiples estacionalidades
    """
    # Configurar períodos estacionales si se proporcionan
    if seasonal_periods is None:
        estimator = TBATS()
    else:
        estimator = TBATS(seasonal_periods=seasonal_periods)
    
    # Entrenar modelo
    model = estimator.fit(train.values)
    
    # Hacer predicciones
    y_pred = model.forecast(steps=len(test))
    y_pred = pd.Series(y_pred, index=test.index)
    
    # Evaluar modelo
    return evaluar_modelo(test, y_pred, f'TBATS {seasonal_periods or ""}'), model

# Función para comparar todos los modelos
def comparar_modelos():
    """
    Compara todos los modelos entrenados y muestra gráficos con los resultados
    """
    if not RESULTADOS_MODELOS:
        print("No se han entrenado modelos para comparar.")
        return
    
    # Crear dataframe con resultados
    resultados = pd.DataFrame({
        'Modelo': list(RESULTADOS_MODELOS.keys()),
        'RMSE': [result['RMSE'] for result in RESULTADOS_MODELOS.values()],
        'MAE': [result['MAE'] for result in RESULTADOS_MODELOS.values()],
        'MAPE': [result['MAPE'] for result in RESULTADOS_MODELOS.values()]
    })
    
    # Ordenar por RMSE
    resultados = resultados.sort_values('RMSE')
    
    # Mostrar tabla de resultados
    print("Comparación de modelos:")
    print(resultados)
    
    # Visualizar métricas por modelo
    plt.figure(figsize=(14, 6))
    
    # RMSE
    plt.subplot(131)
    sns.barplot(x='RMSE', y='Modelo', data=resultados)
    plt.title('RMSE por Modelo')
    plt.tight_layout()
    
    # MAE
    plt.subplot(132)
    sns.barplot(x='MAE', y='Modelo', data=resultados)
    plt.title('MAE por Modelo')
    plt.tight_layout()
    
    # MAPE
    plt.subplot(133)
    sns.barplot(x='MAPE', y='Modelo', data=resultados)
    plt.title('MAPE (%) por Modelo')
    plt.tight_layout()
    
    plt.show()
    
    return resultados

# Función para generar pronósticos con el mejor modelo
def generar_pronostico(modelo, serie, pasos=12, nombre_modelo="Mejor Modelo"):
    """
    Genera pronósticos futuros utilizando el mejor modelo
    """
    # Hacer pronóstico
    if isinstance(modelo, Prophet):
        # Para Prophet
        future = pd.DataFrame({'ds': pd.date_range(start=serie.index[-1], periods=pasos+1, freq='M')[1:]})
        forecast = modelo.predict(future)
        y_pred = pd.Series(forecast['yhat'].values, index=future['ds'])
    elif hasattr(modelo, 'forecast'):
        # Para modelos como ARIMA, SARIMA, Holt-Winters
        y_pred = modelo.forecast(pasos)
    elif hasattr(modelo, 'predict'):
        # Para XGBoost y otros modelos de sklearn
        # Aquí necesitaríamos implementar lógica específica para XGBoost
        # Esta es una solución simple que puede requerir adaptación
        lookback = 12  # Asume los últimos 12 valores
        
        # Preparar datos para predicción
        ultimos_valores = serie.values[-lookback:]
        y_pred = []
        
        for i in range(pasos):
            # Hacer predicción para siguiente valor
            next_pred = modelo.predict(np.array([ultimos_valores]))
            y_pred.append(next_pred[0])
            
            # Actualizar últimos valores (eliminando el primero y añadiendo la predicción)
            ultimos_valores = np.append(ultimos_valores[1:], next_pred[0])
        
        y_pred = pd.Series(
            y_pred,
            index=pd.date_range(start=serie.index[-1], periods=pasos+1, freq='M')[1:]
        )
    else:
        raise ValueError("El tipo de modelo no es compatible con esta función de pronóstico")
    
    # Visualizar pronóstico
    plt.figure(figsize=(12, 6))
    plt.plot(serie.index, serie, label='Histórico', color='blue')
    plt.plot(y_pred.index, y_pred, label='Pronóstico', color='red')
    plt.title(f'Pronóstico para los próximos {pasos} períodos - {nombre_modelo}')
    plt.xlabel('Fecha')
    plt.ylabel('Valor')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return y_pred

# Flujo principal de ejecución para análisis y modelado completo
def analisis_completo(file_path, sku_column, fecha_col='Fecha', sep=';', dayfirst=True, test_size=0.2):
    """
    Realiza un análisis completo y modelado de forecasting para un SKU específico
    """
    # 1. Cargar datos
    print("1. Cargando datos...")
    df = cargar_datos(file_path, sep=sep, fecha_col=fecha_col, dayfirst=dayfirst)
    
    # 2. Seleccionar serie de tiempo para el SKU especificado
    serie = df[sku_column]
    print(f"Serie temporal seleccionada: {sku_column}")
    print(f"Período: {serie.index.min()} a {serie.index.max()}")
    print(f"Total de observaciones: {len(serie)}")
    
    # 3. Visualizar serie de tiempo
    print("\n2. Visualizando serie temporal...")
    visualizar_serie(serie, titulo=f"Serie Temporal - {sku_column}", rolling_window=12)
    
    # 4. Comprobar estacionariedad
    print("\n3. Comprobando estacionariedad...")
    comprobar_estacionariedad(serie)
    
    # 5. Descomponer serie temporal
    print("\n4. Descomponiendo serie temporal...")
    descomposicion, resultado_stl = descomponer_serie(serie, periodo=12)
    
    # 6. Comprobar ACF y PACF
    print("\n5. Analizando autocorrelación...")
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plot_acf(serie, lags=24, ax=plt.gca())
    plt.title('ACF')
    
    plt.subplot(122)
    plot_pacf(serie, lags=24, ax=plt.gca())
    plt.title('PACF')
    plt.tight_layout()
    plt.show()
    
    # 7. División en train y test
    print("\n6. Dividiendo datos en conjuntos de entrenamiento y prueba...")
    train, test = split_train_test(serie, test_size=test_size)
    print(f"Tamaño del conjunto de entrenamiento: {len(train)}")
    print(f"Tamaño del conjunto de prueba: {len(test)}")
    
    # 8. Modelado
    print("\n7. Entrenando modelos de forecasting...")
    
    print("\n7.1. Modelo de Promedio Móvil...")
    modelo_promedio_movil(train, test, window=3)
    
    print("\n7.2. Modelo de Holt-Winters...")
    modelo_holt_winters(train, test, seasonal_periods=12)
    
    print("\n7.3. Modelo ARIMA...")
    # Probar diferentes órdenes de ARIMA
    modelo_arima(train, test, order=(1, 1, 1))
    modelo_arima(train, test, order=(2, 1, 2))

    print("\n7.4. Modelo SARIMA...")
    # Probar diferentes órdenes de SARIMA
    modelo_sarima(train, test, order=(1, 1, 1), seasonal_order=(0, 0, 0, 12))
    modelo_sarima(train, test, order=(2, 1, 2), seasonal_order=(0, 0, 0, 12))

    print("\n7.5. Modelo auto_arima...")
    modelo_auto_arima(train, test, seasonal=True, m=12)

    print("\n7.6. Modelo Prophet...")
    modelo_prophet(train, test, yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)


    print("\n7.7. Modelo LSTM...")
    modelo_lstm(train, test, lookback=12, epochs=100, batch_size=32, neurons=50)

    print("\n7.8. Modelo XGBoost...")
    modelo_xgboost(train, test, lookback=12, max_depth=5, n_estimators=100, learning_rate=0.1)

    print("\n7.9. Modelo TBATS...")
    modelo_tbats(train, test, seasonal_periods=12)

    # 9. Comparar modelos
    print("\n8. Comparando modelos...")
    comparar_modelos()

    # 10. Pronóstico con el mejor modelo
    print("\n9. Generando pronóstico con el mejor modelo...")
    best_model = RESULTADOS_MODELOS['auto_arima (1, 1, 1)x(1, 0, 1, 12)']['y_pred']
    y_pred = generar_pronostico(best_model, serie, pasos=12, nombre_modelo="auto_arima (1, 1, 1)x(1, 0, 1, 12)")

    return serie, train, test, best_model, y_pred

# Ejecutar análisis completo
serie, train, test, best_model, y_pred = analisis_completo(
    file_path="C:/Users/norma/OneDrive/Tecnoquimicas TQ/prueba_tq/data/data_productos.csv",
    sku_column='SKU_25',
    fecha_col='Fecha',
    sep=';',
    dayfirst=True,
    test_size=0.2
)

# Guardar resultados en archivo CSV
RESULTADOS_MODELOS['auto_arima (1, 1, 1)x(1, 0, 1, 12)']['y_pred'].to_csv('data/pronostico.csv')

# Guardar resultados en archivo Excel
writer = pd.ExcelWriter('data/resultados.xlsx')
for key, value in RESULTADOS_MODELOS.items():
    value['y_pred'].to_excel(writer, sheet_name=key)
writer.save()
