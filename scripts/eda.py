import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

class AnalisisSeriesTemporales:
    def __init__(self, df_ventas, df_productos):
        """
        Inicializa el análisis con los DataFrames de ventas y productos
        
        Parameters:
        df_ventas (pandas.DataFrame): DataFrame con fechas como índice y SKUs como columnas
        df_productos (pandas.DataFrame): DataFrame con información de productos
        """
        self.df_ventas = df_ventas.copy()
        self.df_productos = df_productos.copy()
        self.df_ventas.index = pd.to_datetime(self.df_ventas.index)
        
    def analisis_individual(self, sku):
        """
        Realiza análisis detallado de una serie temporal individual
        
        Parameters:
        sku (str): Código del SKU a analizar
        
        Returns:
        dict: Diccionario con resultados del análisis
        """
        serie = self.df_ventas[sku]
        producto_info = self.df_productos[self.df_productos['codigo'] == sku].iloc[0]
        
        # Estadísticas básicas
        stats_basicas = {
            'media': serie.mean(),
            'mediana': serie.median(),
            'std': serie.std(),
            'cv': serie.std() / serie.mean() * 100,
            'asimetria': serie.skew(),
            'curtosis': serie.kurtosis()
        }
        
        # Test de estacionariedad
        adf_test = adfuller(serie.dropna())
        
        # Análisis de estacionalidad
        descomposicion = seasonal_decompose(serie, period=12, model='additive')
        
        # Detección de outliers
        z_scores = np.abs(stats.zscore(serie))
        outliers = {
            'numero': np.sum(z_scores > 3),
            'indices': serie.index[z_scores > 3].tolist()
        }
        
        # Tendencia
        tendencia = {
            'crecimiento_total': ((serie.iloc[-1] - serie.iloc[0]) / serie.iloc[0] * 100),
            'crecimiento_anual': serie.pct_change(12).mean() * 100
        }
        
        # Visualizaciones
        plt.figure(figsize=(15, 10))
        
        # Serie temporal
        plt.subplot(311)
        plt.plot(serie.index, serie.values)
        plt.title(f'Serie Temporal - {sku} - {producto_info["5_producto"]}')
        
        # Descomposición
        plt.subplot(312)
        plt.plot(descomposicion.trend)
        plt.title('Tendencia')
        
        # Boxplot mensual
        plt.subplot(313)
        serie_mensual = serie.groupby(serie.index.month).boxplot()
        plt.title('Patrón Mensual')
        
        return {
            'producto_info': producto_info,
            'stats_basicas': stats_basicas,
            'estacionariedad': {
                'adf_statistic': adf_test[0],
                'p_value': adf_test[1]
            },
            'estacionalidad': descomposicion,
            'outliers': outliers,
            'tendencia': tendencia
        }
    
    def analisis_grupal(self):
        """
        Realiza análisis agregado por diferentes niveles de agrupación
        
        Returns:
        dict: Diccionario con resultados del análisis grupal
        """
        # Unir datos de ventas con información de productos
        df_merged = self.df_ventas.melt(ignore_index=False, var_name='codigo')
        df_merged = df_merged.reset_index()
        df_merged = df_merged.merge(self.df_productos, on='codigo')
        
        # Análisis por categoría
        analisis_categoria = df_merged.groupby(['1_categoria', pd.Grouper(key='index', freq='M')])['value'].sum()
        
        # Análisis por subcategoría
        analisis_subcategoria = df_merged.groupby(['2_sub_categoria', pd.Grouper(key='index', freq='M')])['value'].sum()
        
        # Análisis por grupo
        analisis_grupo = df_merged.groupby(['3_grupo', pd.Grouper(key='index', freq='M')])['value'].sum()
        
        # Análisis por marca
        analisis_marca = df_merged.groupby(['4_marca', pd.Grouper(key='index', freq='M')])['value'].sum()
        
        return {
            'categoria': analisis_categoria,
            'subcategoria': analisis_subcategoria,
            'grupo': analisis_grupo,
            'marca': analisis_marca
        }
    
    def segmentacion_series(self, n_clusters=3):
        """
        Realiza segmentación de series temporales usando características extraídas
        
        Parameters:
        n_clusters (int): Número de clusters para K-means
        
        Returns:
        dict: Resultados de la segmentación
        """
        # Extraer características de las series
        features = pd.DataFrame()
        
        for sku in self.df_ventas.columns:
            serie = self.df_ventas[sku]
            
            features.loc[sku, 'media'] = serie.mean()
            features.loc[sku, 'cv'] = serie.std() / serie.mean()
            features.loc[sku, 'tendencia'] = ((serie.iloc[-1] - serie.iloc[0]) / serie.iloc[0])
            features.loc[sku, 'estacionalidad'] = serie.groupby(serie.index.month).std().mean()
            features.loc[sku, 'autocorr'] = serie.autocorr()
        
        # Normalizar características
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Aplicar K-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(features_scaled)
        
        # Calcular silhouette score
        silhouette_avg = silhouette_score(features_scaled, clusters)
        
        # Agregar clusters al DataFrame de características
        features['cluster'] = clusters
        
        # Unir con información de productos
        resultados = features.merge(self.df_productos, left_index=True, right_on='codigo')
        
        # Análisis de clusters
        analisis_clusters = resultados.groupby('cluster').agg({
            'media': 'mean',
            'cv': 'mean',
            'tendencia': 'mean',
            'estacionalidad': 'mean',
            'autocorr': 'mean',
            'codigo': 'count'
        })
        
        return {
            'features': features,
            'clusters': clusters,
            'silhouette_score': silhouette_avg,
            'analisis_clusters': analisis_clusters,
            'resultados_completos': resultados
        }
    
    def recomendar_modelo(self, sku):
        """
        Recomienda técnicas de modelado basadas en las características de la serie
        
        Parameters:
        sku (str): Código del SKU
        
        Returns:
        dict: Recomendaciones de modelado
        """
        serie = self.df_ventas[sku]
        
        # Analizar características clave
        adf_test = adfuller(serie.dropna())
        es_estacionaria = adf_test[1] < 0.05
        
        descomposicion = seasonal_decompose(serie, period=12, model='additive')
        fuerza_estacional = np.abs(descomposicion.seasonal).mean() / np.abs(serie).mean()
        
        tendencia_significativa = np.abs(serie.corr(pd.Series(range(len(serie))))) > 0.7
        
        # Lógica de recomendación
        recomendaciones = []
        
        if es_estacionaria:
            if fuerza_estacional > 0.1:
                recomendaciones.append("SARIMA")
            else:
                recomendaciones.append("ARIMA")
        else:
            if tendencia_significativa:
                if fuerza_estacional > 0.1:
                    recomendaciones.extend(["ETS", "Prophet"])
                else:
                    recomendaciones.append("Holt-Winters")
            else:
                recomendaciones.append("Prophet")
        
        # Para series con alta volatilidad
        if serie.std() / serie.mean() > 0.5:
            recomendaciones.append("GARCH")
        
        return {
            'caracteristicas': {
                'estacionaria': es_estacionaria,
                'fuerza_estacional': fuerza_estacional,
                'tendencia_significativa': tendencia_significativa
            },
            'modelos_recomendados': recomendaciones
        }

# Ejemplo de uso:
"""
# Cargar datos
df_ventas = pd.read_csv('ventas.csv', index_col='Fecha', parse_dates=True)
df_productos = pd.read_csv('productos.csv')

# Inicializar análisis
analisis = AnalisisSeriesTemporales(df_ventas, df_productos)

# Análisis individual
resultados_sku1 = analisis.analisis_individual('SKU_1')

# Análisis grupal
resultados_grupos = analisis.analisis_grupal()

# Segmentación
segmentacion = analisis.segmentacion_series(n_clusters=4)

# Recomendación de modelos
recomendacion = analisis.recomendar_modelo('SKU_1')
"""