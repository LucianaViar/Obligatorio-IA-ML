# Obligatorio de Machine Learning — cancelación de reservas

Proyecto de clasificación sobre Hotel Booking Demand. El objetivo es estimar, al registrar una reserva, la probabilidad de que sea cancelada. El modelo final congelado es XGBoost (`xgb_depth4`) con umbral 0,22.

## Estructura

```text
Obligatorio-IA-ML/
├── data/hotel_bookings.csv
├── examples/new_reservation.json
├── models/xgb_cancel_pipeline.joblib
├── notebooks/obligatorio_ml.ipynb
├── src/booking_transformers.py
├── src/model_io.py
├── predict_reservation.py
├── requirements.txt
└── README.md
```

## Instalación local

Desde la raíz `Obligatorio-IA-ML`:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Las versiones están fijadas porque los artefactos `joblib`/pickle no constituyen un formato estable entre versiones arbitrarias de Python y scikit-learn. La verificación completa se realizó con Python 3.13.5 y las versiones de `requirements.txt`; cualquier otro runtime, incluido Colab, debe pasar la prueba de inferencia indicada abajo.

## Inferencia independiente del notebook

El modelo puede cargarse y utilizarse sin ejecutar el notebook:

```powershell
python predict_reservation.py
```

Para predecir otra reserva, copiar `examples/new_reservation.json`, conservar las 21 variables requeridas y ejecutar:

```powershell
python predict_reservation.py --input ruta\reserva.json
```

La salida incluye `probabilidad_cancelacion`, el umbral congelado y `prediccion_cancelacion` (`1` indica riesgo de cancelación igual o superior a 0,22).

Con el JSON incluido, la prueba reproducible debe devolver aproximadamente `0.14662006497`, umbral `0.22`, clase `0` y configuración `xgb_depth4`. El SHA-256 verificado del artefacto es `C1BC29E401E22D7BC4747811CB162037E5463EAD3ACCC99012D6A0C747ACD082`.

El artefacto fue creado originalmente dentro del notebook. Por eso pickle registró `HotelBookingFeatureEngineer` y `QuantileClipper` bajo `__main__`. `src/model_io.py` registra las definiciones equivalentes antes de deserializar. Debe utilizarse `load_frozen_artifact`; un `joblib.load` directo no es portable para este archivo. El cargador no reentrena ni modifica el modelo.

Como todo archivo basado en pickle, el modelo debe cargarse únicamente desde este repositorio o desde otra fuente de confianza.

## Ejecución del notebook

```powershell
jupyter lab notebooks/obligatorio_ml.ipynb
```

El notebook busca el CSV en `data/hotel_bookings.csv` al iniciarse desde la raíz y también admite la ruta relativa desde `notebooks/`.

## Google Colab

Para ejecutar el proyecto completo en una sesión limpia:

```python
!git clone https://github.com/LucianaViar/Obligatorio-IA-ML.git
%cd Obligatorio-IA-ML
%pip install -r requirements.txt
```

Después se puede abrir el notebook o comprobar la inferencia con:

```python
!python predict_reservation.py
```

Si solo se sube el notebook, sus celdas pueden ejecutarse cargando manualmente `hotel_bookings.csv`, pero la inferencia reutilizable requiere además `models/`, `src/` y `predict_reservation.py`. Tras `%pip install`, Colab puede solicitar reiniciar la sesión si ya había importado versiones distintas de NumPy, scikit-learn o XGBoost. La compatibilidad debe considerarse confirmada cuando el comando anterior reproduzca la salida de referencia; no es necesario ejecutar el notebook ni volver a entrenar.

## Reproducibilidad y límites

- El artefacto contiene el pipeline completo, las ocho variables derivadas, XGBoost y el umbral 0,22.
- `reservation_status` y demás variables excluidas por fuga no forman parte de la entrada.
- El modelo se entrenó con 70.389 reservas y no se vuelve a ajustar durante la inferencia.
- Las fechas de creación son aproximadas y algunas variables del dataset podrían haber sido actualizadas después de reservar.
- Hubo exposición histórica de 2.923 registros del test en un EDA anterior, aunque no fueron usados para entrenar ni ajustar el modelo final.

## Fuente

N. Antonio, A. de Almeida y L. Nunes, “Hotel booking demand datasets”, *Data in Brief*, vol. 22, pp. 41–49, 2019. DOI: https://doi.org/10.1016/j.dib.2018.11.126
