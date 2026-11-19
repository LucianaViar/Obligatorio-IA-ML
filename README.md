# Obligatorio de Machine Learning - cancelación de reservas

Proyecto de clasificación sobre Hotel Booking Demand. El objetivo es estimar, al registrar una reserva, la probabilidad de que sea cancelada. El modelo final congelado es XGBoost (`xgb_depth4`) con calibración Platt y umbral 0,29.

## Contenido del repositorio

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

Todos estos archivos están incluidos en el repositorio. El CSV, el notebook ejecutado y el modelo entrenado se descargan junto con el proyecto.

## Descargar el proyecto

### Opción 1: Download ZIP

1. Abrir [el repositorio en GitHub](https://github.com/LucianaViar/Obligatorio-IA-ML).
2. Seleccionar **Code → Download ZIP**.
3. Descomprimir el archivo.
4. Abrir una terminal en la carpeta extraída que contiene `README.md` y `requirements.txt`.

### Opción 2: Git

```powershell
git clone https://github.com/LucianaViar/Obligatorio-IA-ML.git
cd Obligatorio-IA-ML
```

Los comandos de las secciones siguientes deben ejecutarse desde la raíz del proyecto, donde se encuentran `README.md` y `requirements.txt`.

## Instalación local

El proyecto completo fue verificado con Python 3.13.5 y las versiones fijadas en `requirements.txt`.

En Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

En macOS o Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Las versiones están fijadas porque los artefactos `joblib`/pickle no son un formato estable entre versiones arbitrarias de Python, NumPy, scikit-learn y XGBoost. En otro runtime debe ejecutarse la prueba de inferencia de la sección siguiente.

## Verificación e inferencia independiente

El modelo puede cargarse y utilizarse sin ejecutar ni volver a entrenar el notebook:

```powershell
python predict_reservation.py
```

La salida esperada para `examples/new_reservation.json` es aproximadamente:

```json
{
  "probabilidad_cancelacion": 0.22419869998,
  "umbral": 0.29,
  "prediccion_cancelacion": 0,
  "configuracion": "xgb_depth4"
}
```

Para predecir otra reserva, copiar `examples/new_reservation.json`, conservar las 21 variables requeridas y ejecutar:

```powershell
python predict_reservation.py --input ruta\reserva.json
```

`prediccion_cancelacion` vale `1` cuando la probabilidad es igual o superior a 0,29. El SHA-256 verificado del artefacto es `7FE1EFA73B847BE6845942A5B335FE480FCC5290E31086478484AC51604C9445`.

El artefacto fue creado originalmente dentro del notebook. Por eso pickle registró `HotelBookingFeatureEngineer` y `QuantileClipper` bajo `__main__`. `src/model_io.py` registra las definiciones equivalentes antes de deserializar. Debe utilizarse `load_frozen_artifact`; un `joblib.load` directo no es portable para este archivo. El cargador no reentrena ni modifica el modelo.

Como todo archivo basado en pickle, el modelo debe cargarse únicamente desde este repositorio o desde otra fuente de confianza.

## Ejecutar el notebook localmente

Después de instalar las dependencias y desde la raíz del proyecto:

```powershell
python -m jupyter lab notebooks/obligatorio_ml.ipynb
```

En JupyterLab, seleccionar **Kernel → Restart Kernel and Run All Cells**. El notebook admite tanto `data/hotel_bookings.csv` al ejecutarse desde la raíz como `../data/hotel_bookings.csv` al ejecutarse desde `notebooks/`.

## Ejecutar el notebook en Google Colab

1. Abrir [Google Colab](https://colab.research.google.com/).
2. Seleccionar **Archivo → Abrir cuaderno → GitHub**.
3. Pegar esta dirección del repositorio:

   ```text
   https://github.com/LucianaViar/Obligatorio-IA-ML
   ```

4. Elegir `notebooks/obligatorio_ml.ipynb`.
5. Insertar una celda de código nueva al principio del notebook y ejecutar una vez:

   ```python
   %cd /content
   !test -d Obligatorio-IA-ML || git clone https://github.com/LucianaViar/Obligatorio-IA-ML.git
   %cd /content/Obligatorio-IA-ML
   %pip install -r requirements.txt
   ```

6. Si Colab solicita reiniciar la sesión por un cambio de versiones, seleccionar **Entorno de ejecución → Reiniciar sesión**, volver a ejecutar la celda anterior y esperar a que termine.
7. Seleccionar **Entorno de ejecución → Ejecutar todas**. La celda de preparación deja el directorio activo en `/content/Obligatorio-IA-ML`, por lo que el notebook encuentra `data/hotel_bookings.csv` y puede guardar o cargar `models/xgb_cancel_pipeline.joblib` con las rutas del proyecto.
8. Como comprobación rápida, se puede ejecutar en una celda separada:

   ```python
   !python predict_reservation.py
   ```

La compatibilidad queda confirmada cuando esa comprobación reproduce la probabilidad, el umbral y la clase de referencia. La inferencia no requiere volver a entrenar el modelo.

## Reproducibilidad y límites

- El artefacto contiene el pipeline completo, las ocho variables derivadas, XGBoost, el calibrador Platt y el umbral 0,29.
- `reservation_status` y las demás variables excluidas por fuga no forman parte de la entrada.
- El modelo se entrenó con 70.389 reservas y no se vuelve a ajustar durante la inferencia.
- Las fechas de creación son aproximadas y algunas variables del dataset podrían haber sido actualizadas después de reservar.
- Hubo exposición histórica de 2.923 registros del test en un EDA anterior, aunque no fueron usados para entrenar ni ajustar el modelo final.

## Fuente

N. Antonio, A. de Almeida y L. Nunes, “Hotel booking demand datasets”, *Data in Brief*, vol. 22, pp. 41-49, 2019. DOI: https://doi.org/10.1016/j.dib.2018.11.126

Artículo y datos publicados bajo licencia CC BY 4.0: https://creativecommons.org/licenses/by/4.0/
