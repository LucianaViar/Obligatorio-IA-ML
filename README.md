# Obligatorio de Machine Learning - cancelación de reservas

Primera etapa de un proyecto de clasificación cuyo objetivo es predecir `is_canceled` a partir del dataset de reservas hoteleras.

## Estructura

```text
Obligatorio-IA-ML/
├── data/
│   └── hotel_bookings.csv
├── notebooks/
│   └── obligatorio_ml.ipynb
├── src/
├── informe/
├── requirements.txt
└── README.md
```

## Alcance de esta etapa

- Carga reproducible del CSV.
- Verificación de dimensiones, tipos, valores ausentes, duplicados y distribución de `is_canceled`.
- Revisión de fuga de información, con énfasis en `reservation_status` y `reservation_status_date`.
- Propuesta de partición temporal y validación sin entrenar modelos.
- Conservación íntegra de los datos: no se eliminan columnas o filas y no se imputan valores.

El PDF provisto es el artículo técnico **Hotel booking demand datasets** (Antonio, Almeida y Nunes, 2019), no una rúbrica o consigna universitaria. Por eso, el notebook separa los requisitos solicitados por el equipo de las definiciones y advertencias metodológicas de la fuente.

## Ejecución local

Desde la carpeta `Obligatorio-IA-ML`:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
jupyter lab notebooks/obligatorio_ml.ipynb
```

El notebook busca el CSV en `data/hotel_bookings.csv` cuando Jupyter se inicia desde la raíz del proyecto y también admite la ruta relativa desde `notebooks/`.

## Google Colab

1. Subir `notebooks/obligatorio_ml.ipynb` a Colab.
2. Si el proyecto completo no está disponible en `/content`, ejecutar la celda de carga y seleccionar `hotel_bookings.csv` cuando Colab lo solicite.
3. Ejecutar **Entorno de ejecución > Ejecutar todo**.

La celda de carga primero intenta localizar el archivo automáticamente y sólo solicita una subida si no lo encuentra.

## Decisiones registradas

- `reservation_status` reproduce exactamente el resultado observado: `Check-Out` aparece únicamente con clase 0; `Canceled` y `No-Show`, únicamente con clase 1. Se marca como fuga directa y no se usará como predictor.
- `reservation_status_date` se registra después de que cambia el estado y puede ocurrir después del evento que se intenta predecir. Se marca como fuga temporal.
- `assigned_room_type`, `booking_changes`, `country`, `deposit_type` y otras variables dependientes del momento operativo requieren confirmar cuándo estará disponible la predicción. En esta etapa sólo se señalan para auditoría; no se eliminan.
- Se propone reservar los últimos cuatro meses como test y los cuatro meses previos como validación. El entrenamiento usa datos anteriores. Dentro del entrenamiento, la futura comparación de modelos deberá usar ventanas temporales expansivas sin mezclar el futuro con el pasado.

## Fuente del dataset

N. Antonio, A. de Almeida y L. Nunes, “Hotel booking demand datasets”, *Data in Brief*, vol. 22, pp. 41-49, 2019. DOI: https://doi.org/10.1016/j.dib.2018.11.126
