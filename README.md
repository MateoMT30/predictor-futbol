# predictor-futbol

Sistema de pronósticos **probabilísticos** para apuestas deportivas de fútbol,
basado en métodos numéricos y estadísticos (Poisson/Dixon-Coles, regresión
binomial negativa, rating Elo y simulación de Montecarlo) — no en corazonadas
ni en intuición.

## ⚠️ Disclaimer (léelo antes de usar esto para apostar)

Este proyecto es un **modelo estadístico basado en datos históricos**. No
predice el futuro, no garantiza resultados, y sus probabilidades pueden estar
mal calibradas si los datos de entrada son escasos, sesgados o no representan
la forma actual de los equipos. **Las apuestas deportivas implican un riesgo
real de pérdida de dinero.** Nada en este repositorio constituye asesoría
financiera ni una promesa de ganancia. Úsalo bajo tu propio riesgo, con
gestión de banca responsable, y nunca apuestes dinero que no puedas permitirte
perder.

## Qué hace

Dado un partido (local, visitante, liga, fecha) y un histórico de partidos,
el sistema estima:

1. **1X2** (probabilidad de victoria local / empate / victoria visitante)
2. **Hándicap** (europeo/asiático, sobre líneas configurables)
3. **Ambos anotan** (BTTS)
4. **Goles totales** — distribución completa y over/under por línea
5. **Córners totales y por equipo** — media, rango esperado (P10-P90), over/under
6. **Tiros al arco totales y por equipo**
7. **Tarjetas amarillas/rojas totales y por equipo**
8. **Value bets**: si le pasas cuotas de casas de apuestas, compara la
   probabilidad implícita (sin el margen/overround de la casa) contra la
   probabilidad del modelo, y marca dónde el modelo ve más valor del que
   la cuota ofrece.

## Metodología (resumen — el detalle está comentado en cada módulo)

| Módulo | Método | Por qué |
|---|---|---|
| `src/models/goles.py` | Dixon-Coles (Poisson con corrección de marcador bajo) | Separa ataque/defensa por equipo y corrige la subestimación de marcadores bajos correlacionados (0-0, 1-0, 0-1, 1-1) que tiene el Poisson independiente puro |
| `src/models/corners.py`, `tiros.py`, `tarjetas.py` | Regresión binomial negativa | Estas variables tienen sobredispersión (varianza > media); Poisson simple subestima la cola |
| `src/ratings.py` | Elo con ventaja de local y multiplicador por margen de goles | Mide fuerza relativa ajustada por la calidad del rival, no solo promedios crudos |
| `src/simulation.py` | Montecarlo (≥10,000 iteraciones) | Deriva rangos esperados y mercados combinados a partir de las distribuciones individuales, sin necesidad de fórmulas cerradas para cada combinación |
| `src/value_bets.py` | Comparación de probabilidad implícita vs. modelo | Detecta "value bets" corrigiendo el margen comercial de la casa de apuestas |

Todos los parámetros (iteraciones de Montecarlo, half-life de recencia, peso
del ajuste por fuerza del rival, líneas de over/under, umbral de value bet,
etc.) están en [`config.yaml`](config.yaml) — no hay nada "quemado" en el código.

## ¿El modelo tiene en cuenta la actualidad, o solo el histórico?

Esta pregunta tiene dos respuestas distintas, y es importante no mezclarlas:

**Sí tiene en cuenta la forma reciente.** El sistema no usa un promedio
histórico plano — pondera los partidos más recientes mucho más que los
viejos (`half_life_days` en `config.yaml`: a los N días, un partido pesa la
mitad), y el rating Elo de cada equipo se recalcula después de cada
resultado, no una vez al final. Si un equipo viene de una mala racha, el
modelo lo refleja de inmediato en su fuerza estimada. Esto es "actualidad"
en el sentido de forma/racha, y está construido en `ratings.py` y
`models/_stat_common.py`.

**No tiene en cuenta noticias, bajas ni lesiones**, y no puede tenerlas: esa
información no vive en una tabla de resultados históricos, vive en prensa
deportiva, alineaciones probables y redes sociales. Ningún ajuste
estadístico sobre partidos pasados puede inferir que el goleador titular
está suspendido para el próximo partido.

Para ese segundo caso, el sistema expone un **ajuste manual explícito**:

```bash
python src/main.py --local "Colombia" --visitante "Argentina" --liga "Mundial" \
    --datos examples/historico_ejemplo.csv \
    --ajuste-local -0.15
```

`--ajuste-local` / `--ajuste-visitante` (o los campos equivalentes en la app
web) reducen o aumentan los goles esperados de ese equipo en el porcentaje
indicado (`-0.15` = 15% menos). Es un mecanismo para que **tú** incorpores
lo que sabes y el modelo no puede saber — no es una corazonada disfrazada,
es información real (una lesión confirmada, una sanción) que el modelo no
tiene forma de descubrir solo. Por eso el reporte siempre muestra un aviso
explícito cuando se usó, dejando claro qué parte es cálculo estadístico y
qué parte es criterio humano añadido encima.

## Instalación

Requiere Python 3.10+.

```bash
pip install -r requirements.txt
```

## Uso

```bash
python src/main.py --local "Colombia" --visitante "Argentina" --liga "Mundial" \
    --datos examples/historico_ejemplo.csv --cuotas examples/cuotas_ejemplo.json
```

Argumentos:

- `--local`, `--visitante`: nombres de los equipos (obligatorio)
- `--liga`: filtra el histórico por liga/torneo (opcional)
- `--fecha`: fecha del partido, informativa (opcional)
- `--datos`: ruta al CSV/JSON de histórico (por defecto usa el ejemplo incluido)
- `--cuotas`: ruta a un JSON de cuotas para detectar value bets (opcional)
- `--config`: ruta alternativa a `config.yaml`
- `--json-out`: si se pasa, además del reporte en consola vuelca todo a un JSON
- `--html-out`: genera un reporte **HTML autocontenido** (un solo archivo,
  sin servidor ni conexión) pensado para abrirse desde el celular — envíalo
  por WhatsApp/correo/Drive y ábrelo directo en el navegador del teléfono

Ejemplo generando el reporte para celular:

```bash
python src/main.py --local "Colombia" --visitante "Argentina" --liga "Mundial" \
    --datos examples/historico_ejemplo.csv --cuotas examples/cuotas_ejemplo.json \
    --html-out reporte.html
```

## Usarlo desde el celular (app web)

Además del CLI, el proyecto incluye [`app.py`](app.py): una app Flask que
reutiliza exactamente el mismo cálculo (`src/`), pensada para abrir desde
el celular sin terminal y sin que tu computador tenga que estar prendida
(una vez desplegada).

Flujo: **elige competición → te muestra los próximos partidos con fecha y
hora → tocas uno → ves el pronóstico.** No hay que escribir nombres de
equipo a mano ni cargar cuotas — eso se quitó del flujo principal porque
generaba fricción sin aportar a la mayoría de usos.

### Configurar la fuente de datos en vivo (football-data.org)

1. Crea una cuenta gratis en [football-data.org](https://www.football-data.org) y copia tu API key.
2. Configúrala como variable de entorno `FOOTBALL_DATA_API_KEY`:
   - En local (PowerShell): `$env:FOOTBALL_DATA_API_KEY = "tu_key"`
   - En Render: Settings > Environment Variables (ver más abajo).

Sin esta variable, la app sigue funcionando en **modo avanzado** (sección
colapsada en la página de inicio): escribes los equipos a mano usando el
CSV de ejemplo, igual que el CLI — útil para probar sin depender de la API.

**Optimización de cuota gratuita:** football-data.org limita a ~10
peticiones/minuto en el plan free. `src/connectors/football_data_connector.py`
cachea en memoria del servidor cada respuesta (próximos partidos e
histórico) durante 1 hora — así muchas visitas en ese lapso cuestan una
sola llamada real a la API, no una por visita.

### Probarla en tu computador

```bash
pip install -r requirements.txt
python app.py
```

Abre `http://localhost:5000` en tu navegador. Para probarla desde el celular
en la misma red WiFi, usa `http://TU_IP_LOCAL:5000` (revisa tu IP con
`ipconfig` en Windows o `ifconfig`/`ip a` en Mac/Linux).

### Desplegarla gratis en Render (accesible desde cualquier red, no solo tu WiFi)

1. Crea una cuenta gratis en [render.com](https://render.com) (no pide tarjeta).
2. **New > Web Service**, y conecta el repositorio de este proyecto (o sube
   la carpeta si Render te da esa opción).
3. Render detecta el [`Procfile`](Procfile) automáticamente
   (`web: gunicorn app:app`) — no hace falta configurar comando de arranque.
   En "Build command" pon `pip install -r requirements.txt`.
4. En **Environment**, agrega `FOOTBALL_DATA_API_KEY` con tu key. También se
   recomienda agregar `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1` y
   `MKL_NUM_THREADS=1` (ver la nota de memoria más abajo) — `app.py` ya los
   fija por código, pero declararlos también en Render es una segunda capa
   de seguridad por si algún día cambia el punto de entrada del proceso.
5. Plan **Free**, y "Create Web Service".
6. Cuando termine el deploy, te da una URL tipo `predictor-futbol.onrender.com`.
   Ábrela desde el celular, agrégala a la pantalla de inicio, y listo.

**Letra chica del plan gratuito de Render:** el servidor se "duerme" tras ~15
minutos sin uso. La primera petición después de eso tarda 30-50 segundos en
responder mientras despierta; las siguientes son normales. Para uso personal
esto no suele ser un problema.

**Sobre memoria (512 MB en el plan free) y numpy/scipy:** en contenedores
compartidos, OpenBLAS (la librería de álgebra lineal que usan numpy/scipy)
suele detectar el número de núcleos del servidor físico completo, no la
fracción real asignada al contenedor, e intenta lanzar un hilo de cálculo
por núcleo detectado — cada uno reservando su propia memoria. Esto puede
disparar el consumo de RAM muy por encima de lo necesario y provocar que
Render mate el proceso (`Worker was sent SIGKILL! Perhaps out of memory?`),
algo que efectivamente ocurrió probando con una competición de muchos
equipos (el Mundial ampliado a 48 selecciones). `app.py` fija
`OMP_NUM_THREADS=1` y variables equivalentes al inicio para evitarlo. Si
aun así ves ese error con una competición muy grande, considera acotar más
el histórico (`--datos`/parámetro de fecha) o subir de plan.

**Limitación de la fuente de datos gratuita:** football-data.org entrega
resultados y goles, pero no córners/tiros/tarjetas. En el flujo automático
(vía API), esos mercados se muestran con un aviso explícito de "sin datos
suficientes" en vez de inventar un cero — es una limitación de la fuente
gratuita, no un bug. Para tenerlos, usa el modo avanzado con un CSV propio
que sí incluya esas columnas.

## Cómo cargar tus propios datos

El histórico debe ser un CSV o JSON con (al menos) estas columnas:

```
fecha, liga, equipo_local, equipo_visitante, goles_local, goles_visitante
```

Y, si las tienes (recomendado, mejora todos los mercados salvo 1X2/goles):

```
corners_local, corners_visitante,
tiros_arco_local, tiros_arco_visitante,
tarjetas_amarillas_local, tarjetas_amarillas_visitante,
tarjetas_rojas_local, tarjetas_rojas_visitante
```

Revisa [`examples/historico_ejemplo.csv`](examples/historico_ejemplo.csv) como
plantilla. `data_loader.py` valida y limpia automáticamente: descarta filas
sin fecha/equipos/goles, y normaliza variantes de nombres de equipo (espacios,
mayúsculas, tildes) para que "Bogotá" y "bogota" no se traten como equipos
distintos.

Las cuotas van en un JSON con el mismo esquema que
[`examples/cuotas_ejemplo.json`](examples/cuotas_ejemplo.json).

## Conectar una API real (API-Football, Understat, etc.)

El sistema nunca habla directamente contra un archivo: siempre pasa por la
interfaz abstracta `DataSourceConnector` (`src/connectors/base.py`). Para
enchufar una API:

1. Crea `src/connectors/mi_api_connector.py` con una clase que herede de
   `DataSourceConnector` e implemente `fetch_matches()` y `fetch_odds()`
   devolviendo los mismos esquemas que usa `CSVConnector`.
2. En `main.py`, reemplaza la instancia de `CSVConnector` por la tuya.

No hace falta tocar `data_loader.py`, `ratings.py`, los modelos, ni
`simulation.py` — todos consumen el DataFrame ya limpio, sin importar de
dónde vino.

## Estructura del proyecto

```
predictor-futbol/
├── config.yaml              # todos los parámetros ajustables
├── data/                    # datos crudos y procesados del usuario
├── examples/                # histórico y cuotas de ejemplo para probar el CLI
├── src/
│   ├── data_loader.py       # ingestión y limpieza
│   ├── connectors/          # interfaz abstracta + conector CSV
│   ├── ratings.py           # sistema Elo
│   ├── models/              # un archivo por mercado (goles, corners, tiros, tarjetas)
│   ├── simulation.py        # motor de Montecarlo
│   ├── value_bets.py        # comparación contra cuotas
│   └── main.py               # CLI
└── tests/                   # pruebas unitarias por módulo
```

## Ejecutar las pruebas

```bash
pytest tests/ -v
```

> **Nota de este entorno de desarrollo:** las pruebas fueron escritas y
> revisadas manualmente línea por línea, pero no se pudieron ejecutar en este
> entorno porque no hay un intérprete de Python instalado en la máquina donde
> se generó el proyecto. Antes de confiar en el modelo para uso real, corre
> `pytest tests/ -v` tú mismo y confirma que todo pasa.

## Limitaciones del modelo (léelas con atención)

- **Independencia entre mercados**: en esta versión, goles, córners, tiros y
  tarjetas se simulan de forma independiente dentro de cada iteración de
  Montecarlo. En la realidad hay correlación cruzada (más tiros suele venir
  con más córners y más goles) que el modelo actual no captura explícitamente.
- **Sensibilidad a la calidad y cantidad de datos**: con muy pocos partidos
  históricos por equipo, tanto Dixon-Coles como las regresiones binomiales
  negativas caen a estimaciones poco informativas (fuerza cercana al
  promedio). El modelo no "inventa" certeza donde no la hay, pero tampoco es
  confiable con historiales muy cortos (recomendado: 15-20+ partidos por
  equipo como mínimo razonable).
- **No modela lesiones, sanciones, clima, motivación específica de un
  partido (finales, derbis) ni cambios de entrenador recientes**, salvo en
  la medida en que ya se reflejen indirectamente en los resultados recientes
  vía la ponderación por recencia.
- **El sistema Elo es una aproximación simplificada**: no sustituye ratings
  especializados (ej. World Football Elo Ratings) que incorporan más señales.
- **Value bets no son garantía de ganancia a corto plazo**: incluso con una
  ventaja estadística real (edge positivo), la varianza de un solo partido
  es enorme. El "edge" se materializa en el largo plazo, sobre muchas
  apuestas, no en un partido individual.
- **Los datos de ejemplo en `examples/` son sintéticos**, solo para probar
  que el pipeline corre de punta a punta — no reflejan estadísticas reales.

## Stack técnico

`pandas`, `numpy`, `scipy` (distribuciones y optimización), `statsmodels`
(regresión binomial negativa), `PyYAML` (config).
