# Notas de proyecto — predictor-futbol

> Este documento existe para poder retomar el proyecto en una conversación
> nueva sin perder contexto. Está escrito para que lo lea Claude (o tú) y
> entienda de inmediato qué hay construido, por qué, y qué falta.

## Qué es esto

App web de pronósticos probabilísticos de fútbol (1X2, goles, córners,
tiros al arco, tarjetas), con enfoque en el Mundial 2026. Corre 100% con
métodos estadísticos reales (Dixon-Coles, Elo, Montecarlo) — nunca inventa
un número: si no hay dato, el reporte dice explícitamente "sin datos
suficientes".

**Repositorio:** https://github.com/MateoMT30/predictor-futbol
**Desplegado en:** Render (plan free) — https://predictor-futbol-v4n8.onrender.com
**Stack:** Python 3.13, Flask, pandas/numpy/scipy/statsmodels, gunicorn.

## Arquitectura en 2 capas

1. **`src/`** — el motor estadístico puro (sin nada de web). Lo usa tanto
   `src/main.py` (CLI) como `app.py` (web).
2. **`app.py`** — la app Flask que se usa de verdad (el CLI quedó como
   herramienta secundaria/de pruebas).

### Flujo principal de la app web

```
GET /                          -> elegir competición
GET /partidos?competition=X    -> lista de próximos partidos, agrupada por
                                   día ("Hoy"/"Mañana"/día de semana),
                                   estilo apps deportivas (365Scores)
GET /predecir?competition=X&local=A&visitante=B -> el reporte
```

Modo avanzado (sin API, para pruebas): formulario manual en `/` que usa
`CSVConnector` sobre `examples/historico_ejemplo.csv`.

## Metodología estadística (por qué cada cosa)

- **Dixon-Coles** (`src/models/goles.py`) para goles: ataque/defensa por
  equipo vía máxima verosimilitud, con corrección de marcador bajo.
  **Regularización L2** agregada (parámetro `regularization=1.5` en
  `GoalsModelConfig`) — sin esto, equipos con pocos partidos en el
  historial (típico al empezar un Mundial) pueden dar goles esperados
  absurdos (se vio un caso real: 24.5 goles esperados). Calibrado
  empíricamente contra datos reales.
- **Elo** (`src/ratings.py`) con ventaja de local y multiplicador por
  margen de goles, actualizado partido a partido.
- **Binomial negativa** (`src/models/_stat_common.py`) para córners,
  tiros, tarjetas — porque tienen sobredispersión (Poisson simple la
  subestima).
- **Montecarlo** (`src/simulation.py`, 15,000 iteraciones) para combinar
  todo y derivar rangos esperados y mercados combinados.
- **Marcador exacto más probable**: además del promedio de goles (xG-like,
  nunca un resultado real posible), se calcula el argmax de la matriz de
  Dixon-Coles — un número entero concreto (ej. "2-1"), que es lo que el
  usuario pidió explícitamente ("necesito un dato exacto, no un
  aproximado").

## Fuentes de datos (esto costó mucho investigar — no repetir la búsqueda)

| Dato | Fuente | Estado |
|---|---|---|
| Resultados, próximos partidos, 1X2, goles | **football-data.org** (API gratis, `FOOTBALL_DATA_API_KEY`) | ✅ Funcionando |
| Tabla de posiciones, goleadores | football-data.org | ✅ Funcionando (solo informativo) |
| Córners, tiros, tiros al arco, xG, posesión, pases, tiros libres, penales | **Reportes oficiales PDF de FIFA** (`src/connectors/fifa_reports_connector.py`) | ✅ Funcionando, solo para Mundial (`competition == "WC"`) |
| Tarjetas amarillas/rojas | Ninguna fuente gratuita encontrada | ❌ Sin resolver — solo vía CSV manual |
| Cuotas de casas de apuestas | Ninguna API gratis las da | ❌ Descartado — función sigue en el CLI (`--cuotas`) pero no en la web |

**Investigado y descartado** (no perder tiempo revisando de nuevo):
- API-Football: stats (córners/tiros) requieren add-on de pago (~€15-29/mes)
- football-data.co.uk: sitio caído + solo ligas de clubes, no Mundial
- FBref: tiene los datos pero su ToS prohíbe explícitamente construir una
  herramienta con datos scrapeados sin permiso
- WC2026API: sin cupo disponible en plan free
- Statorium ($177 pago único), TheStatsAPI ($50/mes): de pago
- Scraping de apps de consumo (365Scores, SofaScore): descartado a
  propósito — no tienen API pública, y automatizar acceso a su backend
  interno viola sus ToS. Distinto de los reportes de FIFA, que sí son
  publicados públicamente por el organizador del torneo sin restricción
  mencionada.

### Cómo funciona el conector de FIFA (`fifa_reports_connector.py`)

- Descarga la página hub (`fifatrainingcentre.com/.../match-report-hub.php`),
  extrae links a PDFs con regex.
- Solo descarga los PDFs de los 2 equipos del partido consultado (no el
  torneo completo — sería pesado e innecesario).
- Parsea el texto del PDF con `pypdf` y regex sobre las secciones "Key
  Statistics" y "Set Plays".
- **Cruza con el histórico de football-data.org por FECHA, no por nombre
  de equipo** (fuentes distintas pueden deletrear un país distinto, ej.
  "South Korea" vs "Korea Republic" — la fecha es un cruce confiable
  porque un equipo no juega dos partidos oficiales el mismo día).
- Todo cacheado en memoria (índice de links 1h, PDFs parseados
  indefinidamente — un reporte de un partido ya jugado no cambia).
- Nunca lanza excepción hacia arriba: si algo falla, el pipeline sigue
  con los datos que ya tenía.

## Bugs reales encontrados y corregidos (contexto útil si algo se ve raro)

1. **`rho` de Dixon-Coles sin acotar** → probabilidades inválidas (100%
   empate). Fix: bounds `(-0.3, 0.3)`.
2. **OOM en Render** (`SIGKILL`) con competiciones de muchos equipos
   (Mundial ampliado a 48). Causas combinadas: (a) no se limitaba el
   rango de fechas del histórico → se traían demasiados equipos/años;
   (b) OpenBLAS/numpy detectando núcleos del host físico en vez de la
   fracción real asignada al contenedor → memoria disparada. Fix: límite
   de 365 días + `OMP_NUM_THREADS=1` y variables equivalentes al inicio
   de `app.py`.
3. **Goles esperados absurdos** (8.76, luego hasta 24.5 en otro caso) con
   equipos de poco historial → regularización L2 (ver arriba).
4. **XSS en `<title>`** de `wrap_page()` (no escapaba el texto) — lo
   detectó el propio test suite.
5. **`NaN` de pandas en columnas de escudo vacías** → `src="nan"` roto en
   el navegador → normalizado a `None` con `_clean_nan()`.
6. **`Under` redundante** en tablas de over/under (si sabes "Over", "Under"
   es la resta) → simplificado a una sola columna.
7. **OOM/SIGKILL parseando PDFs de FIFA** (`_parse_content_stream` de pypdf
   en el traceback). Los PMSR tienen páginas con gráficos vectoriales
   pesados (mapas de pases, heatmaps); al tokenizar sus content streams
   pypdf explota la RAM y mata al worker. Clave: OOM **no lanza excepción**,
   así que el `try/except` de `_parse_pdf` NO lo atrapaba. Fix
   (`fifa_reports_connector.py`): (a) se salta cada página cuyo content
   stream decodificado supere `_MAX_PAGE_CONTENT_BYTES` (800 KB) antes de
   `extract_text` — el texto de estadísticas vive en páginas de tablas,
   livianas; (b) descarga en streaming con corte si el PDF supera
   `_MAX_PDF_BYTES` (15 MB). **Segunda iteración**: el primer filtro medía
   solo el content stream propio de la página y NO alcanzó — los gráficos
   pesados viven en **Form XObjects** que la página solo referencia, y
   `extract_text` recurre dentro de ellos y los tokeniza (ahí explotaba).
   `_page_is_light` ahora suma también el tamaño de los Form XObjects
   referenciados en `/Resources /XObject` y descarta la página si el total
   supera el umbral. **Tercera iteración (definitiva)**: aun con eso, pypdf
   volvió a explotar pero en OTRO punto (`_cmap._parse_encoding`, parseando
   la tabla de caracteres de una fuente). Conclusión: no se puede acotar el
   OOM por adelantado dentro del mismo proceso — pypdf puede reventar en
   content streams, XObjects, fuentes/CMaps, etc. Solución robusta: el
   parseo del PDF corre en un **subproceso aislado**
   (`_pdf_extract_worker.py`, lanzado con `fork` en Linux/Render). El hijo
   se marca como blanco del OOM killer (`/proc/self/oom_score_adj = 1000`),
   así que si revienta la RAM el kernel mata al subproceso y NO al worker
   web; el padre lo detecta como muerte sin dato y sigue sin esas
   estadísticas. Se usa `fork` (no `spawn`) a propósito: `spawn` re-importa
   el `__main__` del padre, que bajo gunicorn re-ejecutaría el arranque. En
   Windows (dev, sin `fork` y con RAM de sobra) se parsea inline. Los
   filtros de páginas pesadas (XObjects) y de tamaño de descarga se
   mantienen como primera línea de defensa dentro del subproceso.
   **Cuarta iteración (tiempo, no memoria)**: tras aislar la memoria, el
   worker seguía muriendo, pero el traceback reveló otra causa —
   `gunicorn/workers/base.py::handle_abort` = **timeout de gunicorn**, no
   OOM (¡el mensaje "Perhaps out of memory?" de gunicorn es genérico para
   cualquier muerte inesperada del worker y despista!). El padre se quedaba
   bloqueado en `proc.join()` esperando al subproceso, y con varios PDFs por
   partido el tiempo total pasaba los 30s de timeout. Fixes: (a) matar al
   hijo con `proc.kill()` (SIGKILL), no `terminate()` (un hijo atascado en C
   de pypdf ignora SIGTERM); (b) presupuesto por equipo `_TEAM_TIME_BUDGET`
   (18s) + tope `_MAX_PDFS_PER_TEAM` (4) + timeout por PDF `_PDF_PARSE_TIMEOUT`
   (10s) — los PDFs ya cacheados no cuentan contra el reloj; (c) Procfile con
   `gunicorn --worker-class gthread --workers 1 --threads 4 --timeout 90`
   (hilos para que un request lento no bloquee a los demás, y margen sobre
   el presupuesto de parseo). Nota para el futuro: si el free tier sigue
   ajustado, la solución "ideal" es sacar el parseo de PDFs del request y
   precomputar las stats a un JSON cacheado en disco (offline/cron), pero eso
   agrega fricción de refresco de datos durante el Mundial.
   **Quinta iteración**: seguía muriendo por timeout de gunicorn. Dos causas:
   (a) el presupuesto era POR EQUIPO (18s) y hay 2 equipos por request →
   ~36s > 30s. Fix: presupuesto TOTAL por request `_REQUEST_TIME_BUDGET`
   (15s), compartido vía `contextvars.ContextVar` (thread-safe, necesario
   porque gthread atiende varios requests en hilos paralelos y una global
   normal se pisaría). Se arranca con `start_request_budget()` en app.py
   alrededor del bloque WC y se limpia en `finally`. (b) **Render ignora el
   Procfile si hay un "Start Command" configurado en el dashboard del
   servicio** — casi seguro `gunicorn app:app` sin flags, por eso el
   `--timeout 90` del Procfile nunca se aplicó. HAY QUE poner en el dashboard
   de Render (Settings > Start Command):
   `gunicorn app:app --worker-class gthread --workers 1 --threads 4 --timeout 90`

## Decisiones de producto (por qué se ve como se ve)

- **Sin cuotas/value bets en la web** — se sacó del flujo principal
  porque generaba fricción sin aportar (no hay fuente gratis de cuotas
  de todas formas). Sigue en el CLI.
- **Hora siempre en Colombia** (`src/i18n.py::to_colombia_time`), nunca UTC.
- **Nombres de equipo en español** (`TEAM_NAMES_ES` dict en `i18n.py`) —
  cosmético únicamente, el cálculo interno usa el nombre original en
  inglés (necesario para cruzar contra el histórico).
- **Local/visitante siempre explícito** en el título del reporte y en la
  lista de partidos.
- **Lista de partidos agrupada por día** ("Hoy"/"Mañana"/día de semana),
  estilo apps deportivas populares — referencia visual que dio el usuario
  (captura de 365Scores).
- **Diseño visual**: paleta oscura con gradientes, tipografía Inter,
  tarjetas con blur, escudos de equipo — sistema compartido en
  `src/web_style.py` (usado tanto por `app.py` como por `report_html.py`).
- **"Marcador exacto más probable"** además del promedio de goles —
  pedido explícito del usuario ("necesito un dato exacto, no un promedio").
- **Datos oficiales de FIFA como tarjeta separada**, marcada
  explícitamente como informativa — no se mezcla con lo que sí calcula
  probabilidades, para no confundir "dato real de contexto" con "cálculo
  del modelo".

## Variables de entorno necesarias en Render

- `FOOTBALL_DATA_API_KEY` — ya configurada en Render con la key de la
  cuenta del usuario (registrada en football-data.org). Si hay que
  regenerarla: football-data.org > cuenta > API key.
- `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1` — ya
  se fijan por código en `app.py`, pero también se recomiendan como
  variables de entorno en Render como segunda capa de seguridad.

## Cómo desplegar cambios nuevos

```powershell
cd "C:\Users\mtamayo\Downloads\marval (2)\Fut\predictor-futbol"
git add -A
git commit -m "descripción del cambio"
git push
```

Render redespliega automático al detectar el push (está conectado a
GitHub, no es "Direct Upload").

## Pendientes / ideas no implementadas

- Tarjetas amarillas/rojas siguen sin fuente automática (ninguna gratuita
  las tiene). Si aparece alguna fuente nueva, revisar antes que nada que
  su ToS permita uso automatizado (ver tabla de fuentes descartadas
  arriba, para no repetir la investigación).
- El "modo avanzado" (CSV manual) es la única forma de cargar
  córners/tarjetas para ligas que no sean el Mundial (el conector FIFA
  solo aplica a `competition == "WC"`).
- No hay pruebas de carga/estrés en Render — el plan free (512 MB RAM)
  ya mostró ser ajustado; si se agregan más equipos/mercados, vigilar
  memoria de nuevo.

## Cómo correr todo localmente

```powershell
cd "C:\Users\mtamayo\Downloads\marval (2)\Fut\predictor-futbol"
pip install -r requirements.txt
$env:FOOTBALL_DATA_API_KEY = "tu_key_aqui"
python app.py
```

Tests: `python -m pytest tests/ -q` (65 tests al momento de escribir esto,
todos deben pasar).
