"""
i18n.py
=======

Utilidades de presentación en español: traducción de nombres de selección
(la API football-data.org devuelve nombres en inglés) y conversión de
horarios a la zona horaria de Colombia.

Importante: esta traducción es **solo para mostrar en pantalla**. Todo el
cálculo (Dixon-Coles, Elo, la simulación) sigue usando el nombre exacto que
entrega la fuente de datos como identificador interno — si tradujéramos el
nombre antes de eso, dejaríamos de poder cruzarlo contra el histórico (que
también viene en inglés). La traducción se aplica al final, justo antes de
pintar el HTML.
"""

from datetime import datetime
from zoneinfo import ZoneInfo

COLOMBIA_TZ = ZoneInfo("America/Bogota")

# Selecciones nacionales más comunes en competiciones FIFA/UEFA cubiertas
# por el plan gratuito de football-data.org. No pretende ser exhaustivo:
# un nombre no listado simplemente se muestra tal cual viene de la API
# (mejor eso que una traducción inventada o incorrecta).
TEAM_NAMES_ES = {
    "Belgium": "Bélgica", "Senegal": "Senegal", "United States": "Estados Unidos",
    "Bosnia and Herzegovina": "Bosnia y Herzegovina", "Spain": "España",
    "Austria": "Austria", "Portugal": "Portugal", "Croatia": "Croacia",
    "Switzerland": "Suiza", "Algeria": "Argelia", "Australia": "Australia",
    "Egypt": "Egipto", "Argentina": "Argentina", "Cape Verde Islands": "Cabo Verde",
    "Colombia": "Colombia", "Ghana": "Ghana", "Brazil": "Brasil", "Germany": "Alemania",
    "France": "Francia", "England": "Inglaterra", "Italy": "Italia",
    "Netherlands": "Países Bajos", "Uruguay": "Uruguay", "Mexico": "México",
    "Japan": "Japón", "South Korea": "Corea del Sur", "Morocco": "Marruecos",
    "Canada": "Canadá", "Ecuador": "Ecuador", "Peru": "Perú", "Chile": "Chile",
    "Paraguay": "Paraguay", "Bolivia": "Bolivia", "Venezuela": "Venezuela",
    "Wales": "Gales", "Scotland": "Escocia", "Ireland": "Irlanda",
    "Poland": "Polonia", "Sweden": "Suecia", "Norway": "Noruega",
    "Denmark": "Dinamarca", "Serbia": "Serbia", "Turkey": "Turquía",
    "Greece": "Grecia", "Ukraine": "Ucrania", "Russia": "Rusia",
    "Saudi Arabia": "Arabia Saudita", "Qatar": "Catar", "Iran": "Irán",
    "Nigeria": "Nigeria", "Cameroon": "Camerún", "Tunisia": "Túnez",
    "Ivory Coast": "Costa de Marfil", "China": "China", "New Zealand": "Nueva Zelanda",
    "Costa Rica": "Costa Rica", "Panama": "Panamá", "Jamaica": "Jamaica",
    "Honduras": "Honduras", "Iceland": "Islandia", "Finland": "Finlandia",
    "Slovakia": "Eslovaquia", "Slovenia": "Eslovenia", "Hungary": "Hungría",
    "Romania": "Rumania", "Czech Republic": "República Checa",
    "South Africa": "Sudáfrica", "Uzbekistan": "Uzbekistán", "Jordan": "Jordania",
    "Iraq": "Irak", "United Arab Emirates": "Emiratos Árabes Unidos",
    "Trinidad and Tobago": "Trinidad y Tobago", "Curaçao": "Curazao",
    "Haiti": "Haití", "Surinam": "Surinam", "DR Congo": "República Democrática del Congo",
}


def team_name_es(name: str) -> str:
    """Devuelve el nombre en español si lo conocemos, o el original si no."""
    return TEAM_NAMES_ES.get(name, name)


def to_colombia_time(dt: datetime) -> datetime:
    """
    Convierte un datetime (asumido en UTC si no trae zona horaria) a la
    hora de Colombia (UTC-5, sin horario de verano). football-data.org
    entrega todas sus fechas en UTC.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo("UTC"))
    return dt.astimezone(COLOMBIA_TZ)


DIAS_SEMANA_ES = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
MESES_ES = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]


def day_label_es(dt: datetime, reference: datetime = None) -> str:
    """
    Etiqueta de día relativo al estilo de apps deportivas (365Scores,
    SofaScore): "Hoy", "Mañana", "Ayer", o el nombre del día si está más
    lejos. `dt` y `reference` deben estar ya en la misma zona horaria
    (normalmente Colombia, vía to_colombia_time) — comparar fechas en UTC
    directamente daría el día equivocado para partidos de noche.
    """
    if reference is None:
        reference = to_colombia_time(datetime.now(ZoneInfo("UTC")))
    delta_days = (dt.date() - reference.date()).days
    if delta_days == 0:
        return "Hoy"
    if delta_days == 1:
        return "Mañana"
    if delta_days == -1:
        return "Ayer"
    dia_semana = DIAS_SEMANA_ES[dt.weekday()]
    return f"{dia_semana} {dt.day} {MESES_ES[dt.month - 1]}"
