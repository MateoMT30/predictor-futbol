"""
scripts/refresh_fifa_cache.py
=============================

Regenera el cache de estadísticas oficiales de FIFA (`data/fifa_cache.json`).

Por qué existe: parsear los PDFs de FIFA cuesta ~8-15s cada uno y es
demasiado pesado para hacerlo dentro de un request en Render free (daba
timeouts y OOM). En vez de eso se precomputan aquí, offline, en un
computador con CPU/RAM normal, y la web solo lee el JSON resultante.

Cómo usarlo:

    cd predictor-futbol
    python scripts/refresh_fifa_cache.py
    git add data/fifa_cache.json
    git commit -m "actualizar cache de datos FIFA"
    git push

Render se redespliega solo con el push y mostrará los datos nuevos. Como un
reporte de un partido YA jugado no cambia, basta re-correrlo cuando se
jueguen partidos nuevos (típicamente después de cada jornada del Mundial).
"""

import os
import sys
import time
from pathlib import Path

# Activar el modo de parseo en vivo ANTES de importar el conector (la bandera
# se lee al importar el módulo). Sin esto, el conector no descargaría ni
# parsearía ningún PDF (que es justo lo que queremos en producción).
os.environ["FIFA_LIVE_PARSE"] = "1"

# Permitir ejecutar el script desde cualquier carpeta: se agrega la raíz del
# repo al path para poder importar `src`.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from src.connectors import fifa_reports_connector as F  # noqa: E402


def main() -> int:
    print("Descargando índice de reportes de FIFA...")
    try:
        links = F._fetch_report_links()
    except Exception as exc:
        print(f"ERROR: no se pudo obtener el índice de reportes: {exc}")
        return 1

    print(f"  {len(links)} PDFs encontrados. Parseando (esto tarda unos minutos)...\n")

    ok = 0
    for i, url in enumerate(links, 1):
        name = url.rsplit("/", 1)[-1]
        t0 = time.time()
        stats = F._parse_pdf(url)
        dt = time.time() - t0
        estado = "OK " if stats else "sin datos"
        print(f"  [{i}/{len(links)}] {estado} ({dt:4.1f}s)  {name}")
        if stats:
            ok += 1

    saved = F.save_disk_cache()
    print(f"\nListo. {ok}/{len(links)} reportes con datos.")
    print(f"Cache guardado en: {F._CACHE_PATH} ({saved} entradas con datos)")
    print("\nAhora sube el cache:")
    print('  git add data/fifa_cache.json')
    print('  git commit -m "actualizar cache de datos FIFA"')
    print("  git push")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
