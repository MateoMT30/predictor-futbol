"""
connectors/_pdf_extract_worker.py
=================================

Extracción de texto de un PDF, pensada para correr en un **subproceso
aislado con límite de memoria**.

Por qué existe este módulo aparte: parsear los PDFs de FIFA con pypdf puede
disparar la memoria de forma impredecible (content streams enormes, Form
XObjects con gráficos, o tablas de caracteres/CMap patológicas de alguna
fuente — se vieron OOM reales en `_parse_content_stream` y en
`_cmap._parse_encoding`). Un OOM manda `SIGKILL` al proceso: **no se puede
atrapar con try/except** porque no lanza una excepción de Python, mata el
proceso entero. La única defensa robusta en una caja con poca RAM (Render
free, 512 MB) es hacer el parseo en un subproceso al que se le fija un tope
de memoria: si revienta, muere solo el subproceso y el worker web
sobrevive.

Este módulo se mantiene deliberadamente **liviano** (solo importa `io`; el
resto se importa dentro de la función) para que, cuando el subproceso se
lance con el método "spawn", arranque con la menor huella de memoria
posible y quepa el margen para pypdf.
"""

import io

# Igual que en fifa_reports_connector: se saltan las páginas cuyo "peso"
# (content stream propio + Form XObjects referenciados) supere este umbral,
# porque son las de gráficos pesados que hacen explotar pypdf. Duplicado
# aquí a propósito para no importar el módulo pesado (con pandas) en el
# subproceso liviano.
_MAX_PAGE_CONTENT_BYTES = 800 * 1024


def _page_is_light(page) -> bool:
    try:
        total = 0
        contents = page.get_contents()
        if contents is not None:
            total += len(contents.get_data())

        resources = page.get("/Resources")
        if resources is not None:
            xobjects = resources.get_object().get("/XObject")
            if xobjects is not None:
                for name in xobjects.get_object():
                    try:
                        xobj = xobjects[name].get_object()
                        if xobj.get("/Subtype") == "/Form":
                            total += len(xobj.get_data())
                    except Exception:
                        return False
                    if total > _MAX_PAGE_CONTENT_BYTES:
                        return False

        return total <= _MAX_PAGE_CONTENT_BYTES
    except Exception:
        return False


def extract_light_text(content: bytes) -> str:
    """Devuelve el texto de las páginas "livianas" del PDF (tablas de
    estadísticas). Se procesa página por página y cada página se envuelve
    en try/except (incluido MemoryError, que sí es atrapable cuando el tope
    de memoria del subproceso está fijado con RLIMIT_AS): si una página
    revienta, se salta y se sigue con las demás en vez de perder todo."""
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(content))
    parts = []
    for page in reader.pages:
        try:
            if _page_is_light(page):
                parts.append(page.extract_text() or "")
        except Exception:
            # Incluye MemoryError si el tope de RLIMIT_AS se alcanza en esta
            # página. Se descarta solo esa página y se continúa.
            continue
    return "\n".join(parts)


def _make_self_oom_victim() -> None:
    """En Linux, marca este proceso como el blanco preferido del OOM killer
    (oom_score_adj = 1000). Así, si el parseo del PDF dispara la memoria y
    el cgroup del contenedor llega a su tope, el kernel mata ESTE subproceso
    en vez del worker web del padre. Best-effort: si no se puede (no-Linux,
    permisos), simplemente no se ajusta."""
    try:
        with open("/proc/self/oom_score_adj", "w") as fh:
            fh.write("1000")
    except Exception:
        pass


def _child_main(content: bytes, conn) -> None:
    """Punto de entrada del subproceso (lanzado con 'fork' en Linux/Render).
    Se marca como blanco del OOM killer, extrae el texto y lo envía por el
    pipe. Cualquier fallo -> se envía None. Si el parseo revienta la RAM, el
    kernel mata este proceso (no el padre) y el padre lo detecta como muerte
    sin dato."""
    try:
        _make_self_oom_victim()
        text = extract_light_text(content)
        conn.send(text)
    except Exception:
        try:
            conn.send(None)
        except Exception:
            pass
    finally:
        try:
            conn.close()
        except Exception:
            pass
