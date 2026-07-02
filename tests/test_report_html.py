from src.report_html import render_html_report


def _fake_summary(media=5.0):
    return {"media": media, "mediana": media, "rango_esperado_p10_p90": [media - 2, media + 2], "desviacion_estandar": 1.5}


def _fake_report():
    return {
        "partido": {"local": "Colombia", "visitante": "Argentina"},
        "1x2": {"local": 0.35, "empate": 0.28, "visitante": 0.37},
        "handicap": {},
        "ambos_anotan": {"si": 0.55, "no": 0.45},
        "goles_esperados": {"local": 1.2, "visitante": 1.4, "total": 2.6},
        "over_under_goles": {2.5: {"over": 0.52, "under": 0.48}},
        "corners": {"local": _fake_summary(5), "visitante": _fake_summary(4), "total": _fake_summary(9)},
        "over_under_corners": {9.5: {"over": 0.5, "under": 0.5}},
        "tiros_al_arco": {"local": _fake_summary(6), "visitante": _fake_summary(5), "total": _fake_summary(11)},
        "over_under_tiros": {8.5: {"over": 0.6, "under": 0.4}},
        "tarjetas": {
            "amarillas_local": _fake_summary(2), "amarillas_visitante": _fake_summary(3),
            "amarillas_total": _fake_summary(5), "rojas_local": _fake_summary(0.1), "rojas_visitante": _fake_summary(0.1),
        },
        "over_under_tarjetas": {3.5: {"over": 0.4, "under": 0.6}},
        "marcador_mas_probable": {"local": 2, "visitante": 1, "probabilidad": 0.18},
    }


def test_render_html_report_contains_team_names():
    report = _fake_report()
    html_doc = render_html_report(report, [])
    assert "Colombia" in html_doc
    assert "Argentina" in html_doc
    assert "<!DOCTYPE html>" in html_doc


def test_render_html_report_escapes_team_names():
    report = _fake_report()
    report["partido"]["local"] = "<script>alert(1)</script>"
    html_doc = render_html_report(report, [])
    assert "<script>alert(1)</script>" not in html_doc
    assert "&lt;script&gt;" in html_doc


def test_render_html_report_includes_value_bets():
    report = _fake_report()
    value_bets = [{
        "mercado": "1X2", "resultado": "local", "probabilidad_modelo": 0.5,
        "probabilidad_implicita": 0.4, "cuota": 2.5, "edge": 0.1, "value_bet": True,
    }]
    html_doc = render_html_report(report, value_bets)
    assert "1X2" in html_doc
    assert "value-bet" in html_doc


def test_render_html_report_shows_no_data_message_when_stats_missing():
    report = _fake_report()
    report["corners"] = None
    report["tiros_al_arco"] = None
    report["tarjetas"] = None
    html_doc = render_html_report(report, [])
    assert html_doc.count("Sin datos suficientes") == 3
    # No debe mostrar un 0.0 falso donde no hay dato
    assert "media 0.0" not in html_doc.lower()


def test_render_html_report_shows_exact_score_and_no_redundant_under_column():
    report = _fake_report()
    html_doc = render_html_report(report, [])
    assert "2-1" in html_doc
    assert "Marcador exacto más probable" in html_doc
    # La tabla de over/under ya no repite "Under" como columna separada
    assert "<th>Under</th>" not in html_doc
    assert "&gt;2.5 goles" in html_doc


def test_render_html_report_marks_side_without_data_instead_of_fake_zero():
    # Un equipo sin historial: su lado va None y el over/under del total
    # se suprime (no debe salir "0.0" como si fuera pronóstico real).
    report = _fake_report()
    report["corners"]["visitante"] = None
    report["corners"]["total"] = None
    report["corners"]["muestras"] = {"local": 12, "visitante": 0}
    report["over_under_corners"] = None
    html_doc = render_html_report(report, [])
    assert "Sin datos en el histórico" in html_doc
    assert "No se calcula" in html_doc
    assert "Muestra:" in html_doc


def test_render_html_report_shows_avisos_banner():
    report = _fake_report()
    report["avisos"] = ["Equipo X no tiene partidos en la competición."]
    html_doc = render_html_report(report, [])
    assert "Avisos sobre los datos" in html_doc
    assert "Equipo X no tiene partidos" in html_doc


def test_tarjeta_quien_clasifica_solo_en_eliminatorias():
    report = _fake_report()
    report["clasificacion_eliminatoria"] = {"local": 0.55, "visitante": 0.45}
    # Sin la marca, no aparece (ligas)
    assert "¿Quién clasifica?" not in render_html_report(report, [])
    # Con la marca, aparece (torneos)
    report["es_eliminatoria"] = True
    html_doc = render_html_report(report, [])
    assert "¿Quién clasifica?" in html_doc
    assert "alargue y penales" in html_doc
