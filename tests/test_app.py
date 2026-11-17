"""Pruebas de la interfaz Gradio sin cargar modelos de IA."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import app
from restaurador.procesamiento import ResultadoPipeline


def test_preparar_editor_prepara_fondo_y_limpia_capas() -> None:
    """Las imágenes subidas se preparan antes de dibujar la máscara."""
    datos_editor = {"background": Image.new("RGB", (1200, 800), "white")}

    editor = app.preparar_editor(datos_editor)

    assert editor is not None
    assert editor["background"].size == (512, 336)
    assert editor["layers"] == []
    assert editor["composite"].size == editor["background"].size


def test_obtener_imagen_y_mascara_extrae_trazo_rojo() -> None:
    """El compuesto del editor se convierte a una máscara binaria alineada."""
    fondo = Image.new("RGB", (16, 16), "white")
    compuesto = fondo.copy()
    for x in range(4, 9):
        for y in range(5, 10):
            compuesto.putpixel((x, y), (255, 0, 0))

    imagen, mascara = app.obtener_imagen_y_mascara(
        {"background": fondo, "composite": compuesto}
    )

    assert imagen is not None
    assert mascara is not None
    assert imagen.size == mascara.size == (16, 16)
    assert np.asarray(mascara)[7, 6] == 255
    assert np.asarray(mascara)[0, 0] == 0


def test_procesar_modo_local_devuelve_resultado_y_archivo() -> None:
    """La interfaz puede ejecutar la demostración y preparar su descarga."""
    fondo = Image.new("RGB", (16, 16), (120, 120, 120))
    compuesto = fondo.copy()
    for x in range(4, 12):
        for y in range(4, 12):
            compuesto.putpixel((x, y), (255, 0, 0))

    salida = app.procesar(
        {"background": fondo, "composite": compuesto},
        app.MODO_BASICO,
        "",
        5,
        0.85,
        42,
    )

    resultado, solicitud, estado = salida[:3]
    archivo = Path(salida[-1])
    try:
        assert isinstance(resultado, ResultadoPipeline)
        assert solicitud
        assert "Proceso terminado" in estado
        assert archivo.is_file()
        with Image.open(archivo) as imagen_descargada:
            assert imagen_descargada.format == "PNG"
    finally:
        archivo.unlink(missing_ok=True)


def test_procesar_rechaza_mascara_vacia() -> None:
    """No se permite iniciar el pipeline si no se marcó una zona."""
    fondo = Image.new("RGB", (16, 16), "white")

    with pytest.raises(app.gr.Error, match="Dibujá con el pincel"):
        app.procesar(
            {"background": fondo, "composite": fondo},
            app.MODO_BASICO,
            "",
            5,
            0.85,
            42,
        )
