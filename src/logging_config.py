# src/logging_config.py
#
# Configuración centralizada del sistema de logging del proyecto.
#
# Por qué usar logging en lugar de print():
#   - Cada mensaje incluye automáticamente la fecha, hora, nivel y módulo de origen.
#   - Los niveles (DEBUG, INFO, WARNING, ERROR) permiten filtrar la verbosidad
#     sin modificar el código: basta con cambiar el nivel en get_logger().
#   - En producción (API Docker) el output va a stdout y puede capturarse por
#     sistemas de monitorización como Loki, CloudWatch o Datadog.
#   - Los mensajes de distintos módulos aparecen con su nombre (train.py, main.py...)
#     lo que facilita depurar de qué parte del pipeline proviene cada traza.

import logging
import sys


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    # Obtener o crear el logger identificado por 'name'.
    # Python reutiliza el mismo objeto logger si ya existe uno con ese nombre,
    # evitando duplicar handlers si get_logger() se llama varias veces.
    logger = logging.getLogger(name)

    # Solo añadir el handler si el logger no tiene ninguno todavía.
    # Sin esta guardia, cada llamada a get_logger() añadiría un handler extra
    # y cada mensaje aparecería duplicado en el output.
    if not logger.handlers:
        # Escribir a stdout (en lugar de stderr) para que los contenedores Docker
        # capturen los logs con 'docker logs' sin separar stdout y stderr.
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            # Formato: "2026-01-15 10:23:45 [INFO] src.train — Época 1/50 ..."
            # El campo %(name)s muestra el módulo que emitió el mensaje,
            # útil para distinguir logs de train.py, main.py, api.py, etc.
            "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logger.addHandler(handler)

    logger.setLevel(level)
    return logger
