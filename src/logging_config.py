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
#
# Destinos de escritura:
#   1. stdout  — visible en consola y capturado por Docker/systemd.
#   2. logs/app.log — fichero persistente con rotación automática por tamaño.
#      El directorio logs/ se crea automáticamente si no existe.
#      Los ficheros de log están excluidos de git (ver .gitignore: logs/).

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler

# Ruta del fichero de log relativa al directorio de trabajo del proceso.
# En local: <raíz del proyecto>/logs/app.log
# En Docker: /app/logs/app.log  (WORKDIR /app en el Dockerfile)
_LOG_DIR = Path("logs")
_LOG_FILE = _LOG_DIR / "app.log"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    # Obtener o crear el logger identificado por 'name'.
    # Python reutiliza el mismo objeto logger si ya existe uno con ese nombre,
    # evitando duplicar handlers si get_logger() se llama varias veces.
    logger = logging.getLogger(name)

    # Solo añadir los handlers si el logger no tiene ninguno todavía.
    # Sin esta guardia, cada llamada a get_logger() añadiría handlers extra
    # y cada mensaje aparecería duplicado tanto en consola como en fichero.
    if not logger.handlers:
        # Formato compartido por consola y fichero: garantiza que ambas salidas
        # sean idénticas y comparables sin necesidad de conversión.
        formatter = logging.Formatter(
            # Formato: "2026-01-15 10:23:45 [INFO] src.train — Época 1/50 ..."
            # El campo %(name)s muestra el módulo que emitió el mensaje,
            # útil para distinguir logs de train.py, main.py, api.py, etc.
            "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Handler 1: consola (stdout).
        # Se usa stdout en lugar de stderr para que los contenedores Docker
        # capturen los logs con 'docker logs' sin separar stdout y stderr.
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # Handler 2: fichero con rotación automática por tamaño.
        # RotatingFileHandler rota el fichero cuando alcanza maxBytes, creando
        # un backup numerado (app.log.1, app.log.2, ...) y comenzando uno nuevo.
        # Parámetros elegidos para un proyecto de investigación local:
        #   maxBytes=5 MB:  un entrenamiento completo genera ~500 líneas (~50 KB);
        #                   5 MB cubre cientos de entrenamientos sin rotar con
        #                   frecuencia excesiva.
        #   backupCount=3:  conserva app.log + 3 backups → máximo ~20 MB en disco.
        #   encoding=utf-8: necesario para caracteres españoles (ñ, é, á, etc.)
        #                   que aparecen en los mensajes de log del proyecto.
        #   mode='a' (por defecto): los logs se acumulan entre reinicios;
        #                   no se sobreescribe el fichero al arrancar de nuevo.
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            _LOG_FILE,
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.setLevel(level)
    return logger
