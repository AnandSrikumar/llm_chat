import logging
import queue
from logging.handlers import QueueHandler, QueueListener

log_queue = queue.Queue(-1)

console_handler = logging.StreamHandler()
console_handler.setFormatter(
    logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
)

queue_handler = QueueHandler(log_queue)

listener = QueueListener(
    log_queue,
    console_handler,
)

_is_initialized = False


def initialize_logging() -> None:
    """Start asynchronous application logging once during app startup."""
    global _is_initialized

    if _is_initialized:
        return

    listener.start()
    _is_initialized = True


def shutdown_logging() -> None:
    """Flush and stop asynchronous application logging during app shutdown."""
    global _is_initialized

    if not _is_initialized:
        return

    listener.stop()
    _is_initialized = False


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)

    logger.setLevel(logging.INFO)

    # Prevent duplicate handlers if get_logger() is called repeatedly
    if not logger.handlers:
        logger.addHandler(queue_handler)
        logger.propagate = False

    return logger
