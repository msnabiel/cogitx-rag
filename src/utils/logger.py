"""Logging configuration"""

import os
import logging
import time

def setup_logging(log_dir: str = "logs", log_file: str = "app.log", level: int = logging.INFO):
    """
    Setup logging configuration

    Args:
        log_dir: Directory for log files
        log_file: Log file name
        level: Logging level
    """
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, log_file), mode='a'),
            logging.StreamHandler()
        ],
        force=True
    )

    logger = logging.getLogger("app")
    logger.info("=== LOGGING SYSTEM INITIALIZED ===")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Log level: {logging.getLevelName(logger.level)}")

    return logger


def get_logger(name: str = "app") -> logging.Logger:
    """Get logger instance"""
    return logging.getLogger(name)


async def log_request_middleware(request, call_next):
    """Request/response logging middleware"""
    logger = get_logger()

    logger.info(f"=== INCOMING REQUEST ===")
    logger.info(f"Method: {request.method}")
    logger.info(f"URL: {request.url}")
    logger.info(f"Path: {request.url.path}")
    logger.info(f"Query params: {request.query_params}")

    headers_dict = dict(request.headers)
    safe_headers = {
        k: v for k, v in headers_dict.items()
        if k.lower() not in ['authorization', 'cookie', 'x-api-key']
    }
    logger.info(f"Headers: {safe_headers}")

    try:
        body = await request.body()
        if body:
            body_str = body.decode('utf-8')
            if len(body_str) > 1000:
                body_str = body_str[:1000] + "... [truncated]"
            logger.info(f"Request Body: {body_str}")
    except Exception as e:
        logger.warning(f"Could not read request body: {e}")

    start_time = time.time()
    response = await call_next(request)
    processing_time = time.time() - start_time

    logger.info(f"=== RESPONSE ===")
    logger.info(f"Status Code: {response.status_code}")
    logger.info(f"Processing Time: {processing_time:.3f}s")

    response_headers = dict(response.headers)
    safe_response_headers = {
        k: v for k, v in response_headers.items()
        if k.lower() not in ['set-cookie']
    }
    logger.info(f"Response Headers: {safe_response_headers}")

    try:
        if hasattr(response, 'body') and response.body:
            response_body = response.body.decode('utf-8')
            if len(response_body) > 2000:
                response_body = response_body[:2000] + "... [truncated]"
            logger.info(f"Response Body: {response_body}")
    except Exception as e:
        logger.debug(f"Could not log response body: {e}")

    logger.info(f"=== END REQUEST/RESPONSE ===\n")
    return response


__all__ = ['setup_logging', 'get_logger', 'log_request_middleware']
