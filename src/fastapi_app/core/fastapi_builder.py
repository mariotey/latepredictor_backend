from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

import logging
from utils.logger import setup_logger

# Logging setup
logger = setup_logger()


def create_fastapi_app():
    app = FastAPI()

    # CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Global validation error handler
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request,
        exc: RequestValidationError
    ):
        body = await request.body()

        logger.error(f"Error: {exc.errors()}")
        logger.error(f"Request body: {body.decode('utf-8') if body else None}")

        return JSONResponse(
            status_code=422,
            content={
                "detail": exc.errors(),
                "body": body.decode("utf-8") if body else None
            }
        )

    return app