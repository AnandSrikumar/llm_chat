from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse


class LLMGenerationError(Exception): ...


class UnsupportedFormatError(Exception): ...

class NotFound(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(message)

async def not_found_handler(
        request: Request,
        exc: NotFound
):
    return JSONResponse(
        status_code=409,
        content={
            "detail": exc.message
        },
    )

async def llm_generation_error_handler(
    request: Request,
    exc: LLMGenerationError,
):
    return JSONResponse(
        status_code=409,
        content={
            "detail": "A response is already being generated for this conversation."
        },
    )


async def unsupported_error_handler(request: Request, exc: UnsupportedFormatError):
    return JSONResponse(
        status_code=415,
        content={"detail": "The file format is not supported"},
    )
