from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse


class LLMGenerationError(Exception): ...


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
