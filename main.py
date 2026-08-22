from app.core.config import Settings
from app.app_factory import create_app


settings = Settings()

app = create_app(settings)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )