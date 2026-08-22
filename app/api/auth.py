from typing import Annotated

from fastapi import APIRouter, Form, HTTPException, status

from app.core.deps import Pg, PASSWORD_MANAGER, JWT_DEP

from app.core.log import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/v1/auth", tags=["auth"])


@router.post("/register")
async def register(
    username: Annotated[str, Form()],
    password: Annotated[str, Form()],
    pg: Pg,
    password_manager: PASSWORD_MANAGER,
):
    password_hash = password_manager.hash(password)

    try:
        row = await pg.fetchone(
            """
            INSERT INTO users (
                username,
                password_hash
            )
            VALUES ($1, $2)
            RETURNING id, username
            """,
            username,
            password_hash,
        )

    except Exception as exc:
        logger.error(f"{exc}")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Username already exists",
        ) from exc

    return {
        "id": row["id"],
        "username": row["username"],
    }


@router.post("/login")
async def login(
    username: Annotated[str, Form()],
    password: Annotated[str, Form()],
    pg: Pg,
    password_manager: PASSWORD_MANAGER,
    security: JWT_DEP,
):
    row = await pg.fetchone(
        """
        SELECT id, username, password_hash
        FROM users
        WHERE username = $1
        """,
        username,
    )

    if row is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )

    if not password_manager.verify(
        password,
        row["password_hash"],
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )

    token = security.create_access_token(username=row["username"], user_id=row["id"])

    return {
        "access_token": token,
        "token_type": "bearer",
    }
