from datetime import datetime, timedelta, timezone

import jwt
from pwdlib import PasswordHash

from app.core.log import get_logger

logger = get_logger(__name__)


class JWT:
    def __init__(self, secret: str, algorithm: str, exp: int = 60):
        self.secret_key = secret
        self.algorithm = algorithm
        self.exp = exp

    def create_access_token(self, username: str, user_id: int):
        logger.info("Creating access token (user_id=%s)", user_id)
        expire = datetime.now(timezone.utc) + timedelta(minutes=self.exp)
        payload = {
            "username": username,
            "id": user_id,
            "exp": expire,
        }
        return jwt.encode(
            payload,
            self.secret_key,
            algorithm=self.algorithm,
        )

    def decode_access_token(self, token: str):
        logger.debug("Decoding access token")
        return jwt.decode(
            token,
            self.secret_key,
            algorithms=[self.algorithm],
        )


class PasswordManager:
    def __init__(self):
        self._password_hash = PasswordHash.recommended()
        logger.info("Password manager initialized")

    def hash(self, password: str) -> str:
        logger.debug("Hashing password")
        return self._password_hash.hash(password)

    def verify(self, password: str, hashed_password: str) -> bool:
        is_valid = self._password_hash.verify(
            password,
            hashed_password,
        )
        logger.debug("Password verification completed (valid=%s)", is_valid)
        return is_valid
