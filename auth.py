import os
from typing import Optional

from fastapi import Depends, HTTPException, Query
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from dotenv import load_dotenv

load_dotenv()

# Mismo JWT_SECRET que el backend principal: el usuario inicia sesión una
# sola vez ahí y ese token también autoriza contra este servicio.
JWT_SECRET = os.getenv("JWT_SECRET")
JWT_ALGORITHM = "HS256"

# auto_error=False para poder caer al fallback por query param cuando no
# viene el header Authorization (p. ej. <img>/<video src> o WebSocket, que
# no pueden mandar headers personalizados).
security_scheme = HTTPBearer(auto_error=False)


def verificar_token(token: str) -> dict:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except JWTError:
        raise HTTPException(401, "Token inválido o expirado")


def requerir_auth(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_scheme),
    token: Optional[str] = Query(default=None),
) -> dict:
    """
    Dependency de FastAPI/WebSocket: acepta el JWT por header
    Authorization: Bearer (fetch normal) o por query param ?token=...
    (para <img>/<video src> y WebSocket, que no pueden mandar headers).
    """
    if credentials:
        return verificar_token(credentials.credentials)
    if token:
        return verificar_token(token)
    raise HTTPException(401, "No se proporcionó token de autenticación")
