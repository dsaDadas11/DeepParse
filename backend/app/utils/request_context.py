import re

from fastapi import Header, HTTPException, status

USER_ID_HEADER = "X-User-Id"
USER_ID_PATTERN = re.compile(r"^u_[a-f0-9]{32}$")


def get_current_user_id(
    x_user_id: str | None = Header(default=None, alias=USER_ID_HEADER),
) -> str:
    user_id = (x_user_id or "").strip().lower()
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Missing {USER_ID_HEADER} header.",
        )

    if not USER_ID_PATTERN.fullmatch(user_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid user context.",
        )

    return user_id
