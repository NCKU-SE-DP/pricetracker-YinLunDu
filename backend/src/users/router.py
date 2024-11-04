from datetime import timedelta
from fastapi import APIRouter, Depends
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from ..auth.models import User
from .schemas import UserAuthSchema
from ..database import session_opener
from ..auth.service import authenticate_user_token, check_user_password_is_correct, create_access_token, pwd_context
from ..config import Config

router = APIRouter(
    prefix="/users",
    tags=["users"],
    responses={404: {"description": "Not found"}},
)

@router.get("/me")
def read_users_me(user=Depends(authenticate_user_token)):
    return {"username": user.username}

@router.post("/login")
async def login_for_access_token(
        form_data: OAuth2PasswordRequestForm = Depends(),
        db_session: Session = Depends(session_opener)
):
    """
    使用者登入並生成訪問令牌 (Access Token)
    """
    # 驗證使用者的帳號和密碼
    user = check_user_password_is_correct(db_session, form_data.username, form_data.password)

    # 創建訪問令牌，30分鐘有效期
    access_token = create_access_token(
        data={"sub": str(user.username)},
        expires_delta=timedelta(minutes=Config.News.NEWS_FETCH_INTERVAL_TIME)
    )
    
    return {"access_token": access_token, "token_type": "bearer"}

@router.post("/register")
def create_user(user_data: UserAuthSchema , db_session: Session = Depends(session_opener)):
    """
    註冊新使用者
    """
    hashed_password = pwd_context.hash(user_data.password)
    new_user = User(username=user_data.username, hashed_password=hashed_password)
    
    db_session.add(new_user)
    db_session.commit()
    db_session.refresh(new_user)
    return new_user