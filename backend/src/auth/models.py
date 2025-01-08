from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import relationship
from ..database import Base, user_news_association_table
from ..config import Config
from ..logger import logger
from ..exceptions_handler import (
    InvalidUsernameSizeException,
    InvalidPasswordSizeException
)

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(Config.Auth.MAX_USERNAME_SIZE), unique=True, nullable=False)
    hashed_password = Column(String(Config.Auth.MAX_PASSWORD_SIZE), nullable=False)
    upvoted_news = relationship(
        "NewsArticle",
        secondary=user_news_association_table,
        back_populates="upvoted_by_users",
    )
    
    @staticmethod
    def validate_username(username: str):
        if len(username) > Config.Auth.MAX_USERNAME_SIZE:
            logger.error(f"User login failed: {InvalidUsernameSizeException(len(username))}")
            raise InvalidUsernameSizeException(len(username))
    
    @staticmethod
    def validate_password(password: str):
        if len(password) > Config.Auth.MAX_PASSWORD_SIZE:
            logger.error(f"User login failed: {InvalidUsernameSizeException(len(password))}")
            raise InvalidPasswordSizeException(len(password))