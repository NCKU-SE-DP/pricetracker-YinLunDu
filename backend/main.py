import json
import sentry_sdk
from apscheduler.schedulers.background import BackgroundScheduler
from fastapi.middleware.cors import CORSMiddleware
import itertools
from sqlalchemy import delete, insert, select
from sqlalchemy.orm import Session, sessionmaker
from typing import List, Optional
import requests
from fastapi import APIRouter, HTTPException, Query, Depends, status, FastAPI
import os
from datetime import datetime, timedelta
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext

from pydantic import BaseModel, Field, AnyHttpUrl
from sqlalchemy import (Column, ForeignKey, Integer, String, Table, Text,
                        create_engine)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker

Base = declarative_base()


user_news_association_table = Table(
    "user_news_upvotes",
    Base.metadata,
    Column("user_id", Integer, ForeignKey("users.id"), primary_key=True),
    Column(
        "news_articles_id", Integer, ForeignKey("news_articles.id"), primary_key=True
    ),
)

# from pydantic import BaseModel


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(50), unique=True, nullable=False)
    hashed_password = Column(String(200), nullable=False)
    upvoted_news = relationship(
        "NewsArticle",
        secondary=user_news_association_table,
        back_populates="upvoted_by_users",
    )


class NewsArticle(Base):
    __tablename__ = "news_articles"
    id = Column(Integer, primary_key=True, autoincrement=True)
    url = Column(String, unique=True, nullable=False)
    title = Column(String, nullable=False)
    time = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    summary = Column(Text, nullable=False)
    reason = Column(Text, nullable=False)
    upvoted_by_users = relationship(
        "User", secondary=user_news_association_table, back_populates="upvoted_news"
    )


engine = create_engine("sqlite:///news_database.db", echo=True)

Base.metadata.create_all(engine)

Session = sessionmaker(bind=engine)

sentry_sdk.init(
    dsn="https://4001ffe917ccb261aa0e0c34026dc343@o4505702629834752.ingest.us.sentry.io/4507694792704000",
    traces_sample_rate=1.0,
    profiles_sample_rate=1.0,
)

app = FastAPI()
background_scheduler = BackgroundScheduler()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

app.add_middleware(
    CORSMiddleware,  # noqa
    allow_origins=["http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import os
from openai import OpenAI


# def generate_summary(content):
#     m = [
#         {
#             "role": "system",
#             "content": "你是一個新聞摘要生成機器人，請統整新聞中提及的影響及主要原因 (影響、原因各50個字，請以json格式回答 {'影響': '...', '原因': '...'})",
#         },
#         {"role": "user", "content": f"{content}"},
#     ]
#
#     completion = OpenAI(api_key="xxx").chat.completions.create(
#         model="gpt-3.5-turbo",
#         messages=m,
#     )
#     return completion.choices[0].message.content

#
# def extract_search_keywords(content):
#     m = [
#         {
#             "role": "system",
#             "content": "你是一個關鍵字提取機器人，用戶將會輸入一段文字，表示其希望看見的新聞內容，請提取出用戶希望看見的關鍵字，請截取最重要的關鍵字即可，避免出現「新聞」、「資訊」等混淆搜尋引擎的字詞。(僅須回答關鍵字，若有多個關鍵字，請以空格分隔)",
#         },
#         {"role": "user", "content": f"{content}"},
#     ]
#
#     completion = OpenAI(api_key="xxx").chat.completions.create(
#         model="gpt-3.5-turbo",
#         messages=m,
#     )
#     return completion.choices[0].message.content


from urllib.parse import quote
import requests
from bs4 import BeautifulSoup
from sqlalchemy.orm import Session

def add_news_article_to_db(news_article_data):
    """
    將新聞文章新增到資料庫
    :param news_article_data: 新聞文章的資訊
    :return:
    """
    session = Session()
    session.add(NewsArticle(
        url=news_article_data["url"],
        title=news_article_data["title"],
        published_time=news_article_data["time"],
        content=" ".join(news_article_data["content"]),  # 將內容列表轉換為字串
        summary=news_article_data["summary"],
        reason_for_inclusion=news_article_data["reason"],
    ))
    session.commit()
    session.close()

def fetch_news_articles(search_keyword, fetch_multiple_pages=False):
    """
    根據搜尋詞獲取新聞文章

    :param search_keyword: 用來搜尋新聞的關鍵字
    :param fetch_multiple_pages: 是否獲取多個頁面的新聞資料
    :return: 包含新聞資料的列表
    """
    all_news_articles = []
    # 獲取多頁面的新聞資料，並不是實際上獲取所有新聞資料
    if fetch_multiple_pages:
        page_results = []
        for page_num in range(1, 10):
            request_params = {
                "page": page_num,
                "id": f"search:{quote(search_keyword)}",
                "channelId": 2,
                "type": "searchword",
            }
            response = requests.get("https://udn.com/api/more", params=request_params)
            page_results.append(response.json()["lists"])

        for page_data in page_results:
            all_news_articles.extend(page_data)
    else:
        request_params = {
            "page": 1,
            "id": f"search:{quote(search_keyword)}",
            "channelId": 2,
            "type": "searchword",
        }
        response = requests.get("https://udn.com/api/more", params=request_params)
        all_news_articles = response.json()["lists"]

    return all_news_articles

def process_and_store_relevant_news(fetch_multiple_pages=False):
    """
    獲取並處理相關的新聞資料，並將符合條件的新聞存入資料庫

    :param fetch_multiple_pages: 是否需要抓取多頁的新聞
    :return:
    """
    news_articles = fetch_news_articles("價格", fetch_multiple_pages=fetch_multiple_pages)
    
    for article in news_articles:
        article_title = article["title"]
        
        relevance_assessment_prompt = [
            {
                "role": "system",
                "content": (
                    "你是一個關聯度評估機器人，請評估新聞標題是否與「民生用品的價格變化」相關，"
                    "並給予 'high'、'medium'、'low' 評價。(僅需回答 'high'、'medium'、'low' 三個詞之一)"
                ),
            },
            {"role": "user", "content": f"{article_title}"},
        ]
        
        ai_response = OpenAI(api_key="xxx").chat.completions.create(
            model="gpt-3.5-turbo",
            messages=relevance_assessment_prompt,
        )
        relevance_rating = ai_response.choices[0].message.content
        
        if relevance_rating == "high":
            article_response = requests.get(article["titleLink"])
            article_soup = BeautifulSoup(article_response.text, "html.parser")
            
            # 提取文章標題與發布時間
            article_title = article_soup.find("h1", class_="article-content__title").text
            publication_time = article_soup.find("time", class_="article-content__time").text
            
            # 提取文章內容
            content_section = article_soup.find("section", class_="article-content__editor")
            article_paragraphs = [
                paragraph.text
                for paragraph in content_section.find_all("p")
                if paragraph.text.strip() != "" and "▪" not in paragraph.text
            ]
            
            detailed_article = {
                "url": article["titleLink"],
                "title": article_title,
                "time": publication_time,
                "content": article_paragraphs,
            }
            
            summary_generation_prompt = [
                {
                    "role": "system",
                    "content": (
                        "你是一個新聞摘要生成機器人，請統整新聞中提及的影響及主要原因 "
                        "(影響、原因各50個字，請以json格式回答 {'影響': '...', '原因': '...'})"
                    ),
                },
                {"role": "user", "content": " ".join(detailed_article["content"])},
            ]
            
            summary_response = OpenAI(api_key="xxx").chat.completions.create(
                model="gpt-3.5-turbo",
                messages=summary_generation_prompt,
            )
            
            summary_result = json.loads(summary_response.choices[0].message.content)
            detailed_article["summary"] = summary_result["影響"]
            detailed_article["reason"] = summary_result["原因"]
            
            # 將處理過的新聞存入資料庫
            add_news_article_to_db(detailed_article)


@app.on_event("startup")
def initialize_news_scheduler():
    """
    初始化新聞爬取排程
    """
    db_session = SessionLocal()
    
    # 如果資料庫中沒有新聞，則進行初始新聞抓取
    if db_session.query(NewsArticle).count() == 0:
        # 可以改用簡單工廠模式來處理不同類型的新聞抓取
        process_and_store_relevant_news()
    
    db_session.close()
    
    # 設置一個每隔100分鐘執行一次的定時任務
    background_scheduler.add_job(process_and_store_relevant_news, "interval", minutes=100)
    background_scheduler.start()


@app.on_event("shutdown")
def shutdown_scheduler():
    background_scheduler.shutdown()


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/users/login")


def session_opener():
    session = Session(bind=engine)
    try:
        yield session
    finally:
        session.close()



def verify(p1, p2):
    return pwd_context.verify(p1, p2)


def check_user_password_is_correct(db_session, username, input_password):
    """
    檢查使用者的密碼是否正確

    :param db_session: 資料庫的 session
    :param username: 使用者的名稱
    :param input_password: 輸入的密碼
    :return: 如果密碼正確，返回使用者物件；否則返回 False
    """
    user = db_session.query(User).filter(User.username == username).first()
    
    if not verify(input_password, user.hashed_password):
        return False
    
    return user


def authenticate_user_token(
    token: str = Depends(oauth2_scheme),
    db_session = Depends(session_opener)
):
    """
    根據 JWT token 認證使用者

    :param token: 用於認證的 JWT token
    :param db_session: 資料庫的 session
    :return: 對應於 token 的使用者，如果找不到則返回 None
    """
    secret_key = '1892dhianiandowqd0n'
    token_payload = jwt.decode(token, secret_key, algorithms=["HS256"])
    
    # 根據 token payload 中的 "sub" 字段（通常是 username）查找使用者
    return db_session.query(User).filter(User.username == token_payload.get("sub")).first()


def create_access_token(data, expires_delta=None):
    """create access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    print(to_encode)
    encoded_jwt = jwt.encode(to_encode, '1892dhianiandowqd0n', algorithm="HS256")
    return encoded_jwt


@app.post("/api/v1/users/login")
async def login_for_access_token(
        form_data: OAuth2PasswordRequestForm = Depends(),
        db_session: Session = Depends(session_opener)
):
    """
    使用者登入並生成訪問令牌 (Access Token)
    """
    # 驗證使用者的帳號和密碼
    user = check_user_password_is_correct(db_session, form_data.username, form_data.password)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # 創建訪問令牌，30分鐘有效期
    access_token = create_access_token(
        data={"sub": str(user.username)},
        expires_delta=timedelta(minutes=30)
    )
    
    return {"access_token": access_token, "token_type": "bearer"}

class UserAuthSchema(BaseModel):
    username: str
    password: str
@app.post("/api/v1/users/register")
def create_user(user_data: UserAuthSchema , db_session: Session = Depends(session_opener)):
    """
    註冊新使用者
    """
    # 將密碼進行哈希處理
    hashed_password = pwd_context.hash(user_data.password)
    
    # 創建新的使用者實體
    new_user = User(username=user_data.username, hashed_password=hashed_password)
    
    # 將新使用者加入資料庫
    db_session.add(new_user)
    db_session.commit()
    db_session.refresh(new_user)
    
    return new_user


@app.get("/api/v1/users/me")
def read_users_me(user=Depends(authenticate_user_token)):
    return {"username": user.username}


_id_counter = itertools.count(start=1000000)


def get_article_upvote_details(article_id, user_id, db_session):
    """
    獲取新聞文章的點贊詳情

    :param article_id: 文章的 ID
    :param user_id: 使用者的 ID (可選)
    :param db_session: 資料庫的 session
    :return: (點贊總數, 當前使用者是否已點贊)
    """
    # 計算該文章的點贊總數
    total_upvotes = (
        db_session.query(user_news_association_table)
        .filter_by(news_articles_id=article_id)
        .count()
    )
    
    # 檢查當前使用者是否已點贊
    has_voted = False
    if user_id:
        has_voted = (
            db_session.query(user_news_association_table)
            .filter_by(news_articles_id=article_id, user_id=user_id)
            .first()
            is not None
        )
    
    return total_upvotes, has_voted


@app.get("/api/v1/news/news")
def read_news(db_session=Depends(session_opener)):
    """
    獲取最新的新聞文章

    :param db_session: 資料庫的 session
    :return: 包含新聞文章及其點贊詳情的列表
    """
    # 獲取所有新聞，按時間倒序排列
    news_articles = db_session.query(NewsArticle).order_by(NewsArticle.time.desc()).all()
    
    # 構建返回的結果集，包含每篇文章的點贊數和是否已點贊
    result = []
    for article in news_articles:
        upvotes, is_upvoted = get_article_upvote_details(article.id, None, db_session)
        result.append(
            {**article.__dict__, "upvotes": upvotes, "is_upvoted": is_upvoted}
        )
    
    return result


@app.get(
    "/api/v1/news/user_news"
)
def read_user_news(
    db_session=Depends(session_opener),
    user=Depends(authenticate_user_token)
):
    """
    Retrieve news articles for a user, including upvote details.

    :param db_session: Database session dependency
    :param user: Authenticated user from token
    :return: List of news articles with upvote details
    """
    news_articles = db_session.query(NewsArticle).order_by(NewsArticle.time.desc()).all()
    articles_with_upvotes = []

    for article in news_articles:
        upvote_count, is_user_upvoted = get_article_upvote_details(article.id, user.id, db_session)
        articles_with_upvotes.append(
            {
                **article.__dict__,
                "upvotes": upvote_count,
                "is_upvoted": is_user_upvoted,
            }
        )

    return articles_with_upvotes

class PromptRequest(BaseModel):
    prompt: str

@app.post("/api/v1/news/search_news")
async def search_news(request: PromptRequest):
    user_prompt = request.prompt
    extracted_news_list = []
    prompt_messages = [
        {
            "role": "system",
            "content": "你是一個關鍵字提取機器人，用戶將會輸入一段文字，表示其希望看見的新聞內容，請提取出用戶希望看見的關鍵字，請截取最重要的關鍵字即可，避免出現「新聞」、「資訊」等混淆搜尋引擎的字詞。(僅須回答關鍵字，若有多個關鍵字，請以空格分隔)",
        },
        {"role": "user", "content": f"{user_prompt}"},
    ]

    ai_completion = OpenAI(api_key="xxx").chat.completions.create(
        model="gpt-3.5-turbo",
        messages=prompt_messages,
    )
    extracted_keywords = ai_completion.choices[0].message.content
    
    # Fetch news articles based on extracted keywords
    relevant_news_items = fetch_news_articles(extracted_keywords, is_initial=False)
    for news_item in relevant_news_items:
        try:
            news_response = requests.get(news_item["titleLink"])
            news_soup = BeautifulSoup(news_response.text, "html.parser")
            
            # Extract title and time
            news_title = news_soup.find("h1", class_="article-content__title").text
            news_time = news_soup.find("time", class_="article-content__time").text
            
            # Locate the section containing the article content
            news_content_section = news_soup.find("section", class_="article-content__editor")
            content_paragraphs = [
                paragraph.text
                for paragraph in news_content_section.find_all("p")
                if paragraph.text.strip() != "" and "▪" not in paragraph.text
            ]
            
            # Create detailed news dictionary
            detailed_news_info = {
                "url": news_item["titleLink"],
                "title": news_title,
                "time": news_time,
                "content": content_paragraphs,
            }
            detailed_news_info["content"] = " ".join(detailed_news_info["content"])
            detailed_news_info["id"] = next(_id_counter)
            extracted_news_list.append(detailed_news_info)
        except Exception as error:
            print(error)
    
    return sorted(extracted_news_list, key=lambda x: x["time"], reverse=True)

class NewsSumaryRequestSchema(BaseModel):
    content: str

@app.post("/api/v1/news/news_summary")
async def news_summary(
        payload: NewsSumaryRequestSchema, u=Depends(authenticate_user_token)
):
    response = {}
    m = [
        {
            "role": "system",
            "content": "你是一個新聞摘要生成機器人，請統整新聞中提及的影響及主要原因 (影響、原因各50個字，請以json格式回答 {'影響': '...', '原因': '...'})",
        },
        {"role": "user", "content": f"{payload.content}"},
    ]

    completion = OpenAI(api_key="xxx").chat.completions.create(
        model="gpt-3.5-turbo",
        messages=m,
    )
    result = completion.choices[0].message.content
    if result:
        result = json.loads(result)
        response["summary"] = result["影響"]
        response["reason"] = result["原因"]
    return response


@app.post("/api/v1/news/{id}/upvote")
def upvote_article(
        article_id,
        db_session=Depends(session_opener),
        user=Depends(authenticate_user_token),
):
    upvote_status_message = toggle_article_upvote(article_id, user.id, db_session)
    return {"message": upvote_status_message}


def toggle_article_upvote(article_id, user_id, db_session):
    # 檢查是否已存在用戶對該文章的 upvote
    existing_upvote_record = db_session.execute(
        select(user_news_association_table).where(
            user_news_association_table.c.news_articles_id == article_id,
            user_news_association_table.c.user_id == user_id,
        )
    ).scalar()

    if existing_upvote_record:
        # 如果已存在，則移除 upvote
        delete_upvote = delete(user_news_association_table).where(
            user_news_association_table.c.news_articles_id == article_id,
            user_news_association_table.c.user_id == user_id,
        )
        db_session.execute(delete_upvote)
        db_session.commit()
        return "Upvote removed"
    else:
        # 如果不存在，則添加 upvote
        add_upvote = insert(user_news_association_table).values(
            news_articles_id=article_id, user_id=user_id
        )
        db_session.execute(add_upvote)
        db_session.commit()
        return "Article upvoted"


def news_exists(id2, db: Session):
    return db.query(NewsArticle).filter_by(id=id2).first() is not None


@app.get("/api/v1/prices/necessities-price")
def get_necessities_prices(
        category=Query(None), commodity=Query(None)
):
    return requests.get(
        "https://opendata.ey.gov.tw/api/ConsumerProtection/NecessitiesPrice",
        params={"CategoryName": category, "Name": commodity},
    ).json()
