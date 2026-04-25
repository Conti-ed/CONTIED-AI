import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .models import ContiRequest, ContiResponse
from . import services

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log", encoding="utf-8") # 로그 파일 한글 깨짐 방지
    ]
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def contextmanager_lifespan(app: FastAPI):
    # 시작 시: AI 서비스 및 데이터 초기화
    logger.info("AI 서비스를 초기화하는 중입니다...")
    try:
        await services.initialize()
        logger.info("모든 서비스가 성공적으로 초기화되었습니다.")
    except Exception as e:
        logger.error(f"서비스 초기화 중 오류가 발생했습니다: {e}")
        # AI 초기화에 실패하더라도 서버는 시작할 수 있도록 처리
    yield
    # 종료 시: 자원 정리 필요 시 수행
    logger.info("API 서버를 종료합니다...")

app = FastAPI(
    title="CONTIED AI API",
    description="예배 인도자를 위한 AI 기반 찬양 콘티 추천 서비스",
    version="2.0.0",
    lifespan=contextmanager_lifespan
)

# CORS 설정 (프론트엔드 연동용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "CONTIED-AI 서버가 정상 작동 중입니다.", "version": "2.0.0"}

@app.post("/api/conti", response_model=ContiResponse)
async def create_conti(request: ContiRequest):
    logger.info(f"추천 요청 수신 - 키워드: {request.keywords}, 말씀 범위: {request.bible_verse_range}")
    
    try:
        result = await services.create_recommendation(
            keywords=request.keywords,
            bible_range=request.bible_verse_range,
            seed=request.seed,
        )
        
        if "error" in result:
            logger.error(f"추천 생성 중 오류 발생: {result['error']}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result["error"]
            )
            
        return ContiResponse(**result)
        
    except Exception as e:
        logger.exception(f"콘티 생성 중 예기치 않은 오류가 발생했습니다: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="서버 내부 오류로 인해 추천을 생성할 수 없습니다."
        )

if __name__ == "__main__":
    import uvicorn
    # 외부 접속 허용을 위해 0.0.0.0으로 실행
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
