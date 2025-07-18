from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from api.personal_router import router as personal_router
from api.user_router import router as user_router
from api.crawling_router import router as crawling_router
from api.gemini_router import router as gemini_router
app = FastAPI(title="퍼스널 컬러 분석 API", description="얼굴 이미지로 퍼스널 컬러를 분석합니다")

# 정적 파일 서빙
app.mount("/static", StaticFiles(directory="static"), name="static")

# 라우터 등록
app.include_router(personal_router)
app.include_router(user_router)
app.include_router(crawling_router)
app.include_router(gemini_router)

@app.get("/")
async def read_index():
    return FileResponse('static/index.html')