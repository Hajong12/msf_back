from fastapi import APIRouter, HTTPException
from schemas.item_schema import item_info_request, item_info_response, item_info_snapshot, item_input_snapshot, look_info
from schemas.personal_schema import PersonalColorResponse, PersonalColorAnalysis
from schemas.user_schema import user_style_summary, user_profile
from schemas.gemini_schema import GeminiExamplePrompt
from service.crowling_service import crowling_item_snap, category_codes
from service.gemini_service import analyze_personal_color, structured_personal_color_analysis, extract_crawling_tasks
from typing import Optional, List
from db.user_session import SessionLocal
from sqlalchemy.orm import Session
from fastapi import Depends

router = APIRouter(prefix="/gemini", tags=["gemini"])


def get_db():
    """
    데이터베이스 세션을 생성하고 관리하는 의존성 함수
    - 세션을 생성하고 요청이 완료되면 자동으로 닫힘
    - FastAPI의 Depends를 통해 자동으로 주입됨
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/analyze-color", response_model=str)
async def analyze_personal_color_endpoint(face_color: PersonalColorResponse):
    """
    퍼스널 컬러 분석을 수행하는 엔드포인트
    - face_color: 분석할 얼굴 색상 정보
    - Gemini API를 사용하여 퍼스널 컬러 분석 결과를 텍스트로 반환
    - 분석 중 오류가 발생하면 500 에러 반환
    """
    try:
        result = await analyze_personal_color(face_color)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"분석 중 오류가 발생했습니다: {str(e)}")

@router.post("/analyze-structured", response_model=GeminiExamplePrompt) #gemini 출력확인용 함수
async def analyze_structured_personal_color(
    user_id: int,
    db: Session = Depends(get_db)
):
    """
    구조화된 퍼스널 컬러 분석을 수행하는 엔드포인트
    - user_id: 분석할 사용자 ID
    - db: 데이터베이스 세션
    - 사용자 스타일 정보와 프로필을 포함한 종합 분석을 제공
    - Gemini API를 통해 구조화된 추천 결과를 반환
    - 분석 중 오류가 발생하면 500 에러 반환
    """
    try:
        result = await structured_personal_color_analysis(user_id, db)
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"구조화된 분석 중 오류가 발생했습니다: {str(e)}")










