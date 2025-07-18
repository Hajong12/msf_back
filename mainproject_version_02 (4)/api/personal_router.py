from fastapi import APIRouter
from typing import List
from fastapi import UploadFile
from fastapi.responses import FileResponse
from service.facecolor_service import main, extract_face_only
from service.gemini_service import analyze_personal_color
from schemas.personal_schema import FaceColor, PersonalColorResponse
import json
from fastapi import Form, Response
from sqlalchemy.orm import Session
from fastapi import Depends
from db.user_session import SessionLocal
import os
from fastapi import HTTPException

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

router = APIRouter(prefix="/personal", tags=["personal"])

@router.post("/facecolor", response_model=FaceColor)
async def extract_face_color(file: UploadFile):
    """
    이미지에서 얼굴 부위별 색상을 추출하는 엔드포인트
    - file: 분석할 이미지 파일 (UploadFile)
    - 얼굴 부위별 색상 정보를 추출하여 FaceColor 모델로 반환
    - 이미지 처리 중 오류가 발생할 수 있음
    """
    face_color = await main(file)
    return face_color

@router.post("/analyze-all" , response_model=PersonalColorResponse)
async def analyze_face_all(file: UploadFile, user_id: int, db: Session = Depends(get_db)):
    """
    이미지를 받아서 색상 추출부터 퍼스널 컬러 분석까지 한 번에 처리하는 엔드포인트
    - file: 분석할 이미지 파일 (UploadFile)
    - user_id: 분석할 사용자 ID
    - db: 데이터베이스 세션
    
    처리 과정:
    1. 얼굴 색상 추출: 이미지에서 얼굴 부위별 색상 정보 추출
    2. 퍼스널 컬러 분석: Gemini API를 통한 퍼스널 컬러 분석
    3. 통합 결과 반환: PersonalColorResponse 형태로 분석 결과 반환
    
    오류 발생 시 에러 메시지를 포함한 PersonalColorResponse 반환
    """
    try:
        face_color = await main(file)
        analysis_text = await analyze_personal_color(face_color, user_id, db)
        return PersonalColorResponse(personal_color_analysis=analysis_text)
    except HTTPException as e:
        # HTTPException의 detail만 반환
        return PersonalColorResponse(personal_color_analysis=f"{e.detail}")
    except Exception as e:
        return PersonalColorResponse(personal_color_analysis=f"분석 중 오류가 발생했습니다: {str(e)}")
    


@router.post("/extract-face-image")
async def extract_face_image_endpoint(file: UploadFile):
    """
    얼굴만 추출한 이미지를 직접 반환하는 엔드포인트
    - file: 분석할 이미지 파일 (UploadFile)
    - 얼굴만 추출한 PNG 이미지를 직접 반환
    - 투명 배경으로 얼굴만 잘라낸 이미지
    """
    try:
        # 얼굴 추출 수행 (바이트 데이터 반환)
        image_bytes = await extract_face_only(file)
        
        # 이미지 바이트를 직접 반환
        from fastapi.responses import Response
        return Response(
            content=image_bytes,
            media_type="image/png",
            headers={"Content-Disposition": "inline; filename=face_only_image.png"}
        )
    except HTTPException as e:
        # 서비스 레이어에서 발생한 HTTPException을 그대로 다시 발생시킴
        raise e
    except Exception as e:
        # 예상치 못한 다른 모든 예외는 500 에러로 처리
        raise HTTPException(
            status_code=500,
            detail=f"이미지 추출 중 예상치 못한 오류가 발생했습니다: {str(e)}"
        )
