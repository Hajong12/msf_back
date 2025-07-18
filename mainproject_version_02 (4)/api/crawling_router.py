from fastapi import APIRouter, HTTPException
from schemas.item_schema import item_info_request, item_info_response, item_info_snapshot, item_input_snapshot, look_info
from schemas.user_schema import user_style_summary, user_profile
from schemas.gemini_schema import GeminiExamplePrompt
from service.gemini_service import extract_crawling_tasks, structured_personal_color_analysis
from service.crowling_service import crowling_item_snap, category_codes, process_and_group_crawling_tasks
from typing import Optional, List
from db.user_session import SessionLocal
from sqlalchemy.orm import Session
from fastapi import Depends

router = APIRouter(prefix="/crawling", tags=["crawling"])

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


@router.post("/analyze-item", response_model=List[look_info])
async def analyze_structured_personal_color(
    user_id : int,
    filter : int,
    db : Session = Depends(get_db)
):
    """
    구조화된 퍼스널 컬러 분석을 통한 상품 추천 및 크롤링 엔드포인트
    - user_id: 분석할 사용자 ID
    - filter: 필터링 옵션 (상품 색 넣을지 안넣을지지)
    - db: 데이터베이스 세션
    
    처리 과정:
    1. Gemini API를 통한 사용자 맞춤 상품 추천 분석
    2. 추천 결과를 크롤링 태스크로 변환
    3. 각 태스크에 대해 실제 상품 크롤링 수행
    4. 크롤링된 상품들을 룩 형태로 그룹화하여 반환
    """
    try:
        result = await structured_personal_color_analysis(user_id, db)
        print(f"Gemini API result type: {type(result)}")
        print(f"Gemini API result: {result}")
        
    except Exception as e:
        print(f"Error in structured_personal_color_analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"구조화된 분석 중 오류가 발생했습니다: {str(e)}")
        

    try:
        parsed_recommendations = GeminiExamplePrompt.model_validate(result)
        print(f"Parsed recommendations: {parsed_recommendations}")
    except Exception as e:
        print(f"Error parsing recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"응답 파싱 중 오류가 발생했습니다: {str(e)}")
    
    tasks_as_objects = extract_crawling_tasks(parsed_recommendations)

    # 디버깅을 위한 로그 추가
    print(f"Extracted tasks count: {len(tasks_as_objects)}")
    for i, task in enumerate(tasks_as_objects):
        print(f"Task {i}: category_id={task.category_id}, item_code={task.item_code}, look_name={task.look_name}")
    
    # Gemini API 결과에서 look_description 정보를 가져오기 위한 매핑
    look_descriptions = {}
    for recommendation in parsed_recommendations.recommendations:
        for look in recommendation.looks:
            look_descriptions[look.look_name] = look.look_description
    
    look_info_list = await process_and_group_crawling_tasks(
        tasks_as_objects, user_id, db, look_descriptions, filter
    )
    
    print(f"Final result: {len(look_info_list)} look_info objects")
    return look_info_list




@router.get("/my_page/{product_id}/snap", response_model=item_info_snapshot)
async def get_item_snap(product_id: str):
    """
    무신사 상품 스냅 정보를 크롤링하여 반환하는 엔드포인트
    - product_id: 크롤링할 상품의 ID
    - 상품이 존재하지 않으면 404 에러 반환
    - 크롤링 중 오류가 발생하면 500 에러 반환
    """
    try:
        result = crowling_item_snap(product_id)
        
        if result is None:
            raise HTTPException(status_code=404, detail="상품을 찾을 수 없습니다.")
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"크롤링 중 오류가 발생했습니다: {str(e)}")
