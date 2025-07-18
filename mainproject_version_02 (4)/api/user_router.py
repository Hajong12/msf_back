from fastapi import APIRouter
from db.user_session import SessionLocal
from schemas.user_schema import UserCreate, UserResponse, user_style_summary, UserUpdate, user_style_summary_update, UserLogin
from schemas.item_schema import item_info_response, look_info, user_looks_response
from service.user_service import (
    create_user, create_styling_summary, get_user_info, get_styling_summary_info,
    add_favorite_item, get_user_favorites, remove_favorite_item,
    create_look, get_user_look_list, get_look_detail, delete_user_look,
    update_user, update_styling_summary, delete_user, delete_styling_summary, update_user_personal_color, update_user_password, user_create_check, login_user
)
from sqlalchemy.orm import Session
from fastapi import Depends

router = APIRouter(prefix="/users", tags=["users"])

# 의존성 주입을 위한 함수
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


@router.get("/user_info", response_model=UserResponse)
def get_user_info_endpoint(user_id: int, db: Session = Depends(get_db)):
    """
    사용자 정보 조회 엔드포인트
    - user_id를 받아서 해당 사용자의 상세 정보를 반환
    - 사용자가 존재하지 않으면 404 에러 반환
    """
    return get_user_info(db, user_id)

@router.get("/styling_summary_info", response_model=user_style_summary)
def get_styling_summary_info_endpoint(user_id: int, db: Session = Depends(get_db)):
    """
    스타일링 요약 정보 조회 엔드포인트
    - user_id를 받아서 해당 사용자의 스타일링 요약 정보를 반환
    - 스타일링 요약이 존재하지 않으면 404 에러 반환
    """
    return get_styling_summary_info(db, user_id)

@router.post("/user_create_check")
def user_create_check_endpoint(username: str, db: Session = Depends(get_db)):
    """
    사용자 생성 전 아이디 중복 확인 엔드포인트
    - username을 받아서 해당 사용자명이 이미 존재하는지 확인
    - 중복 여부를 boolean 값으로 반환
    """
    return user_create_check(db, username)

@router.post("/user_login")
def login_user_endpoint(user: UserLogin, db: Session = Depends(get_db)):
    """
    사용자 로그인 엔드포인트
    - 사용자명과 비밀번호를 받아서 로그인 처리
    - 로그인 성공 시 사용자 정보 반환, 실패 시 에러 반환
    """
    return login_user(db, user)

@router.post("/user_create")
def create_user_endpoint(user: UserCreate, db: Session = Depends(get_db)):
    """
    사용자 생성 엔드포인트
    - 새로운 사용자 정보를 받아서 데이터베이스에 저장
    - 중복된 사용자명이나 이메일이 있으면 400 에러 반환
    """
    return create_user(db, user)

@router.post("/styling_summary_create")
def create_styling_summary_endpoint(styling_summary: user_style_summary, user_id: int, db: Session = Depends(get_db)):
    """
    스타일링 요약 생성 엔드포인트
    - user_id와 스타일링 요약 정보를 받아서 저장
    - 해당 사용자가 존재하지 않으면 404 에러 반환
    """
    return create_styling_summary(db, user_id, styling_summary)


@router.patch("/user_personal_color_update")
def update_user_personal_color_endpoint(user_id: int, personal_color_name: str, db: Session = Depends(get_db)):
    """
    사용자 퍼스널 컬러 수정 엔드포인트
    - user_id와 새로운 퍼스널 컬러명을 받아서 업데이트
    - 해당 사용자가 존재하지 않으면 404 에러 반환
    """
    return update_user_personal_color(db, user_id, personal_color_name)

@router.patch("/user_password_update")
def update_user_password_endpoint(user_id: int, old_password: str, new_password: str, db: Session = Depends(get_db)):
    """
    사용자 비밀번호 수정 엔드포인트
    - user_id, 기존 비밀번호, 새 비밀번호를 받아서 업데이트
    - 기존 비밀번호가 틀리면 에러 반환
    """
    return update_user_password(db, user_id, old_password, new_password)

@router.patch("/user_update")
def update_user_endpoint(user: UserUpdate, user_id: int, db: Session = Depends(get_db)):
    """
    사용자 정보 수정 엔드포인트
    - user_id와 수정할 사용자 정보를 받아서 업데이트
    - 해당 사용자가 존재하지 않으면 404 에러 반환
    """
    return update_user(db, user_id, user)

@router.patch("/styling_summary_update")
def update_styling_summary_endpoint(styling_summary: user_style_summary_update, user_id: int, db: Session = Depends(get_db)):
    """
    스타일링 요약 정보 수정 엔드포인트
    - user_id와 수정할 스타일링 요약 정보를 받아서 업데이트
    - 해당 스타일링 요약이 존재하지 않으면 404 에러 반환
    """
    return update_styling_summary(db, user_id, styling_summary)


@router.delete("/user_delete")
def delete_user_endpoint(user_id: int, db: Session = Depends(get_db)):
    """
    사용자 삭제 엔드포인트
    - user_id를 받아서 해당 사용자와 관련된 모든 데이터를 삭제
    - 해당 사용자가 존재하지 않으면 404 에러 반환
    """
    return delete_user(db, user_id)

@router.delete("/styling_summary_delete")
def delete_styling_summary_endpoint(user_id: int, db: Session = Depends(get_db)):
    """
    스타일링 요약 삭제 엔드포인트
    - user_id를 받아서 해당 사용자의 스타일링 요약을 삭제
    - 해당 스타일링 요약이 존재하지 않으면 404 에러 반환
    """
    return delete_styling_summary(db, user_id)



# 즐겨찾기 관련 엔드포인트들
@router.post("/favorites/add")
def add_favorite_item_endpoint(favorite_item: item_info_response, user_id: int, db: Session = Depends(get_db)):
    """
    사용자의 즐겨찾기에 상품 추가 엔드포인트
    - user_id와 상품 정보를 받아서 즐겨찾기에 추가
    - 해당 사용자가 존재하지 않으면 404 에러 반환
    - 상품이 데이터베이스에 없으면 먼저 저장 후 즐겨찾기 추가
    """
    return add_favorite_item(db, user_id, favorite_item)

@router.get("/favorites")
def get_user_favorites_endpoint(user_id: int, db: Session = Depends(get_db)):
    """
    사용자의 즐겨찾기 목록 조회 엔드포인트
    - user_id를 받아서 해당 사용자의 모든 즐겨찾기 상품 목록을 반환
    - 해당 사용자가 존재하지 않으면 404 에러 반환
    """
    return get_user_favorites(db, user_id)

@router.delete("/favorites/{product_id}")
def remove_favorite_item_endpoint(user_id: int, product_id: int, db: Session = Depends(get_db)):
    """
    사용자의 즐겨찾기에서 상품 제거 엔드포인트
    - user_id와 product_id를 받아서 해당 상품을 즐겨찾기에서 제거
    - 해당 사용자가 존재하지 않으면 404 에러 반환
    """
    return remove_favorite_item(db, user_id, product_id)

# 룩 관련 엔드포인트들
@router.post("/looks/create")
def create_look_endpoint(look_data: look_info, user_id: int, db: Session = Depends(get_db)):
    """
    룩 저장 엔드포인트
    - user_id와 룩 정보를 받아서 데이터베이스에 저장
    - 해당 사용자가 존재하지 않으면 404 에러 반환
    - 룩에 포함된 상품들이 데이터베이스에 없으면 먼저 저장
    """
    return create_look(db, user_id, look_data)

@router.get("/looks", response_model=user_looks_response)
def get_user_looks_endpoint(user_id: int, db: Session = Depends(get_db)):
    """
    사용자의 룩 목록 조회 엔드포인트 (룩 정보 + 아이템들 포함)
    - user_id를 받아서 해당 사용자의 모든 룩 목록을 반환
    - 각 룩에는 포함된 상품들의 정보도 함께 반환
    - 해당 사용자가 존재하지 않으면 404 에러 반환
    """
    looks = get_user_look_list(db, user_id)
    return {"looks": looks}

@router.get("/looks/{look_id}")
def get_look_detail_endpoint(look_id: int, db: Session = Depends(get_db)):
    """
    특정 룩의 상세 정보 조회 엔드포인트
    - look_id를 받아서 해당 룩의 상세 정보와 포함된 상품들을 반환
    - 해당 룩이 존재하지 않으면 404 에러 반환
    """
    return get_look_detail(db, look_id)

@router.delete("/looks/{look_id}")
def delete_look_endpoint(look_id: int, db: Session = Depends(get_db)):
    """
    룩 삭제 엔드포인트
    - look_id를 받아서 해당 룩과 관련된 모든 데이터를 삭제
    - 해당 룩이 존재하지 않으면 404 에러 반환
    """
    return delete_user_look(db, look_id)