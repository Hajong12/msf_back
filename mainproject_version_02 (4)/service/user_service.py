from sqlalchemy.orm import Session
from passlib.context import CryptContext
from fastapi import HTTPException
from crud.user_crud import (
    create_user_in_db, create_styling_summary_in_db, get_user_by_id, 
    get_styling_summary_by_id, create_item_in_db, get_item_by_id,
    create_user_favorite_item_in_db, get_user_favorite_items, delete_user_favorite_item,
    create_look_in_db, create_look_item_in_db, get_user_looks, get_look_by_id, get_look_items, delete_look,
    update_user_in_db, update_styling_summary_in_db, delete_user_in_db, delete_styling_summary_in_db, update_user_personal_color_in_db, update_user_password_in_db, get_user_by_username
)
from model.user_model import User, StylingSummary, Item, UserFavoriteItem, Favorite, FavoriteOutfitItem
from schemas.user_schema import UserCreate, user_style_summary, UserUpdate, user_style_summary_update, UserLogin
from schemas.item_schema import item_info_response, look_info

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_user_info(db: Session, user_id: int):
    find_user = get_user_by_id(db, user_id)
    if not find_user:
        raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다.")
    return find_user

def user_create_check(db: Session, username: str):
    find_user = get_user_by_username(db, username)
    if find_user:
        raise HTTPException(status_code=400, detail="이미 존재하는 아이디입니다.")
    return True

def login_user(db: Session, user: UserLogin):
    find_user = get_user_by_username(db, user.username)
    if not find_user:
        raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다.")
    if not verify_password(user.password, find_user.password_hash):
        raise HTTPException(status_code=400, detail="비밀번호가 일치하지 않습니다.")
    return find_user.id

def get_styling_summary_info(db: Session, user_id: int):
    find_styling_summary = get_styling_summary_by_id(db, user_id)
    if not find_styling_summary:
        raise HTTPException(status_code=404, detail="스타일링 요약을 찾을 수 없습니다.")
    return find_styling_summary


#db와 입력 스키마에 담긴 데이터를 검증하고 모델로 변환환
def create_user(db: Session, user: UserCreate):
    
    # 중복데이터 검사
    existing_user = db.query(User).filter(
        (User.email == user.email) | (User.username == user.username)
    ).first()

    # 찾는 값이 있다면 True, 없다면 False
    if existing_user:
        raise HTTPException(status_code=400, detail="이미 존재하는 사용자명 또는 이메일입니다.")

    # 모델 데이터 변환
    db_user = User(
        username=user.username,
        email=user.email,
        name=user.name,
        password_hash=hash_password(user.password)
    )

    return create_user_in_db(db, db_user)

def update_user_password(db: Session, user_id: int, old_password: str, new_password: str):
    find_user = get_user_by_id(db, user_id)
    if not find_user:
        raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다.")
    
    if not verify_password(old_password, find_user.password_hash):
        raise HTTPException(status_code=400, detail="비밀번호가 일치하지 않습니다.")
    
    return update_user_password_in_db(db, find_user, hash_password(new_password))

def update_user_personal_color(db: Session, user_id: int, personal_color_name: str):
    find_user = get_user_by_id(db, user_id)
    if not find_user:
        raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다.")
    return update_user_personal_color_in_db(db, find_user, personal_color_name)

def update_user(db: Session, user_id: int, user_update : UserUpdate):
    find_user = get_user_by_id(db, user_id)
    if not find_user:
        raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다.")
    
    new_data = user_update.model_dump(exclude_unset=True) # 설정되지 않은 필드(None 값) 제외
    return update_user_in_db(db, find_user, new_data)


def update_styling_summary(db: Session, user_id: int, styling_summary_update: user_style_summary_update):
    find_styling_summary = get_styling_summary_by_id(db, user_id)
    if not find_styling_summary:
        raise HTTPException(status_code=404, detail="스타일링 요약을 찾을 수 없습니다.")
    
    new_data = styling_summary_update.model_dump(exclude_unset=True)
    return update_styling_summary_in_db(db, find_styling_summary, new_data)

def delete_user(db: Session, user_id: int):
    find_user = get_user_by_id(db, user_id)
    if not find_user:
        raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다.")
    return delete_user_in_db(db, find_user)

def delete_styling_summary(db: Session, user_id: int):
    find_styling_summary = get_styling_summary_by_id(db, user_id)
    if not find_styling_summary:
        raise HTTPException(status_code=404, detail="스타일링 요약을 찾을 수 없습니다.")
    return delete_styling_summary_in_db(db, find_styling_summary)

def create_styling_summary(db: Session, user_id: int, styling_summary: user_style_summary):
    find_user = get_user_by_id(db, user_id)
    if not find_user:
        raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다.")
    
    db_styling_summary = StylingSummary(
        user_id=user_id,
        budget=styling_summary.budget,
        occasion=styling_summary.occasion,
        height=styling_summary.height,
        gender=styling_summary.gender,
        top_size=styling_summary.top_size,
        bottom_size=styling_summary.bottom_size,
        shoe_size=styling_summary.shoe_size,
        body_feature=styling_summary.body_feature,
        preferred_styles=styling_summary.preferred_styles
    )
    return create_styling_summary_in_db(db, db_styling_summary)

# 상품 관련 서비스 함수들
def create_item(db: Session, item: item_info_response):
    """상품을 데이터베이스에 저장"""
    # 이미 존재하는 상품인지 확인
    existing_item = get_item_by_id(db, item.product_id)
    if existing_item:
        return existing_item
    
    db_item = Item(
        product_id=item.product_id,
        product_name=item.product_name,
        image_url=item.image_url,
        price=item.price
    )
    return create_item_in_db(db, db_item)

def add_favorite_item(db: Session, user_id: int, favorite_item: item_info_response):
    """사용자의 즐겨찾기에 상품 추가"""
    # 사용자 존재 확인
    find_user = get_user_by_id(db, user_id)
    if not find_user:
        raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다.")

    # 상품이 데이터베이스에 없으면 먼저 저장
    item = create_item(db, favorite_item)
    
    # 사용자와 상품 연결 (즐겨찾기 추가)
    return create_user_favorite_item_in_db(db, user_id, item.product_id)

def get_user_favorites(db: Session, user_id: int):
    """사용자의 즐겨찾기 목록 조회"""
    # 사용자 존재 확인
    find_user = get_user_by_id(db, user_id)
    if not find_user:
        raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다.")
    
    # 즐겨찾기 관계 조회
    favorite_relations = get_user_favorite_items(db, user_id)
    
    # 상품 정보와 함께 반환
    favorites = []
    for relation in favorite_relations:
        item = get_item_by_id(db, relation.product_id)
        if item:
            favorites.append(item)
    
    return favorites

def remove_favorite_item(db: Session, user_id: int, product_id: int):
    """사용자의 즐겨찾기에서 상품 제거"""
    # 사용자 존재 확인
    find_user = get_user_by_id(db, user_id)
    if not find_user:
        raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다.")
    
    return delete_user_favorite_item(db, user_id, product_id)




# 룩 관련 서비스 함수들
def create_look(db: Session, user_id: int, look_data: look_info):
    """룩 저장"""
    # 사용자 존재 확인
    find_user = get_user_by_id(db, user_id)
    if not find_user:
        raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다.")

    # 룩 정보 저장
    look = create_look_in_db(db, user_id, look_data.look_name, look_data.look_description)
    
    # 룩에 포함된 아이템들 처리
    for category, item_data in look_data.items.items():
        if item_data is not None:  # None이 아닌 아이템만 처리
            # product_id를 정수로 변환
            item_with_int_id = item_info_response(
                product_id=int(item_data.product_id),
                product_name=item_data.product_name,
                image_url=item_data.image_url,
                price=item_data.price
            )
            
            # 상품이 데이터베이스에 없으면 먼저 저장
            item = create_item(db, item_with_int_id)
            
            # 룩과 상품 연결
            create_look_item_in_db(db, look.id, item.product_id)
    
    return look.id

def get_user_look_list(db: Session, user_id: int):
    """사용자의 룩 목록 조회 (룩 정보 + 아이템들 포함)"""
    # 사용자 존재 확인
    find_user = get_user_by_id(db, user_id)
    if not find_user:
        raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다.")
    
    # 사용자의 모든 룩 조회
    looks = get_user_looks(db, user_id)
    
    # 각 룩에 대한 상세 정보 구성
    look_details = []
    for look in looks:
        # 룩에 포함된 아이템들 조회
        look_items = get_look_items(db, look.id)
        
        # 아이템 정보와 함께 반환
        items = []
        for look_item in look_items:
            item = get_item_by_id(db, look_item.product_id)
            if item:
                items.append(item)
        
        # 룩 정보와 아이템들을 함께 구성
        look_detail = {
            "look_id": look.id,
            "look_name": look.outfit_name,
            "look_description": look.outfit_dev,
            "items": items
        }
        look_details.append(look_detail)
    
    return look_details

def get_look_detail(db: Session, look_id: int):
    """특정 룩의 상세 정보 조회"""
    look = get_look_by_id(db, look_id)
    if not look:
        raise HTTPException(status_code=404, detail="룩을 찾을 수 없습니다.")
    
    # 룩에 포함된 아이템들 조회
    look_items = get_look_items(db, look_id)
    
    # 아이템 정보와 함께 반환
    items = []
    for look_item in look_items:
        item = get_item_by_id(db, look_item.product_id)
        if item:
            items.append(item)
    
    return {
        "look": look,
        "items": items
    }

def delete_user_look(db: Session, look_id: int):
    """룩 삭제"""
    return delete_look(db, look_id)