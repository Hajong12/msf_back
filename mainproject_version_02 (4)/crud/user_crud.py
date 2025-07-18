from sqlalchemy.orm import Session
from model.user_model import User, StylingSummary, Item, UserFavoriteItem, Favorite, FavoriteOutfitItem
from fastapi import HTTPException

def get_user_by_id(db : Session, user_id: int):
    return db.query(User).filter(User.id == user_id).first()

def get_user_by_username(db : Session, username: str):
    return db.query(User).filter(User.username == username).first()

def get_styling_summary_by_id(db : Session, user_id: int):
    return db.query(StylingSummary).filter(StylingSummary.user_id == user_id).first()

def update_user_password_in_db(db: Session, user: User, password: str):
    db.query(User).filter(User.id == user.id).update({User.password_hash : password})
    db.commit()
    db.refresh(user)
    return user

def update_user_personal_color_in_db(db: Session, user: User, personal_color_name: str):
    db.query(User).filter(User.id == user.id).update({User.personal_color_name : personal_color_name})
    db.commit()
    db.refresh(user)
    return user

def update_user_in_db(db: Session, user: User, new_data: dict):
    for key, value in new_data.items():
        setattr(user, key, value)
    db.commit()
    db.refresh(user)
    return user

def update_styling_summary_in_db(db: Session, styling_summary: StylingSummary, new_data: dict):
    # None이 아닌 값만 업데이트
    for key, value in new_data.items():
        if value is not None:  # None이 아닌 값만 업데이트
            setattr(styling_summary, key, value)
    db.commit()
    db.refresh(styling_summary)
    return styling_summary

def delete_user_in_db(db: Session, user: User):
    db.delete(user)
    db.commit()
    return user

def delete_styling_summary_in_db(db: Session, styling_summary: StylingSummary):
    db.delete(styling_summary)
    db.commit()
    return styling_summary

#db, 모델 받아서 저장처리
def create_user_in_db(db: Session, db_user: User):
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def create_user_personal_color_in_db(db: Session, user_id : int, personal_color_name : str):
    db_user = db.query(User).filter(User.id == user_id).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다.")
    db.query(User).filter(User.id == user_id).update({User.personal_color_name : personal_color_name})
    db.commit()
    db.refresh(db_user)
    return db_user

def create_styling_summary_in_db(db: Session, db_styling_summary: StylingSummary):
    db.add(db_styling_summary)
    db.commit()
    db.refresh(db_styling_summary)
    return db_styling_summary

# 상품 관련 CRUD 함수들
def create_item_in_db(db: Session, db_item: Item):
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item

def get_item_by_id(db: Session, product_id: int):
    return db.query(Item).filter(Item.product_id == product_id).first()

def create_user_favorite_item_in_db(db: Session, user_id: int, product_id: int):
    # 이미 즐겨찾기에 추가되어 있는지 확인
    existing_favorite = db.query(UserFavoriteItem).filter(
        UserFavoriteItem.user_id == user_id,
        UserFavoriteItem.product_id == product_id
    ).first()
    
    if existing_favorite:
        raise HTTPException(status_code=400, detail="이미 즐겨찾기에 추가된 상품입니다.")
    
    db_favorite = UserFavoriteItem(
        user_id=user_id,
        product_id=product_id
    )
    db.add(db_favorite)
    db.commit()
    db.refresh(db_favorite)
    return db_favorite

def get_user_favorite_items(db: Session, user_id: int):
    return db.query(UserFavoriteItem).filter(UserFavoriteItem.user_id == user_id).all()

def delete_user_favorite_item(db: Session, user_id: int, look_id: int):
    favorite_item = db.query(UserFavoriteItem).filter(
        UserFavoriteItem.user_id == user_id,
        UserFavoriteItem.product_id == look_id
    ).first()
    
    if not favorite_item:
        raise HTTPException(status_code=404, detail="즐겨찾기에서 찾을 수 없는 상품입니다.")
    
    db.delete(favorite_item)
    db.commit()
    return {"message": "즐겨찾기에서 삭제되었습니다."}

# 룩 관련 CRUD 함수들 (기존 Favorite 모델 활용)
def create_look_in_db(db: Session, user_id: int, look_name: str, look_description: str):
    """룩 정보를 Favorite 테이블에 저장"""
    db_look = Favorite(
        user_id=user_id,
        outfit_name=look_name,
        outfit_dev=look_description
    )
    db.add(db_look)
    db.commit()
    db.refresh(db_look)
    return db_look

def create_look_item_in_db(db: Session, favorite_id: int, product_id: int):
    """룩에 포함된 아이템을 FavoriteOutfitItem 테이블에 저장"""
    # 이미 해당 룩에 같은 상품이 추가되어 있는지 확인
    existing_item = db.query(FavoriteOutfitItem).filter(
        FavoriteOutfitItem.favorite_id == favorite_id,
        FavoriteOutfitItem.product_id == product_id
    ).first()
    
    if existing_item:
        # 이미 존재하면 기존 항목 반환 (중복 오류 방지)
        return existing_item
    
    db_look_item = FavoriteOutfitItem(
        favorite_id=favorite_id,
        product_id=product_id
    )
    db.add(db_look_item)
    db.commit()
    db.refresh(db_look_item)
    return db_look_item

def get_user_looks(db: Session, user_id: int):
    """사용자의 모든 룩 조회"""
    return db.query(Favorite).filter(Favorite.user_id == user_id).all()

def get_look_by_id(db: Session, look_id: int):
    """특정 룩 조회"""
    return db.query(Favorite).filter(Favorite.id == look_id).first()

def get_look_items(db: Session, look_id: int):
    """룩에 포함된 아이템들 조회"""
    return db.query(FavoriteOutfitItem).filter(FavoriteOutfitItem.favorite_id == look_id).all()

def delete_look(db: Session, look_id: int):
    """룩 삭제 (CASCADE로 아이템들도 함께 삭제됨)"""
    look = get_look_by_id(db, look_id)
    if not look:
        raise HTTPException(status_code=404, detail="룩을 찾을 수 없습니다.")
    
    db.delete(look)
    db.commit()
    return {"message": "룩이 삭제되었습니다."}
