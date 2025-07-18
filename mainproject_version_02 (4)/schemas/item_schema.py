from pydantic import BaseModel, Field
from typing import List, Dict, Union, Optional


class item_info_request(BaseModel):
    category: str
    small_category_id: str
    big_category_id: str
    color : str

# 입력용 스키마 (문자열 product_id)
class item_info_input(BaseModel):
    product_id: str
    product_name: str
    image_url: str
    price: int

# 출력용 스키마 (정수 product_id)
class item_info_response(BaseModel):
    product_id: int
    product_name: str
    image_url: str
    price: int


class look_info(BaseModel):
    look_name: str
    look_description: str
    items: Dict[str, Union[item_info_response, None]] = Field(default_factory=dict)
    
    class Config:
        # None 값을 허용하도록 설정
        extra = "allow"  


class item_input_snapshot(BaseModel):
    product_id : str

class item_info_snapshot(BaseModel):
    snap_img_url : List[str]

# 룩 조회 응답 스키마
class look_detail_response(BaseModel):
    look_id: int
    look_name: str
    look_description: str
    items: List[item_info_response]

class user_looks_response(BaseModel):
    looks: List[look_detail_response]