from pydantic import BaseModel
from typing import List

#크롤링 작업의 결과물을 정의할 데이터 클래스
class CrawlingTask(BaseModel):
    category_id: str
    item_code: str
    color: str
    style_name: str
    look_name: str