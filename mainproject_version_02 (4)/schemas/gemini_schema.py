from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union

class ItemInfo(BaseModel):
    category: Optional[str] = None
    item_code: Optional[str] = None
    category_id: Optional[str] = None
    color: Optional[str] = None

class LookInfo(BaseModel):
    look_name: str
    look_description: str
    items: Dict[str, Union[ItemInfo, None]] = Field(default_factory=dict)
    
    class Config:
        # None 값을 허용하도록 설정
        extra = "allow"  

class StyleRecommendation(BaseModel):
    style_name: str
    looks: List[LookInfo]

class GeminiExamplePrompt(BaseModel):
    recommendations: List[StyleRecommendation]
    
    
    