from pydantic import BaseModel
from typing import List

class FaceColor(BaseModel):
    eyes: List[str]
    nose: List[str]
    lips: List[str]
    hair: List[str]
    skin: List[str]

class PersonalColorAnalysis(BaseModel):
    personal_color_type: str
    personal_color_type_not: str
    skin_type_analysis: str
    PCCS_Tone_circle: List[str]
    Hair_color_hex: List[str]
    Hair_color_name: List[str]
    Hair_tone: str
    Accessory_color_recommendation: List[str]
    makeup_tips: str

class PersonalColorResponse(BaseModel):
    personal_color_analysis: str

