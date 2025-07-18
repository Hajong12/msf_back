import google.generativeai as genai
import os
import instructor
from schemas.personal_schema import FaceColor, PersonalColorAnalysis, PersonalColorResponse
from schemas.user_schema import user_style_summary, user_profile
from schemas.gemini_schema import GeminiExamplePrompt
from schemas.crowling_schema import CrawlingTask
from crud.user_crud import get_styling_summary_by_id, get_user_by_id, create_user_personal_color_in_db
from dotenv import load_dotenv
from sqlalchemy.orm import Session
from typing import Dict, Any, Optional, List
import logging
import sys
import os

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 로깅 설정
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter(u'%(asctime)s [%(levelname)8s] %(message)s')
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(formatter)
logger.addHandler(consoleHandler)


# .env 파일 로드
load_dotenv()

# Gemini API 설정
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

class GeminiColorConsultant:
    def __init__(self):
        """
        Gemini API 초기화
        
        Gemini API 키를 설정하고 텍스트 모델과 구조화된 출력 모델을 초기화합니다.
        """
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY 환경변수가 설정되지 않았습니다.")
        
        genai.configure(api_key=GEMINI_API_KEY)
        # 일반 텍스트용 모델
        self.text_model = genai.GenerativeModel('gemini-2.5-flash')
        # 구조화된 출력용 모델
        self.structured_model = instructor.from_gemini(
            client=genai.GenerativeModel(
                model_name="models/gemini-2.5-flash",
            ),
        )

    def create_personal_color_prompt(self, face_color: FaceColor) -> str:
        """
        퍼스널 컬러 진단 프롬프트 생성 (고도화 버전 v2)
        
        Args:
            face_color (FaceColor): 얼굴 부위별 색상 팔레트 정보
            
        Returns:
            str: 퍼스널 컬러 분석을 위한 고도화된 프롬프트
        """
        # 각 부위별 색상 팔레트 정보를 문자열로 변환
        skin_palette = ", ".join([f"#{c}" for c in face_color.skin])
        hair_palette = ", ".join([f"#{c}" for c in face_color.hair])
        lips_palette = ", ".join([f"#{c}" for c in face_color.lips])
        nose_palette = ", ".join([f"#{c}" for c in face_color.nose])
        eyes_color = f"#{face_color.eyes[0]}" if face_color.eyes else "N/A"

        prompt = f"""
당신은 세계 최고의 퍼스널 컬러 컨설턴트입니다. 당신의 임무는 주어진 색상 팔레트 데이터를 기반으로 고객의 퍼스널 컬러를 매우 정확하게 진단하는 것입니다.

분석 데이터:
- 피부색 팔레트: [{skin_palette}] (피부의 가장 밝은 부분, 중간 부분, 그림자 부분의 색상을 나타냅니다.)
- 코 색상 팔레트: [{nose_palette}] (피부톤 분석에 참고하세요.)
- 입술색 팔레트: [{lips_palette}] (타고난 입술 색은 웜/쿨 진단에 매우 중요한 단서입니다.)
- 머리카락색 팔레트: [{hair_palette}]
- 눈동자색: {eyes_color}

분석 지침:
1.  **언더톤 분석 (가장 중요):** **피부, 코, 입술색 팔레트**에 나타난 여러 색상들의 미묘한 조화를 **종합적으로 분석**하여, 전체적으로 따뜻한 느낌(노란색, 복숭아색 기반)을 주는지, 아니면 차가운 느낌(푸른색, 핑크색 기반)을 주는지를 판단하여 고객의 **언더톤(웜/쿨)**을 결정하세요.
2.  **명도 & 채도 분석:**
    - **명도 (Brightness):** 전체적인 색상(피부, 머리카락)이 얼마나 밝거나 어두운지 분석하세요. (예: 밝은 파스텔 톤, 중간 톤, 깊고 어두운 톤)
    - **채도 (Saturation):** 색상이 얼마나 선명하거나 부드러운지 분석하세요. (예: 쨍하고 선명한 원색, 회색빛이 섞인 부드러운 색)
3.  **종합 진단:** 위에서 분석한 **언더톤, 명도, 채도**를 종합하여, 아래 8가지 세부 타입 중 가장 적합한 **단 하나의 타입**을 최종적으로 결정하세요.

8가지 퍼스널 컬러 타입:
- **Spring Bright (봄 브라이트):** 따뜻한 언더톤. 선명하고(고채도) 밝은(고명도) 색상이 특징. 생기발랄한 느낌.
- **Summer Mute (여름 뮤트):** 차가운 언더톤. 부드럽고(저채도) 밝은(고명도) 회색빛이 섞인 차분한 느낌.
- **Autumn Deep (가을 딥):** 따뜻한 언더톤. 깊고(저명도) 차분한(저채도) 색상이 특징. 성숙하고 분위기 있는 느낌.
- **Winter Bright (겨울 브라이트):** 차가운 언더톤. 강렬하고(고채도) 선명한 색상과 흑백 대비가 특징. 카리스마 있는 느낌.
- **Spring Light (봄 라이트):** 따뜻한 언더톤. 매우 밝고(고명도) 부드러운 파스텔 톤이 특징. 화사하고 깨끗한 느낌.
- **Summer Light (여름 라이트):** 차가운 언더톤. 밝고(고명도) 부드러운 파스텔 톤이 특징. 시원하고 청순한 느낌.
- **Autumn Mute (가을 뮤트):** 따뜻한 언더톤. 부드럽고(저채도) 차분한 중간 밝기(중명도)의 색상이 특징. 자연스럽고 부드러운 느낌.
- **Winter Deep (겨울 딥):** 차가운 언더톤. 매우 어둡고(저명도) 깊은 색상이 특징. 무게감 있고 시크한 느낌.

최종 답변은 다른 설명 없이, 결정된 **퍼스널 컬러 타입명 하나만** 간결하게 출력하세요.
가을 뮤트 답변이 너무 많이 나오고 있는데 한국인 기준으로 만들어줘

"""
        print("--- Generated Gemini Prompt ---")
        print(prompt)
        print("-------------------------------")
        return prompt
    

    async def create_analyze_structured(self, 
                                      user_id: int,
                                      db: Session
                                    ) -> str:        
        """
        구조화된 퍼스널 컬러 분석 프롬프트 생성
        
        Args:
            styling_summary (user_style_summary): 사용자 스타일 정보
            user_profile (user_profile): 사용자 프로필 정보
            
        Returns:
            str: 구조화된 분석을 위한 프롬프트
        """
        styling_summary = get_styling_summary_by_id(db, user_id)
        user_profile = get_user_by_id(db, user_id)
        # 스타일링 요약 정보 추가
        
        styling_info = ""
        if styling_summary:
            styling_info = f"""
            ###데이터###
            - 예산: {styling_summary.budget}원
            - 상황: {styling_summary.occasion}
            - 키: {styling_summary.height}cm
            - 성별: {styling_summary.gender}
            - 상의 사이즈: {styling_summary.top_size}
            - 하의 사이즈: {styling_summary.bottom_size}
            - 신발 사이즈: {styling_summary.shoe_size}
            - 체형 특징: {', '.join(styling_summary.body_feature) if styling_summary.body_feature else '없음'}
            - 선호 스타일: {', '.join(styling_summary.preferred_styles) if styling_summary.preferred_styles else '없음'}
            ###데이터###
            """
        
        # 현제 퍼스널 컬러가 존재하는 경우 가져와서 사용
        profile_info = ""
        if user_profile and user_profile.personal_color_name:
            profile_info = f"""
            ### 현재 퍼스널 컬러 정보 ###
            - 현재 퍼스널 컬러: {user_profile.personal_color_name}
            """
        
        s_prompt = f"""
        당신은 전문 패션 스타일리스트로 활동하면서, 퍼스널 컬러와 성별, 상의 사이즈, 하의 사이즈, 체형(str), 신발 사이즈, 선호하는 스타일, 
        그리고 옷을 입을 상황(데이트 등) 정보를 입력받아 고객에게 최적의 옷 조합을 제안하는 업무를 수행합니다. 
        주로 퍼스널 컬러 진단 결과와 신체 치수를 기반으로 컬러 매칭, 실루엣 강약 조절, 
        아이템 밸런스를 고려하여 상의·하의·아우터·신발·원피스까지 포함한 완벽한 코디를 구성합니다. 

분석 정보:
{styling_info}
{profile_info}

다음 형식으로 응답하세요:

        ### 분석 가이드라인 ###
        카테고리 (ex 맨투맨, 슬랙스등

반드시 카테고리는 아래의 카테고리에서 골라야해합니다.

상의              001
맨투맨/스웨트       001005
후드 티셔츠          001004
셔츠/블라우스       001002
긴소매 티셔츠       001010
반소매 티셔츠       001001
피케/카라 티셔츠       001003
니트/스웨터          001006
민소매 티셔츠       001011
기타 상의          001008

하의             003
데님 팬츠         003002
트레이닝/조거팬츠      003004
코튼 팬츠         003007
슈트 팬츠/슬랙스      003008
숏 팬츠         003009
레깅스            003005
점프 슈트/오버올      003010
기타 하의         003006

아우터            002
후드 집업         002022
블루종/MA-1         002001
레더/라이더스 재킷      002002
카디건            002020
트러거 재킷         002017
슈트/블레이저 재킷      002003
스타디움 재킷      002004
나일론/코치 재킷      002006
아노락 재킷         002019
트레이닝 재킷      002018
환절기 코트         002008
사파리/헌팅 재킷      002014
베스트            002021
숏패딩            002012
무스탕/퍼         002025
플리스/뽀글이      002023
겨울 싱글 코트      002007
겨울 더블 코트      002024
겨울 기타 코트      002009
롱패딩/헤비 아우터      002013
패딩 베스트         002016
기타 아우터         002015

원피스/스커트 100
미니원피스         100001
미디원피스         100002
맥시원피스         100003
미니스커트         100004
미디스커트         100005
롱스커트         100006

신발 103
스니커즈          103004
패딩/퍼 신발       103007
부츠/워커          103002
구두             103001
샌들/슬리퍼          103003
스포츠화          103005

색상
반드시 색상도 마찬가지로 아래 목록에 있는 항목들만 사용해야합니다

화이트          WHITE
실버             SILVER
라이트 그레이       LIGHTGREY
그레이          GRAY
다크 그레이          DARKGREY
블랙             BLACK
레드             RED
딥레드          DEEPRED
버건디          BURGUNDY
브릭             BRICK
페일 핑크          PALEPINK
라이트 핑크          LIGHTPINK
핑크             PINK
다크 핑크          DARKPINK
피치             PEACH
로즈골드          ROSEGOLD
라이트 오렌지       LIGHTORANGE
오렌지          ORANGE
다크              DARKORANGE
아이보리          IVORY
오트밀          OATMENT
라이트 옐로우       LIGHTYELLOW
옐로우          YELLOW
머스타드          MUSTARD
골드             GOLD
라임             LIME
라이트 그린          LIGHTGREEN
그린             GREEN
올리브 그린          OLIVEGREEN
카키             KHAKI
다크 그린          DARKGREEN
민트             MENT
스카이 블루          SKYBLUE
블루             BLUE
다크 블루          DARKBLUE
네이비          NAVY
다크 네이비          DARKNAVY
라벤더          LAVENDER
퍼플             PURPLE
라이트 브라운       LIGHTBROWN
브라운          BROWN
다크 브라운          DAKTBROWN
카멜             CAMEL
샌드             SAND
베이지          BEIGE
다크 베이지          DARKBEIGE
카키 베이지          KHAKIBEIGE
데님             DENIM
연청             LIGHTBLUEDENIM
중청             MEDIUMBLUEDENIM
진청             DARKBLUEDENIM
흑청             BLACKDENIM

        반드시 진단 결과 정책의 내용은 지켜야합니다, 만약 정책을 어길시 패널티를 받게 될것입니다.

        ### 진단 결과 정책###
        반드시 선호 스타일 하나당 3개의 세트 코디를 만들어야 합니다. 스타일별로 구분 가능하게 만들어야합니다. 반드시 작성된 모든 스타일에 대한 코디를 3개씩 추천해야합니다
        룩 이외의 다른것을 출력하지 않습니다, 양식에 있는 정보를 모두 받아야합니다,
        옷에 맞는 색상은 반드시 퍼스널 컬러에 맞는 색으로 추천해야합니다.
        look_Des에는 색상 관련 사항을 포함하지 않습니다. 띄어쓰기를 사용합니다.
        비어있는 아이템은 null값을 사용하세요.
        모든 아이템 필드는 반드시 객체 형태로 반환하세요  .
        아우터, 상의, 하의, 신발, 원피스 등 모든 카테고리는 반드시 포함해야 합니다.
        반드시 코드에 띄어쓰기 없이 쭉 나열하여 작성합니다. 줄띄움은 필요 없습니다.
        상의,하의,신발,원피스,아우터같이 대분류카테고리는 반드시 코드를 작성해야합니다.
        만약 여성이라면 원피스 추천도 해주세요.
        만약 원피스를 추천을 해줬으면 상의,하의는 null값을 반환해야합니다.
        ### 진단 결과 정책###        
        """
        return s_prompt

    async def get_personal_color_analysis(self, face_color: FaceColor) -> str:
        """
        Gemini API를 통한 퍼스널 컬러 분석
        
        Args:
            face_color (FaceColor): 얼굴 부위별 색상 정보
            
        Returns:
            str: 퍼스널 컬러 분석 결과 (텍스트)
        """
        try:
            # 프롬프트 생성
            prompt = self.create_personal_color_prompt(face_color)
            # Gemini API 호출
            response = self.text_model.generate_content(prompt)
            # 텍스트 응답만 반환
            return response.text
            
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "quota" in error_msg.lower():
                return "API 할당량이 소진되었습니다. 잠시 후 다시 시도해주세요."
            elif "API_KEY" in error_msg:
                return "API 키가 설정되지 않았습니다. 환경변수를 확인해주세요."
            else:
                return f"분석 중 오류가 발생했습니다: {error_msg}"

    async def get_personal_color_structured(self, 
                                          user_id: int,
                                          db : Session) -> GeminiExamplePrompt:        
        """
        구조화된 퍼스널 컬러 분석
        
        Args:
            styling_summary (user_style_summary): 사용자 스타일 정보
            user_profile (user_profile): 사용자 프로필 정보
            
        Returns:
            GeminiExamplePrompt: 구조화된 분석 결과
        """
        # 기본 예시 프롬프트 생성
        # example_prompt = GeminiExamplePrompt(
        #     recommendations=[]
        # )
        
        s_prompt = await self.create_analyze_structured(
            user_id,
            db
        )
        try:
            result = self.structured_model.create(
                response_model=GeminiExamplePrompt,
                messages=[{"role": "user", "content": s_prompt}]
            )
            return result
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "quota" in error_msg.lower():
                raise Exception("API 할당량이 소진되었습니다. 잠시 후 다시 시도해주세요.")
            elif "API_KEY" in error_msg:
                raise Exception("API 키가 설정되지 않았습니다. 환경변수를 확인해주세요.")
            else:
                raise Exception(f"구조화된 분석 중 오류가 발생했습니다: {error_msg}")

# 서비스 함수
async def analyze_personal_color(face_color: FaceColor, user_id: int, db: Session) -> str:
    """
    퍼스널 컬러 분석 메인 함수
    
    Args:
        face_color (FaceColor): 얼굴 부위별 색상 정보
        
    Returns:
        str: 퍼스널 컬러 분석 결과
    """
    consultant = GeminiColorConsultant()
    result = await consultant.get_personal_color_analysis(face_color)
    create_user_personal_color_in_db(db, user_id, result)
    return result

async def structured_personal_color_analysis(
                                          user_id: int,
                                          db : Session) -> GeminiExamplePrompt:
    """
    구조화된 퍼스널 컬러 분석 메인 함수
    
    Args:
        styling_summary (user_style_summary): 사용자 스타일 정보
        user_profile (user_profile): 사용자 프로필 정보
        
    Returns:
        GeminiExamplePrompt: 구조화된 분석 결과
    """
    consultant = GeminiColorConsultant()
    result = await consultant.get_personal_color_structured(
        user_id,
        db
    )
    return result

# 스키마를 기반으로 데이터를 추출하는 함수
def extract_crawling_tasks(parsed_data: GeminiExamplePrompt) -> List[CrawlingTask]:
    """
    파싱된 Pydantic 객체에서 크롤링에 필요한 데이터를 추출합니다.
    
    Args:
        parsed_data (GeminiExamplePrompt): 파싱된 분석 결과
        
    Returns:
        List[CrawlingTask]: 크롤링 작업 리스트
    """
    tasks = []
    for recommendation in parsed_data.recommendations:
        for look in recommendation.looks:
            for item_info in look.items.values():
                if item_info and item_info.category_id and item_info.item_code and item_info.color:
                    # 필수 필드들이 모두 존재하는 경우에만 작업 생성
                    task = CrawlingTask(
                        category_id=item_info.category_id,
                        item_code=item_info.item_code,
                        color=item_info.color,
                        style_name=recommendation.style_name,
                        look_name=look.look_name
                    )
                    tasks.append(task)
    return tasks

