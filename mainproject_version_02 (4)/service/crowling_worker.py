import sys
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import json
import os
import time
import logging
from bs4 import BeautifulSoup
from typing import List, Dict

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from schemas.user_schema import user_style_summary
from schemas.item_schema import item_info_request, item_info_response, item_info_snapshot, look_info
from schemas.crowling_schema import CrawlingTask

# 로깅 설정
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter(u'%(asctime)s [%(levelname)8s] %(%(levelname)8s] %(message)s')
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(formatter)
logger.addHandler(consoleHandler)

# Chrome 옵션 설정 (크롤링 성능 최적화)
chrome_options = Options()
chrome_options.add_argument('--headless')  # 브라우저 창 숨기기 (백그라운드 실행)
chrome_options.add_argument('--no-sandbox')  # 샌드박스 비활성화 (보안 우회)
chrome_options.add_argument('--disable-dev-shm-usage')  # 공유 메모리 사용 비활성화
chrome_options.add_argument('--disable-gpu')  # GPU 가속 비활성화 (안정성 향상)
chrome_options.add_argument('--disable-webgl')  # WebGL 비활성화 (메모리 절약)
chrome_options.add_argument('--disable-software-rasterizer')  # 소프트웨어 래스터라이저 비활성화
chrome_options.add_argument('--disable-extensions')  # 확장 프로그램 비활성화 (성능 향상)
chrome_options.add_argument('--disable-plugins')  # 플러그인 비활성화 (메모리 절약)
chrome_options.add_argument('--disable-images')  # 이미지 로딩 비활성화 (속도 향상)
chrome_options.add_argument('--window-size=1920,1080')  # 브라우저 창 크기 설정
chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')  # 사용자 에이전트 설정

# 로그 레벨 설정
chrome_options.add_argument('--log-level=3')  # 오류만 표시 (로그 노이즈 감소)
chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])  # Chrome 로그 비활성화

# 사용자 성별 매핑
male = {"남": "M", "여": "F", "기타" : "A"}

# 카테고리 정보
# 대분류 카테고리 코드 (무신사 카테고리 체계)
category_codes = {
    "001": "상의",
    "002": "아우터",
    "003": "하의",
    "100": "원피스/스커트",
    "103": "신발"
}

# 카테고리별 아이템 설정 (최소 가격 및 스타일 정보 포함 여부)
item_configs = {
    "100" :     {"min_price": 4900, "has_style": True},  # 원피스/스커트
    "001" :     {"min_price": 990, "has_style": True},    # 상의
    "002" :     {"min_price": 4900, "has_style": True},   # 아우터
    "003" :     {"min_price": 3500, "has_style": True},   # 하의
    "103" :     {"min_price": 4500, "has_style": False},  # 신발 (스타일 정보 없음)
}

# 색상 매핑 (한국어 -> 영문 코드)
color_map = {
   '화이트': 'WHITE',
   '실버': 'SILVER',
   '라이트 그레이': 'LIGHTGREY',
   '그레이': 'GRAY',
   '다크 그레이': 'DARKGREY',
   '블랙': 'BLACK',
   '레드': 'RED',
   '딥레드': 'DEEPRED',
   '버건디': 'BURGUNDY',
   '브릭': 'BRICK',
   '페일 핑크': 'PALEPINK',
   '라이트 핑크': 'LIGHTPINK',
   '핑크': 'PINK',
   '다크 핑크': 'DARKPINK',
   '피치': 'PEACH',
   '로즈골드': 'ROSEGOLD',
   '라이트 오렌지': 'LIGHTORANGE',
   '오렌지': 'ORANGE',
   '다크': 'DARKORANGE',
   '아이보리': 'IVORY',
   '오트밀': 'OATMENT',
   '라이트 옐로우': 'LIGHTYELLOW',
   '옐로우': 'YELLOW',
   '머스타드': 'MUSTARD',
   '골드': 'GOLD',
   '라임': 'LIME',
   '라이트 그린': 'LIGHTGREEN',
   '그린': 'GREEN',
   '올리브 그린': 'OLIVEGREEN',
   '카키': 'KHAKI',
   '다크 그린': 'DARKGREEN',
   '민트': 'MENT',
   '스카이 블루': 'SKYBLUE',
   '블루': 'BLUE',
   '다크 블루': 'DARKBLUE',
   '네이비': 'NAVY',
   '다크 네이비': 'DARKNAVY',
   '라벤더': 'LAVENDER',
   '퍼플': 'PURPLE',
   '라이트 브라운': 'LIGHTBROWN',
   '브라운': 'BROWN',
   '다크 브라운': 'DAKTBROWN',
   '카멜': 'CAMEL',
   '샌드': 'SAND',
   '베이지': 'BEIGE',
   '다크 베이지': 'DARKBEIGE',
   '카키 베이지': 'KHAKIBEIGE',
   '데님': 'DENIM',
   '연청': 'LIGHTBLUEDENIM',
   '중청': 'MEDIUMBLUEDENIM',
   '진청': 'DARKBLUEDENIM',
   '흑청': 'BLACKDENIM'
}

# 스타일 매핑 (한국어 -> 숫자 코드)
style_map = {
   '캐주얼': 1,
   '스트릿': 2,
   '고프코어': 3,
   '워크웨어': 4,
   '프레피': 5,
   '시티보이': 6,
   '스포티': 7,
   '로맨틱': 8,
   '걸리시': 9,
   '클래식': 10,
   '미니멀': 11,
   '시크': 12,
   '레트로': 13,
   '에스닉': 14,
   '리조트': 15
}

# 무신사 기본 URL
musinsa_base = "https://www.musinsa.com"

def crowling_item(item_type, category, user_style, user_male, user_top_size, user_bottom_size, user_shoe_size, user_color, user_price, user_filter):
    """
    무신사 URL 생성 함수
    
    Args:
        item_type (str): 아이템 타입 (카테고리 코드)
        category (str): 카테고리
        user_style (str): 사용자 스타일
        user_male (str): 사용자 성별
        user_top_size (str): 상의 사이즈
        user_bottom_size (str): 하의 사이즈
        user_shoe_size (str): 신발 사이즈
        user_color (str): 색상
        user_price (int): 가격
        user_filter (int): 필터 값
        
    Returns:
        str: 생성된 무신사 URL
    """
    # 설정 딕셔너리에서 현재 아이템의 설정을 가져옵니다.
    config = item_configs.get(item_type)
    user_size = ""
    if item_type == "003":  # 하의인 경우
        user_size = user_bottom_size
    elif item_type == "103":  # 신발인 경우
        user_size = user_shoe_size
    else:  # 상의, 아우터, 원피스/스커트인 경우
        user_size = user_top_size



    # 지원하지 않는 item_type이 들어오면 에러 처리
    if not config:
        print(f"오류: 지원하지 않는 아이템 종류입니다 - {item_type}")
        return

    # URL의 기본 부분을 만듭니다 (성별 필터 적용)
    musinsa_url = f"{musinsa_base}/category/{category}?gf={male[user_male]}"

    # 'has_style'이 True일 때만 URL에 스타일 파라미터를 추가합니다.
    if config["has_style"] and user_style:
        musinsa_url += f"&style={user_style}"

    # 나머지 공통 파라미터를 추가합니다 (사이즈, 가격 범위)
    musinsa_url += f"&size={user_size}&minPrice={config['min_price']}&maxPrice={user_price}"

    # 필터(색상)가 활성화된 경우 파라미터를 추가합니다.
    if user_filter == 1:
        musinsa_url += f"&color={user_color}"

    return musinsa_url


def extract_price_info(container):
    """
    가격 정보를 추출하고 정규화하는 함수
    
    Args:
        container: BeautifulSoup 컨테이너 객체
        
    Returns:
        int: 정규화된 가격 (원 단위)
    """
    price_container = container.find('div', class_='sc-hKDTPf sc-fmZSGO fGOKsY fCqHUk')
    if not price_container:
        return 0
    
    price_spans = price_container.find_all('span', class_='text-body_13px_semi')
    if len(price_spans) >= 2:
        # 할인률과 가격이 모두 있는 경우
        actual_price = price_spans[1].get_text(strip=True)
    elif len(price_spans) == 1:
        # 할인률이 없는 경우
        actual_price = price_spans[0].get_text(strip=True)
    else:
        return 0
    
    # 가격 정규화 (41,400원 -> 41400)
    price_text = actual_price.replace('원', '').replace(',', '').strip()
    try:
        return int(price_text)
    except ValueError:
        return 0
    except ValueError:
        return 0

def extract_image_url(container):
    """
    이미지 URL을 추출하는 함수
    
    Args:
        container: BeautifulSoup 컨테이너 객체
        
    Returns:
        str: 이미지 URL 또는 빈 문자열
    """

    img_tag = container.find('img', class_='max-w-full w-full absolute m-auto inset-0 h-auto z-0 visible object-cover')
    if img_tag is None:
        img_tag = container.find('img', class_='max-w-full w-full absolute m-auto inset-0 h-auto z-0 visible object-contain')
    
    if img_tag:
        image_url = img_tag.get('src')
        if image_url:
            # 쿼리 파라미터 제거
            base_url = image_url.split('?')[0]
            return base_url
    
    return None

def extract_product_info(container) -> item_info_response:
    """
    상품 정보를 추출하는 함수
    
    Args:
        container: BeautifulSoup 컨테이너 객체
        
    Returns:
        item_info_response: 추출된 상품 정보
    """
    try:
        product_links = container.find_all('a', {'data-item-id': True})
        
        if not product_links:
            return None
            
        link = product_links[0]
        product_id = link.get('data-item-id')
        # 상품명을 정확한 위치에서 찾기
        product_name_element = container.find('span', class_='text-body_13px_reg sc-dYOLZc sc-hoLldG kpFgRS bNmpOr font-pretendard')
        if product_name_element:
            product_name = product_name_element.get_text(strip=True)
        else:
            # 대안: 링크의 텍스트에서 찾기
            product_name = link.get_text(strip=True)
        
        # 가격 정보 찾기
        price = extract_price_info(container)
        
        # 이미지 URL 찾기
        image_url = extract_image_url(container)
        
        return item_info_response(
            product_id=product_id,
            product_name=product_name,
            image_url=image_url,
            price=price
        )
    except Exception as e:
        print(f"상품 정보 추출 실패: {str(e)}")
        return None


def process_crawling_results(soup, category):
    """
    크롤링 결과를 처리하는 함수
    
    Args:
        soup: BeautifulSoup 객체
        category: 카테고리 정보
        
    Returns:
        list: 처리된 상품 정보 리스트
    """
    product_containers = soup.find_all('div', class_="sc-igtioI eSJwIO")
    logger.info(f"\n=== {category} 카테고리 상품 목록 ===")
    logger.info(f"총 {len(product_containers)}개의 상품 컨테이너를 찾았습니다.")
    
    if not product_containers:
        logger.warning("상품을 찾을 수 없습니다.")
        # 페이지 소스 일부를 로그로 출력하여 디버깅
        page_source_preview = soup.prettify()[:1000]
        logger.info(f"페이지 소스 미리보기: {page_source_preview}")
        return None
    
    container = product_containers[0]
    product_info = extract_product_info(container)
    
    if product_info:
        logger.info(f"{product_info.product_name} ({product_info.price}원) [ID: {product_info.product_id}]")
        return product_info
    else:
        logger.info("상품 링크를 찾을 수 없습니다.")
        # 컨테이너 내용을 로그로 출력
        container_preview = str(container)[:500]
        logger.info(f"컨테이너 내용 미리보기: {container_preview}")
        return None

def crowling_item_info(item, user_style, filter: int):
    """
    멀티프로세싱을 위한 워커 함수
    
    Args:
        item: 크롤링할 아이템 정보
        user_style: 사용자 스타일 정보
        filter: 필터 값
        
    Returns:
        item_info_response: 크롤링 결과
    """
    wd = None
    try:
        wd = webdriver.Chrome(options=chrome_options)
        
        if item['category_id'] not in item_configs:
            raise ValueError("잘못된 대카테고리입니다.")
        
        # 무신사 페이지 접속 URL 생성
        crowling_url = crowling_item(item['category_id'], item['item_code'], style_map[item['style_name']], user_style['gender'], user_style['top_size'], user_style['bottom_size'], user_style['shoe_size'], item['color'], user_style['budget'], filter)
        
        logger.info(f"접속 URL: {crowling_url}")
        logger.info(f"Item data: {item}")
        logger.info(f"User style: {user_style}")
        logger.info(f"Filter: {filter}")
        
        wd.get(crowling_url)
        
        # 페이지 로딩 완료 대기 (명시적 대기)
        wait = WebDriverWait(wd, 10)
        try:
            # 상품 컨테이너가 로드될 때까지 대기
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div.sc-igtioI.eSJwIO")))
        except TimeoutException:
            logger.info("상품 컨테이너를 찾을 수 없습니다. 기본 대기 시간 사용")
            time.sleep(3)  # 기본 대기 시간
        
        logger.info(f"페이지 제목: {wd.title}")
        
        # 상품 목록 처리 (HTML 파싱)
        soup = BeautifulSoup(wd.page_source, 'html.parser')
        return process_crawling_results(soup, item['item_code'])
        
    except Exception as e:
        logger.error(f"크롤링 중 오류 발생: {str(e)}")
        raise e
    finally:
        if wd:
            wd.quit()
            logger.info("브라우저 종료 및 크롤링 완료")

if __name__ == "__main__":
    """
    멀티프로세싱 워커 스크립트 메인 함수
    명령행 인수: <item_json> <user_style_json> <filter_value>
    """
    result = None
    try:
        if len(sys.argv) != 4:
            raise ValueError("Usage: python crowling_worker.py <item_json> <user_style_json> <filter_value>")
        
        # 명령행 인수 파싱
        item_data = json.loads(sys.argv[1])
        user_style_data = json.loads(sys.argv[2])
        filter_value = int(sys.argv[3])

        # JSON에서 user_style 객체 재구성 (타입 검증 없이)
        user_style_obj = user_style_data

        # 크롤링 실행
        logger.info(f"Starting crawling for item: {item_data}")
        logger.info(f"User style data: {user_style_obj}")
        logger.info(f"Filter value: {filter_value}")
        
        result = crowling_item_info(item_data, user_style_obj, filter_value)
        
        logger.info(f"Crawling result: {result}")
        
        # Pydantic 모델을 JSON 직렬화를 위해 딕셔너리로 변환
        if result:
            print(json.dumps(result.model_dump()))
        else:
            logger.warning("No result found from crawling")
            print(json.dumps(None))  # 결과가 없는 경우 None 반환

    except Exception as e:
        error_message = {"error": str(e)}
        print(json.dumps(error_message), file=sys.stderr)
        sys.exit(1)
