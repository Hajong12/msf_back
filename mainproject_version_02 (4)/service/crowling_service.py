from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import asyncio
import subprocess
import json
import sys
import os
import time
import logging
from bs4 import BeautifulSoup
from typing import List, Dict
import multiprocessing
from schemas.item_schema import item_info_request, item_info_response, item_info_snapshot, look_info
from schemas.crowling_schema import CrawlingTask
from crud.user_crud import get_styling_summary_by_id
from sqlalchemy.orm import Session

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from schemas.item_schema import item_info_request, item_info_response, item_info_snapshot


# 로깅 설정
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter(u'%(asctime)s [%(levelname)8s] %(message)s')
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

# 워커 스크립트 경로 설정
CROWLING_WORKER_PATH = os.path.join(os.path.dirname(__file__), 'crowling_worker.py')

# 로깅 설정 (주석 처리됨)
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)





def _run_crowling_worker_process(item_data_json: str, user_style_json: str, filter_value: int):
    """
    각 서브프로세스에서 crowling_worker.py를 실행하고 결과를 받아오는 함수
    
    Args:
        item_data_json (str): 아이템 데이터 JSON 문자열
        user_style_json (str): 사용자 스타일 정보 JSON 문자열
        filter_value (int): 필터링 값
        
    Returns:
        dict: 워커 프로세스의 실행 결과
    """
    try:
        logger.info(f"[Main PID:{os.getpid()}] Starting worker for item: {item_data_json}")
        
        result = subprocess.run(
            [sys.executable, CROWLING_WORKER_PATH, item_data_json, user_style_json, str(filter_value)],
            capture_output=True,
            text=True,
            check=True
        )
        
        output_data = json.loads(result.stdout.strip())
        logger.info(f"[Main PID:{os.getpid()}] Worker finished. Result: {output_data}")
        return output_data

    except subprocess.CalledProcessError as e:
        logger.error(f"[Main PID:{os.getpid()}] Worker process failed with error: {e}", file=sys.stderr)
        logger.error(f"Stderr: {e.stderr}", file=sys.stderr)
        return {"error": f"Subprocess failed: {e.stderr.strip()}"}
    except json.JSONDecodeError as e:
        logger.error(f"[Main PID:{os.getpid()}] JSON decoding error: {e}", file=sys.stderr)
        logger.error(f"Raw output: {result.stdout}", file=sys.stderr)
        return {"error": "Failed to parse worker output JSON."}
    except Exception as e:
        logger.error(f"[Main PID:{os.getpid()}] Unexpected error: {e}", file=sys.stderr)
        return {"error": str(e)}


def crowling_item_snap(product_id):
    """
    특정 상품의 스냅샷 이미지를 크롤링하는 함수
    
    Args:
        product_id (str): 상품 ID
        
    Returns:
        list: 크롤링된 이미지 URL 리스트
    """
    product_url = f"{musinsa_base}/products/{product_id}"

    wd = None
    scraped_images = []

    try:
        wd = webdriver.Chrome(options=chrome_options)
        wait = WebDriverWait(wd, 10)
        wd.get(product_url)

        try:
            snap_review_section = wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, ".sc-g3hx4t-2.fyXrfB"))
            )
            wd.execute_script("arguments[0].scrollIntoView(true);", snap_review_section)
            logger.info("Scrolled to '.sc-g3hx4t-2.fyXrfB' section.")
        except TimeoutException:
            logger.info("Could not find the snap/review section. Proceeding with current view.")
        
        time.sleep(2)  # 컨텐츠 로딩 대기

        # 2. 스냅 이미지 수집
        soup = BeautifulSoup(wd.page_source, 'html.parser')
        snap_elements = soup.select('div.sc-1hsleli-1.zzIYj')
        logger.info(f"Found {len(snap_elements)} snap elements.")

        for snap in snap_elements:
            if len(scraped_images) >= 3:
                break
            # 'object-cover' 클래스를 가진 img 태그 탐색
            image_tag = snap.select_one('img.object-cover')
            if image_tag and image_tag.get('src'):
                scraped_images.append(image_tag['src'])
        
        logger.info(f"Collected {len(scraped_images)} images from snaps.")

        # 3. 이미지가 3개 미만이면 스타일 후기 탭에서 추가 수집
        if len(scraped_images) < 3:
            logger.info("Less than 3 images found, moving to style reviews.")
            try:
                # '스타일' 텍스트를 포함하는 버튼 클릭
                style_button = wait.until(
                    EC.element_to_be_clickable((By.XPATH, "//*[contains(@class, 'GoodsReviewTabGroup__TabItemWrapper') and contains(., '스타일')]"))
                )
                style_button.click()
                logger.info("Clicked on 'Style' review tab.")
                time.sleep(2)  # 탭 컨텐츠 로딩 대기

                # 페이지 소스를 다시 파싱
                soup = BeautifulSoup(wd.page_source, 'html.parser')
                review_items = soup.select('div.review-list-item__Container-sc-13zantg-0')
                logger.info(f"Found {len(review_items)} style review elements.")

                for review in review_items:
                    if len(scraped_images) >= 3:
                        break
                    # 'ExpandableImage__Image' 클래스를 가진 img 태그 탐색
                    image_tag = review.select_one('img.ExpandableImage__Image-sc-hg8nrj-1')
                    if image_tag and image_tag.get('src'):
                        # 중복 이미지 방지
                        if image_tag['src'] not in scraped_images:
                            scraped_images.append(image_tag['src'])
                
                logger.info(f"Collected {len(scraped_images)} images after checking style reviews.")

            except (TimeoutException, NoSuchElementException) as e:
                logger.info(f"Could not find or click the style review tab: {e}")

    except Exception as e:
        logger.info(f"An error occurred: {e}")
    finally:
        if wd:
            wd.quit()

    # 4. 최종적으로 3개가 안되면 빈 문자열로 채우기
    while len(scraped_images) < 3:
        scraped_images.append("")

    return item_info_snapshot(snap_img_url=scraped_images[:3])


async def process_and_group_crawling_tasks(
    tasks_as_objects: List[CrawlingTask],
    user_id: int,
    db: Session,
    look_descriptions: Dict[str, str],
    filter: int
) -> List[look_info]:
    """
    크롤링 태스크를 처리하고 결과를 look_name별로 그룹화하여 반환합니다.
    
    Args:
        tasks_as_objects (List[CrawlingTask]): 크롤링할 작업 리스트
        styling_summary (user_style_summary): 사용자 스타일 정보
        look_descriptions (Dict[str, str]): 룩별 설명 정보
        filter (int): 필터링 값
        
    Returns:
        List[look_info]: 그룹화된 룩 정보 리스트
    """
    look_groups = {}
    
    styling_summary = get_styling_summary_by_id(db, user_id)
    # 멀티프로세싱 풀을 위한 인수 준비
    pool_args = []
    for task in tasks_as_objects:
        item_data_json = json.dumps(task.model_dump())
        # SQLAlchemy 모델을 딕셔너리로 변환
        styling_summary_dict = {
            'budget': styling_summary.budget,
            'occasion': styling_summary.occasion,
            'height': styling_summary.height,
            'gender': styling_summary.gender,
            'top_size': styling_summary.top_size,
            'bottom_size': styling_summary.bottom_size,
            'shoe_size': styling_summary.shoe_size,
            'body_feature': styling_summary.body_feature,
            'preferred_styles': styling_summary.preferred_styles
        }
        user_style_json = json.dumps(styling_summary_dict)
        pool_args.append((item_data_json, user_style_json, filter))

    # 멀티프로세싱.Pool을 사용하여 작업을 병렬로 실행
    num_processes = multiprocessing.cpu_count()  # 사용 가능한 모든 CPU 코어 사용
    if num_processes < 1:
        num_processes = 1
    
    logger.info(f"Starting parallel crawling with {num_processes} processes...")
    
    # Windows 호환성을 위해 freeze_support() 호출
    multiprocessing.freeze_support()

    with multiprocessing.Pool(processes=num_processes) as pool:
        # pool.starmap은 _run_crowling_worker_process가 여러 인수를 받기 때문에 사용
        results = pool.starmap(_run_crowling_worker_process, pool_args)

    # 풀에서 결과 처리
    for i, result_data in enumerate(results):
        task = tasks_as_objects[i]  # 원본 작업 객체 가져오기
        
        look_name = task.look_name
        look_description = look_descriptions.get(look_name, f"{look_name} 스타일")
        
        logger.info(f"Processing result for task with look_name: {look_name}")
        
        if look_name not in look_groups:
            look_groups[look_name] = {
                'look_name': look_name,
                'look_description': look_description,
                'items': {}
            }
        
        if result_data and "error" not in result_data:
            # 딕셔너리에서 item_info_response 재구성
            product_info = item_info_response(**result_data)
            logger.info(f"Found item for {look_name}: {product_info.product_name}")
            item_key = f"{category_codes[task.category_id]}"
            look_groups[look_name]['items'][item_key] = product_info
        else:
            error_msg = result_data.get('error', 'Unknown error') if result_data else 'No data returned'
            logger.info(f"No items found or error for {look_name}: {error_msg}")
            item_key = f"{category_codes[task.category_id]}"
            look_groups[look_name]['items'][item_key] = item_info_response(
                product_id="0",
                product_name="상품을 찾을 수 없습니다",
                image_url="",
                price=0
            )
    
    look_info_list = []
    for look_data in look_groups.values():
        look_info_list.append(look_info(
            look_name=look_data['look_name'],
            look_description=look_data['look_description'],
            items=look_data['items']
        ))
    
    logger.info(f"Final result: {len(look_info_list)} look_info objects")
    return look_info_list