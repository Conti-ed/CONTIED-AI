# -*- coding: utf-8 -*-
import json
import difflib
import pandas as pd
import numpy as np
import os
import re
import pickle
import logging
import asyncio
from typing import Dict, List, Tuple, Any, Optional

from google import genai
from google.genai import types
from .config import settings

logger = logging.getLogger(__name__)

# 저장 경로 설정
DATA_DIR = os.path.join(settings.BASE_DIR, 'app', 'data')
CACHE_DIR = os.path.join(settings.BASE_DIR, 'app', 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)

# 캐시 파일 경로 (바이너리 임베딩 데이터)
SONG_EMBEDDINGS_FILE = os.path.join(CACHE_DIR, 'song_embeddings.npy')

# Gemini 모델 설정
REASONER_MODEL = 'gemini-3.1-flash-lite-preview'
EMBEDDING_MODEL = 'gemini-embedding-001'
DIMENSION = 3072  # 임베딩 벡터의 출력 차원

# 새로운 Google GenAI 클라이언트 초기화
client = genai.Client(api_key=settings.GEMINI_API_KEY)

# 전역 변수 (메모리 로딩 용)
songs_df: Optional[pd.DataFrame] = None
bible_dict: Optional[Dict[str, str]] = None
song_embeddings: Optional[np.ndarray] = None

def load_data(file_name='data.csv') -> Optional[pd.DataFrame]:
    """찬양 곡 데이터베이스(CSV)를 메모리에 로드합니다."""
    global songs_df
    full_path = os.path.join(DATA_DIR, file_name)
    encodings = ['utf-8', 'cp949', 'euc-kr', 'iso-8859-1']

    for encoding in encodings:
        try:
            df = pd.read_csv(full_path, encoding=encoding)
            logger.info(f"성공: {file_name} 파일을 {encoding} 인코딩으로 불러왔습니다.")
            songs_df = df
            return df
        except Exception as e:
            logger.debug(f"인코딩 시도 실패 ({encoding}): {e}")
            continue

    logger.error("오류: 모든 인코딩 방식으로도 파일을 불러올 수 없습니다.")
    return None

def load_bible(file_name='bible.txt') -> Optional[Dict[str, str]]:
    """성경 구절 텍스트 파일을 딕셔너리 형태로 로드합니다."""
    global bible_dict
    full_path = os.path.join(DATA_DIR, file_name)
    encodings = ['utf-8', 'cp949', 'euc-kr', 'iso-8859-1']

    for encoding in encodings:
        try:
            bible_dict = {}
            with open(full_path, 'r', encoding=encoding) as file:
                for line in file:
                    parts = line.strip().split(' ', 1)
                    if len(parts) == 2:
                        verse_id, text = parts
                        bible_dict[verse_id] = text
            logger.info(f"성공: 성경 데이터를 {encoding} 인코딩으로 불러왔습니다.")
            return bible_dict
        except Exception as e:
            logger.debug(f"성경 로딩 시도 실패 ({encoding}): {e}")
            continue

    logger.error("오류: 성경 데이터를 불러올 수 없습니다.")
    return None

async def get_embedding_async(text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> np.ndarray:
    """텍스트를 3072차원 임베딩 벡터로 변환합니다. (429 에러 발생 시 재시도 포함)"""
    if not text or not text.strip():
        return np.zeros(DIMENSION)

    max_retries = 5
    base_delay = 5  # 초

    for attempt in range(max_retries):
        try:
            response = await client.aio.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=text,
                config={
                    'task_type': task_type,
                    'title': "Song embedding" if task_type == "RETRIEVAL_DOCUMENT" else None
                }
            )

            # 임베딩 데이터가 리스트로 올 경우 처리
            if hasattr(response, 'embeddings') and len(response.embeddings) > 0:
                all_values = [np.array(e.values) for e in response.embeddings]
                final_emb = np.mean(all_values, axis=0) if len(all_values) > 1 else all_values[0]
            else:
                final_emb = np.array(response.embeddings[0].values)

            # 차원이 정확한지 검증 및 보정
            if final_emb.shape[0] != DIMENSION:
                if final_emb.shape[0] > DIMENSION:
                    final_emb = final_emb[:DIMENSION]
                else:
                    final_emb = np.pad(final_emb, (0, DIMENSION - final_emb.shape[0]))

            return final_emb

        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                delay = base_delay * (attempt + 1)
                logger.warning(f"할당량 초과(429). {delay}초 후 다시 시도합니다... (시도 {attempt + 1}/{max_retries})")
                await asyncio.sleep(delay)
                continue

            logger.error(f"임베딩 생성 최종 오류 (시도 {attempt + 1}): {e}")
            return np.zeros(DIMENSION)

    return np.zeros(DIMENSION)

async def compute_song_embeddings():
    """모든 곡 가사를 임베딩하여 캐시 파일로 저장하거나 로드합니다."""
    global song_embeddings

    if os.path.exists(SONG_EMBEDDINGS_FILE):
        logger.info("캐시된 임베딩 데이터를 확인하는 중...")
        try:
            cached_embs = np.load(SONG_EMBEDDINGS_FILE)
            if len(cached_embs) == len(songs_df) and cached_embs.shape[1] == DIMENSION:
                logger.info(f"유효한 {DIMENSION}차원 임베딩 {len(songs_df)}건을 로드했습니다.")
                song_embeddings = cached_embs
                return
            logger.info(f"캐시 불일치 감지 ({cached_embs.shape[1]}차원 != {DIMENSION}차원). 다시 계산합니다.")
        except Exception as e:
            logger.warning(f"캐시 파일을 읽을 수 없습니다: {e}")

    logger.info(f"전체 찬양({len(songs_df)}곡)의 임베딩을 계산합니다. 이 작업은 처음에만 한 차례 수행됩니다.")

    # API 요청 제한을 조절하기 위한 세마포어 (무료 티어는 동시 요청을 최소화)
    semaphore = asyncio.Semaphore(2)

    async def get_song_emb(idx, row):
        async with semaphore:
            # 제목, 아티스트, 가사를 조합하여 문맥 제공
            combined_text = f"{row.get('title', '')} {row.get('artist', '')} {row.get('lyrics', '')[:500]}"
            emb = await get_embedding_async(combined_text)
            # 요청 간 약간의 지연 시간 추가 (TPM 제한 방어)
            await asyncio.sleep(0.5)
            if (idx + 1) % 10 == 0:
                logger.info(f"진행 상황: 총 {len(songs_df)}곡 중 {idx + 1}곡 처리 완료...")
            return emb

    tasks = [get_song_emb(i, r) for i, r in songs_df.iterrows()]
    embeddings = await asyncio.gather(*tasks)

    song_embeddings = np.array(embeddings)
    np.save(SONG_EMBEDDINGS_FILE, song_embeddings)
    logger.info(f"임베딩 계산 완료. 캐시가 {SONG_EMBEDDINGS_FILE}에 저장되었습니다.")

def cosine_similarity_manual(v1: np.ndarray, v2_matrix: np.ndarray) -> np.ndarray:
    """넘파이를 사용한 코사인 유사도 계산 (두 벡터 사이의 각도 점수)"""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2_matrix, axis=1)
    if norm1 == 0:
        return np.zeros(len(v2_matrix))
    dot_product = np.dot(v2_matrix, v1)
    # 0으로 나누기 방지
    norm2[norm2 == 0] = 1e-9
    return dot_product / (norm1 * norm2)

def parse_verse(verse: str) -> Tuple[str, int, int]:
    """성경 구절 문자열(예: 창세기1:1)을 파싱합니다."""
    match = re.match(r'([가-힣]+)(\d+):(\d+)', verse)
    if match:
        book, chapter, v_num = match.groups()
        return book, int(chapter), int(v_num)
    raise ValueError(f"지원하지 않는 구절 형식입니다: {verse}")

def get_bible_verses(start_verse: str, end_verse: str) -> str:
    """시작 구절부터 끝 구절까지의 텍스트를 추출합니다."""
    try:
        start_book, start_chap, start_num = parse_verse(start_verse)
        end_book, end_chap, end_num = parse_verse(end_verse)
        if start_book != end_book:
            return ""

        verses = []
        for chapter in range(start_chap, end_chap + 1):
            s = start_num if chapter == start_chap else 1
            e = end_num if chapter == end_chap else 1000
            for num in range(s, e + 1):
                v_id = f'{start_book}{chapter}:{num}'
                if v_id in bible_dict:
                    verses.append(bible_dict[v_id])
                else:
                    break
        return ' '.join(verses)
    except:
        return ""

async def extract_search_intent(user_keywords: List[str], bible_text: str) -> str:
    """사용자가 입력한 정보에서 핵심적인 검색 문장을 AI가 추출해냅니다."""
    prompt = f"""
    예배 찬양 콘티를 위해 아래 키워드와 성경구절의 영적 의미를 하나로 통합하세요.
    - 입력 키워드: {', '.join(user_keywords)}
    - 성경 구절: {bible_text[:500]}
    부연 설명 없이 검색에 사용할 '단 한 문장'만 한국어로 출력하세요.
    """
    try:
        response = await client.aio.models.generate_content(model=REASONER_MODEL, contents=prompt)
        # 인사말이나 따옴표 제거 작업
        content = response.text.strip().replace('"', '').split('\n')[0]
        return content
    except Exception as e:
        logger.error(f"검색 의도 파악 오류: {e}")
        return ' '.join(user_keywords)

def _description_mentions_real_songs(
    description: str,
    song_titles: List[str],
    threshold: float = 0.6,
) -> bool:
    """description에 인용 부호로 감싼 곡명 후보가 등장할 때, 실제 곡명과 fuzzy match하는지 확인.
    인용 없으면 True(통과), 인용이 있는데 실제 곡명과 너무 다르면 False 반환."""
    candidates = re.findall(r'[\'"“‘]([^\'\"”’]{2,30})[\'"”’]', description)
    if not candidates:
        return True  # 인용 없으면 환각 위험 없음 — 통과
    for cand in candidates:
        best = max(
            (difflib.SequenceMatcher(None, cand, t).ratio() for t in song_titles),
            default=0.0,
        )
        if best < threshold:
            return False
    return True


async def generate_title_and_description(
    keywords: List[str],
    bible_range: str,
    bible_text: str,
    songs: pd.DataFrame,
) -> Dict[str, str]:
    """
    제목과 설명을 한 번의 LLM 호출로 생성합니다 (JSON structured output 사용).
    인자:
        keywords   - 사용자 입력 키워드 리스트
        bible_range - "롬1:1~1:5" 형식 성경 범위 문자열
        bible_text  - 성경 본문 발췌 (최대 600자)
        songs       - 선택된 곡 DataFrame (title, artist, lyrics 컬럼 포함)
    반환:
        {"title": str, "description": str}
    """
    # 곡 목록 및 가사 발췌 구성
    song_lines: List[str] = []
    for i, (_, row) in enumerate(songs.iterrows(), start=1):
        title = str(row.get('title', ''))
        artist = str(row.get('artist', ''))
        lyrics_raw = str(row.get('lyrics', ''))
        lyrics_excerpt = lyrics_raw[:200] if lyrics_raw else ''
        if lyrics_excerpt:
            song_lines.append(f'  {i}. {title} — {artist}\n     가사 발췌: "{lyrics_excerpt}"')
        else:
            song_lines.append(f'  {i}. {title} — {artist}')
    songs_block = '\n'.join(song_lines)

    # 성경 본문 블록 (빈 경우 생략 지시)
    bible_text_trimmed = bible_text[:600] if bible_text else ''
    if bible_text_trimmed:
        bible_line = f'- 성경 본문: "{bible_text_trimmed}"'
    else:
        bible_line = '- 성경 본문: (없음 — 키워드 비중을 높여 작성하세요)'

    prompt = f"""당신은 예배 찬양 콘티의 제목과 설명을 짓는 전문가입니다.

[입력]
- 키워드: {', '.join(keywords)}
- 성경 범위: {bible_range if bible_range else '(없음)'}
{bible_line}
- 선택된 곡 ({len(songs)}곡):
{songs_block}

[제목 규칙]
1. 15자 이내 한국어
2. 특수기호([], **, "" 등) 절대 금지
3. '콘티 제목' 같은 메타 표현 금지
4. 핵심 주제와 감동을 담은 짧고 임팩트 있는 한 줄
예시(다양한 톤 참고):
  - 주님께 나아가는 길  (행위 + 대상)
  - 은혜의 강물         (비유)
  - 감사로 열리는 문    (행위 + 비유)
  - 흔들리지 않는 반석  (영적 안정)
  - 다시 일어서는 우리  (회복 주제)

[설명 규칙]
1. 250자 이내 한국어 2개 문단 (\\n\\n 으로 구분)
2. 첫 문단: '우리는 ~합니다' 형태의 고백체로, 본문/키워드의 핵심 영적 메시지
3. 두 번째 문단: 선택된 찬양들 중 1-2곡을 구체적으로 언급하며 본문과 어떻게 연결되는지 한 문장 포함
4. 마크다운(*, [], #, 숫자 리스트) 절대 금지
5. 따뜻하고 깊은 묵상이 느껴지는 어조

[참고 예시]
입력 키워드: 회복, 새벽
성경: 시편 30:5
선택된 곡: "주의 인자하심", "새벽 이슬같은 주의 청년들이"
출력 예시 description:
"우리는 슬픔이 밤새 머물지라도 새벽이면 기쁨이 임하시는 주님의 신실하심을 고백합니다. 어떤 어둠 속에서도 다시 시작하게 하시는 주의 인자하심을 의지합니다.

찬양 '주의 인자하심'과 '새벽 이슬같은 주의 청년들이'는 시편 30편의 회복의 약속을 노래합니다. 오늘 예배가 다시 일어서는 결단의 자리가 되기를 소망합니다."

위 예시와 같은 구조와 어조로, 아래 실제 입력을 사용하여 작성하세요.

반드시 JSON 형식으로 응답: {{"title":"...","description":"..."}}"""

    # fallback 기본값
    fallback_title = "새로운 예배 콘티"
    fallback_desc = (
        f"'{', '.join(keywords)}' 주제를 바탕으로 구성된 맞춤형 예배 콘티입니다. "
        "선별된 찬양들을 통해 하나님의 풍성한 은혜를 누리는 시간이 되시길 소망합니다."
    )

    try:
        response = await client.aio.models.generate_content(
            model=REASONER_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type='application/json',
                response_schema={
                    'type': 'object',
                    'properties': {
                        'title': {'type': 'string'},
                        'description': {'type': 'string'},
                    },
                    'required': ['title', 'description'],
                },
            ),
        )

        if not response.text:
            raise ValueError("AI 응답이 비어있습니다.")

        data = json.loads(response.text)
        raw_title: str = data.get('title', fallback_title).strip()
        raw_desc: str = data.get('description', fallback_desc).strip()

        # 안전망 후처리: 특수문자 제거, 제목 길이 슬라이스
        title = re.sub(r'[*\[\]"\'\.]', '', raw_title)
        if len(title) > 15:
            title = title[:15]

        desc = re.sub(r'[*\[\]#]', '', raw_desc)

        return {"title": title, "description": desc}

    except Exception as e:
        logger.exception(f"제목/설명 생성 실패 (Fallback 적용): {e}")
        return {"title": fallback_title, "description": fallback_desc}

async def create_recommendation(
    keywords: List[str],
    bible_range: Optional[str],
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """추천 생성의 메인 파이프라인 (추출 -> 임베딩 -> 유사도 -> 생성)"""
    try:
        if seed is not None:
            logger.info(f"seed={seed} 적용 (재생성 모드)")

        # 성경 구절 파싱 및 추출
        bible_text = ""
        if bible_range and bible_range.strip():
            bible_range = bible_range.replace(" ", "")
            if '~' not in bible_range:
                bible_range = f"{bible_range}~{bible_range}"
            start_v, end_v = bible_range.split('~')
            bible_text = get_bible_verses(start_v, end_v)
        else:
            bible_range = ""

        # 1. AI를 활용한 정교한 검색 쿼리(의도) 생성
        search_query = await extract_search_intent(keywords, bible_text)
        logger.info(f"생성된 검색어(Semantic Query): {search_query}")

        # 2. 쿼리를 임베딩 벡터(3072차원)로 변환
        query_emb = await get_embedding_async(search_query, "RETRIEVAL_QUERY")
        logger.info(f"쿼리 임베딩 변환 완료 (차원: {query_emb.shape[0]})")

        # 3. 데이터베이스와 유사도 계산
        similarities = cosine_similarity_manual(query_emb, song_embeddings)
        songs_df['similarity'] = similarities

        # 4. 유사도 내림차순 정렬 후 Top-15 추출
        matched = songs_df.sort_values('similarity', ascending=False).head(15)

        if seed is not None:
            # seed가 있으면 Top-15 안에서 셔플 후 5곡 선택 → "다시 생성" 시 다른 결과
            rng = np.random.default_rng(seed)
            shuffled_idx = rng.permutation(len(matched))[:5]
            recs = matched.iloc[sorted(shuffled_idx)]  # 유사도 순서 보존
        else:
            recs = matched.head(5)  # 기본: 결정론적 Top-5

        # 5. AI를 이용한 제목 및 설명 한 번에 생성 (LLM 호출 1회)
        generated = await generate_title_and_description(
            keywords=keywords,
            bible_range=bible_range,
            bible_text=bible_text,
            songs=recs,
        )

        # 5-1. 출력 검증: description 내 인용 곡명이 실제 곡명과 너무 다르면 1회 재시도
        song_titles = [str(s.get('title', '')) for s in recs.to_dict('records')]
        if not _description_mentions_real_songs(generated["description"], song_titles):
            logger.warning("설명에 환각 곡명 의심 — 1회 재시도")
            retry = await generate_title_and_description(
                keywords=keywords,
                bible_range=bible_range,
                bible_text=bible_text,
                songs=recs,
            )
            if not _description_mentions_real_songs(retry["description"], song_titles):
                logger.warning("재시도 후에도 환각 곡명 의심 — 원본 결과로 통과")
            else:
                generated = retry

        title = generated["title"]
        desc = generated["description"]

        # 6. 응답에 video_id, title, artist 포함 (BE의 videoId 기반 매핑을 위해)
        song_records = recs[['id', 'videoId', 'title', 'artist']].rename(
            columns={'videoId': 'video_id'}
        ).to_dict('records')

        logger.info(
            f"추천 곡 video_id 목록: {[r.get('video_id') for r in song_records]}"
        )

        return {
            "title": title,
            "description": desc,
            "songs": song_records,
        }
    except Exception as e:
        logger.exception(f"추천 생성 실패: {e}")
        return {"error": str(e)}

async def initialize():
    """애플리케이션 시작 시 데이터와 임베딩을 준비합니다."""
    global songs_df, bible_dict
    load_data('data.csv')
    load_bible('bible.txt')
    if songs_df is not None:
        await compute_song_embeddings()
    logger.info(f"{EMBEDDING_MODEL}을 이용한 시맨틱 추천 엔진 준비 완료.")
