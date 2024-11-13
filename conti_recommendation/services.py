# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from functools import lru_cache
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
from kiwipiepy import Kiwi
import re
import random
import openai
from decouple import config
import os
from django.conf import settings
import pickle
from scipy.sparse import save_npz, load_npz
import logging

logger = logging.getLogger(__name__)

# 프로젝트의 BASE_DIR 설정 (Django settings.py 파일의 BASE_DIR을 사용)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# cache 디렉터리 경로 설정 및 생성
CACHE_DIR = os.path.join(BASE_DIR, 'conti_recommendation', 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)  # 디렉터리 생성 (존재하지 않을 경우)

# TF-IDF 캐시 파일 및 벡터라이저 파일 경로 설정
TFIDF_CACHE_FILE = os.path.join(CACHE_DIR, 'tfidf_vectors.npz')
TFIDF_VECTORIZER_FILE = os.path.join(CACHE_DIR, 'tfidf_vectorizer.pkl')

# Word2Vec 모델 경로
WORD2VEC_MODEL_FILE = os.path.join(CACHE_DIR, 'word2vec.model')

PROCESSED_LYRICS_FILE = os.path.join(CACHE_DIR, 'processed_lyrics.pkl')

# OpenAI API 키 설정
openai.api_key = config('OPENAI_API_KEY')

# 전역 변수 선언
songs_df = None
bible_dict = None
tfidf_matrix = None
tfidf_vectorizer = None
word2vec_model = None

kiwi = Kiwi()

def load_data(file_path='data.csv'):
    global songs_df
    full_path = os.path.join(settings.BASE_DIR, 'conti_recommendation', 'data', file_path)
    encodings = ['utf-8', 'cp949', 'euc-kr', 'iso-8859-1']

    for encoding in encodings:
        try:
            df = pd.read_csv(full_path, encoding=encoding)
            logger.info(f"Successfully loaded the file using {encoding} encoding.")
            songs_df = df
            return df
        except UnicodeDecodeError:
            continue
        except FileNotFoundError:
            logger.error(f"Error: File {full_path} not found.")
            return None
        except pd.errors.EmptyDataError:
            logger.error(f"Error: File {full_path} is empty.")
            return None
        except pd.errors.ParserError:
            logger.error(f"Error: Unable to parse {full_path}. Make sure it's a valid CSV file.")
            return None

    logger.error("Error: Unable to decode the file with any of the attempted encodings.")
    return None

def load_bible(file_path='bible.txt'):
    global bible_dict
    full_path = os.path.join(settings.BASE_DIR, 'conti_recommendation', 'data', file_path)
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
            logger.info(f"Successfully loaded the Bible file using {encoding} encoding.")
            return bible_dict
        except UnicodeDecodeError:
            continue

    logger.error("Error: Unable to decode the Bible file with any of the attempted encodings.")
    return None

def compute_and_cache_tfidf():
    """
    TF-IDF 벡터를 계산하고 캐싱
    - df: 가사 데이터가 포함된 데이터프레임
    - column: 가사가 저장된 데이터프레임의 컬럼 이름
    - cache_file: TF-IDF 벡터를 저장할 파일 경로
    - vectorizer_file: TF-IDF 벡터라이저를 저장할 파일 경로
    """
    # 캐시 파일이 존재하는지 확인
    global tfidf_matrix, tfidf_vectorizer
    if os.path.exists(TFIDF_CACHE_FILE) and os.path.exists(TFIDF_VECTORIZER_FILE):
        logger.info(f"Loading cached TF-IDF vectors and vectorizer from {TFIDF_CACHE_FILE} and {TFIDF_VECTORIZER_FILE}...")
        tfidf_matrix = load_npz(TFIDF_CACHE_FILE)
        with open(TFIDF_VECTORIZER_FILE, 'rb') as f:
            tfidf_vectorizer = pickle.load(f)
        return tfidf_matrix, tfidf_vectorizer

    # TF-IDF 벡터를 새로 계산하고 캐싱
    logger.info("Computing and caching TF-IDF vectors...")
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(songs_df['processed_lyrics'])

    # 캐시 파일로 저장
    save_npz(TFIDF_CACHE_FILE, tfidf_matrix)
    with open(TFIDF_VECTORIZER_FILE, 'wb') as f:
        pickle.dump(tfidf_vectorizer, f)

    return tfidf_matrix, tfidf_vectorizer

def parse_verse(verse):
    match = re.match(r'([가-힣]+)(\d+):(\d+)', verse)
    if match:
        book, chapter, verse = match.groups()
        return book, int(chapter), int(verse)
    else:
        raise ValueError(f"Invalid verse format: {verse}")

def get_bible_verses(start_verse, end_verse):
    global bible_dict
    start_book, start_chapter, start_num = parse_verse(start_verse)
    end_book, end_chapter, end_num = parse_verse(end_verse)

    if start_book != end_book:
        raise ValueError("Start and end verses must be from the same book")

    verses = []
    for chapter in range(start_chapter, end_chapter + 1):
        start = start_num if chapter == start_chapter else 1
        end = end_num if chapter == end_chapter else 1000

        for num in range(start, end + 1):
            verse_id = f'{start_book}{chapter}:{num}'
            if verse_id in bible_dict:
                verses.append(bible_dict[verse_id])
            else:
                break

    return ' '.join(verses)

def preprocess_korean_text(text, pos_filter=None):
    global kiwi
    analyzed = kiwi.analyze(text)
    tokens = []
    for sentence in analyzed:
        for token, pos, _, _ in sentence[0]:
            if pos_filter:
                pos_filters = pos_filter.split(',')
                if pos in pos_filters:
                    tokens.append(token)
            else:
                tokens.append(token)
    return ' '.join(tokens)

def extract_keywords(text, top_n=5, pos_filter=None):
    if pos_filter is None:
        pos_filter = 'NNG,NNP'
    processed_text = preprocess_korean_text(text, pos_filter)
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform([processed_text])
    feature_names = tfidf.get_feature_names_out()
    tfidf_scores = tfidf_matrix.toarray()[0]
    word_scores = list(zip(feature_names, tfidf_scores))
    word_scores.sort(key=lambda x: x[1], reverse=True)
    return [word for word, score in word_scores[:top_n]]

def safe_mean_vector(model, words):
    vectors = [model.wv[word] for word in words if word in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)

def train_word2vec_model(sentences, vector_size=100, window=5, min_count=5, workers=4):
    return Word2Vec(sentences, vector_size=vector_size, window=window, min_count=min_count, workers=workers)

def load_or_train_word2vec():
    global word2vec_model
    if os.path.exists(WORD2VEC_MODEL_FILE):
        logger.info("Loading cached Word2Vec model.")
        word2vec_model = Word2Vec.load(WORD2VEC_MODEL_FILE)
    else:
        logger.info("Training Word2Vec model.")
        sentences = [text.split() for text in songs_df['processed_lyrics']]
        word2vec_model = train_word2vec_model(sentences)
        word2vec_model.save(WORD2VEC_MODEL_FILE)
        logger.info("Word2Vec model trained and saved.")

    return word2vec_model

def calculate_similarities(keywords_tfidf, keywords_w2v):
    tfidf_similarities = cosine_similarity(keywords_tfidf, tfidf_matrix).flatten()

    # Word2Vec 유사도 계산
    split_lyrics = songs_df['processed_lyrics'].str.split().explode()
    
    # Word2Vec 어휘에 있는 단어만 필터링
    filtered_words = split_lyrics[split_lyrics.isin(word2vec_model.wv.key_to_index)]
    
    if filtered_words.empty:
        logger.warning("No words found in Word2Vec vocabulary.")
        # 모든 곡에 대해 0 벡터를 할당
        text_vectors = pd.DataFrame(np.zeros((len(songs_df), word2vec_model.vector_size)), index=songs_df.index)
    else:
        # 존재하는 단어들의 벡터 가져오기
        vectors = word2vec_model.wv[filtered_words]
        # 'vectors'를 DataFrame으로 변환, 인덱스는 곡의 인덱스
        vectors_df = pd.DataFrame(vectors, index=filtered_words.index)
        # 각 곡별 평균 벡터 계산
        text_vectors = vectors_df.groupby(level=0).mean()
        # 누락된 곡에 대해 0 벡터 할당
        text_vectors = text_vectors.reindex(songs_df.index, fill_value=0)
    
    # 코사인 유사도 계산
    w2v_similarities = cosine_similarity([keywords_w2v], text_vectors).flatten()

    return tfidf_similarities, w2v_similarities

def match_songs_with_keywords(user_keywords, bible_verse_range, similarity_threshold=0.4):
    # 성경 구절을 기반으로 키워드 추출
    try:
        start_verse, end_verse = bible_verse_range.split('~')
        bible_text = get_bible_verses(start_verse, end_verse)
        bible_keywords = extract_keywords(bible_text)
    except ValueError as e:
        logger.error(f"Error parsing bible_verse_range: {e}")
        return None, None, None

    all_keywords = user_keywords + bible_keywords

    # 입력 키워드를 전처리한 후, TF-IDF 및 Word2Vec 벡터 계산
    processed_keywords = preprocess_korean_text(' '.join(all_keywords))
    keywords_tfidf = tfidf_vectorizer.transform([processed_keywords])
    keywords_w2v = safe_mean_vector(word2vec_model, processed_keywords.split())

    # 유사도 계산
    tfidf_similarities, w2v_similarities = calculate_similarities(keywords_tfidf, keywords_w2v)

    # 정규화
    if np.max(tfidf_similarities) != np.min(tfidf_similarities):
        tfidf_similarities = (tfidf_similarities - np.min(tfidf_similarities)) / (np.max(tfidf_similarities) - np.min(tfidf_similarities))
    else:
        tfidf_similarities = np.zeros_like(tfidf_similarities)

    if np.max(w2v_similarities) != np.min(w2v_similarities):
        w2v_similarities = (w2v_similarities - np.min(w2v_similarities)) / (np.max(w2v_similarities) - np.min(w2v_similarities))
    else:
        w2v_similarities = np.zeros_like(w2v_similarities)

    # 최종 유사도 계산
    final_similarities = 0.5 * tfidf_similarities + 0.5 * w2v_similarities
    songs_df['similarity'] = final_similarities

    # 매칭된 곡 추출
    matched_songs = songs_df[songs_df['similarity'] > similarity_threshold].sort_values('similarity', ascending=False)

    # Noun 키워드 추출
    noun_keywords = extract_keywords(' '.join(all_keywords), pos_filter='NNG,NNP')

    return matched_songs, noun_keywords, bible_text

def recommend_songs(matched_songs, min_recommendations=4, max_recommendations=5):
    num_recommendations = random.randint(min_recommendations, max_recommendations)

    if len(matched_songs) <= num_recommendations:
        return matched_songs

    top_half = matched_songs.head(len(matched_songs) // 2)
    recommendations = top_half.sample(min(num_recommendations, len(top_half)))

    if len(recommendations) < num_recommendations:
        remaining = matched_songs.tail(len(matched_songs) - len(matched_songs) // 2)
        additional = remaining.sample(num_recommendations - len(recommendations))
        recommendations = pd.concat([recommendations, additional])

    return recommendations.sort_values('similarity', ascending=False)

def generate_gpt4o_mini_conti_title(keywords, bible_verse_range, recommended_songs):
    prompt = f"""
    다음 정보를 참고하여 창의적이고 매력적인 한국어 플레이리스트 제목을 작성해주세요.

    - 키워드: {', '.join(keywords)}
    - 성경 구절 범위: {bible_verse_range}
    - 추천 노래: {', '.join([song['title'] for _, song in recommended_songs.iterrows()])}

    플레이리스트의 주제와 내용을 잘 반영하면서도 흥미롭고 기억에 남는 제목을 만들어주세요. 부제는 필요 없으며, 따옴표나 기타 부가적인 텍스트 없이 순수한 제목만 반환해주세요.
    """

    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=[
                {"role": "system", "content": "You are a creative assistant specialized in generating Korean playlist titles."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=50,
            n=1,
            temperature=0.7,
        )

        title = response.choices[0].message.content.strip()
        return title
    except Exception as e:
        logger.exception(f"An error occurred while generating the title: {e}")
        return "콘티 제목 생성 중 오류 발생"

def generate_gpt4o_mini_conti_description(keywords, bible_verse_range, recommended_songs, bible_text, conti_title):
    prompt = f"""
    다음 정보를 참고하여 300자 이내의 한국어 플레이리스트 설명을 작성해주세요.

    - 플레이리스트 제목: {conti_title}
    - 키워드: {', '.join(keywords)}
    - 성경 구절 범위: {bible_verse_range}
    - 성경 구절 내용: {bible_text}
    - 추천 노래: {', '.join([song['title'] for _, song in recommended_songs.iterrows()])}

    설명에는 다음 내용을 포함해주세요:
    1. 플레이리스트의 주제와 목적
    2. 선택된 성경 구절의 의미 및 플레이리스트와의 연관성
    3. 추천된 노래들이 주제와 어떻게 연결되는지
    4. 청취자들에게 주는 영감

    설명은 영적이고 감동적인 톤으로 작성해 주세요.
    """

    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=[
                {"role": "system", "content": "You are a thoughtful assistant specialized in creating meaningful descriptions for spiritual playlists."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            n=1,
            temperature=0.7,
        )

        description = response.choices[0].message.content.strip()
        return description
    except Exception as e:
        logger.exception(f"An error occurred while generating the description: {e}")
        return "플레이리스트 설명 생성 중 오류 발생"

def create_conti(user_keywords, bible_verse_range):
    try:
        if songs_df is None:
            logger.error("Songs data is not loaded.")
            return {"error": "Internal server error."}

        if bible_dict is None:
            logger.error("Bible data is not loaded.")
            return {"error": "Internal server error."}

        # 1. 키워드 및 성경 구절 매칭
        matched_songs, noun_keywords, bible_text = match_songs_with_keywords(
            user_keywords, bible_verse_range
        )
        if matched_songs is None:
            return {"error": "Failed to match songs with keywords and bible verses."}

        logger.info(f"Keywords used (nouns only): {', '.join(noun_keywords)}")
        logger.info(f"Found {len(matched_songs)} matching songs.")

        # 2. 4~5개 곡 추천
        recommended_songs = recommend_songs(matched_songs)
        if recommended_songs is None or len(recommended_songs) == 0:
            return {"error": "No suitable songs found."}

        logger.info(f"\nRecommended {len(recommended_songs)} songs:")
        for _, song in recommended_songs.iterrows():
            logger.info(f"ID: {song['id']}, Title: {song['title']}, Artist: {song['artist']}, Similarity: {song['similarity']:.4f}")

        # 3. 재생목록 제목 생성
        conti_title = generate_gpt4o_mini_conti_title(noun_keywords, bible_verse_range, recommended_songs)
        if not conti_title:
            return {"error": "Failed to generate playlist title."}

        logger.info(f"\nConti Title: {conti_title}")

        # 4. 재생목록 설명 생성
        conti_description = generate_gpt4o_mini_conti_description(
            noun_keywords, bible_verse_range, recommended_songs, bible_text, conti_title
        )
        if not conti_description:
            return {"error": "Failed to generate playlist description."}

        logger.info(f"\nConti Description:\n{conti_description}")

        return {
            "title": conti_title,
            "description": conti_description,
            "songs": recommended_songs[['id']].to_dict('records')
        }
    except Exception as e:
        logger.exception(f"An error occurred: {e}")
        return {"error": "Failed to generate conti."}

def initialize_services():
    global songs_df, bible_dict, tfidf_matrix, tfidf_vectorizer, word2vec_model

    songs_df = load_data('data.csv')
    if songs_df is not None:
        if os.path.exists(PROCESSED_LYRICS_FILE):
            logger.info("Loading cached processed lyrics...")
            with open(PROCESSED_LYRICS_FILE, 'rb') as f:
                songs_df['processed_lyrics'] = pickle.load(f)
        else:
            logger.info("Processing lyrics...")
            songs_df['processed_lyrics'] = songs_df['lyrics'].apply(lambda x: preprocess_korean_text(x) if pd.notnull(x) else "")
            with open(PROCESSED_LYRICS_FILE, 'wb') as f:
                pickle.dump(songs_df['processed_lyrics'], f)
        
        tfidf_matrix, tfidf_vectorizer = compute_and_cache_tfidf()
        word2vec_model = load_or_train_word2vec()
    bible_dict = load_bible('bible.txt')