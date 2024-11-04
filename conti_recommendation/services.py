# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from functools import lru_cache
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
from konlpy.tag import Okt
import re
import random
import openai
import nltk
from decouple import config
import os
from django.conf import settings
import pickle
from scipy.sparse import save_npz, load_npz
import jpype

# NLTK 데이터 다운로드 (필요 시)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# 프로젝트의 BASE_DIR 설정 (Django settings.py 파일의 BASE_DIR을 사용)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# cache 디렉터리 경로 설정 및 생성
CACHE_DIR = os.path.join(BASE_DIR, 'conti_recommendation', 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)  # 디렉터리 생성 (존재하지 않을 경우)

# TF-IDF 캐시 파일 및 벡터라이저 파일 경로 설정
TFIDF_CACHE_FILE = os.path.join(CACHE_DIR, 'tfidf_vectors.npz')
TFIDF_VECTORIZER_FILE = os.path.join(CACHE_DIR, 'tfidf_vectorizer.pkl')

# JVM 초기화 함수
def initialize_jvm():
    if not jpype.isJVMStarted():
        try:
            # JVM 경로 설정 (필요에 따라 수정)
            jvm_path = jpype.getDefaultJVMPath()
            print(f"Starting JVM at: {jvm_path}")

            # 필요한 .jar 파일 경로 설정
            javadir = r'C:\Projects\konlpy\java'
            classpath = (
                f"{javadir}{os.sep}open-korean-text-2.1.0.jar;"
                f"{javadir}{os.sep}scala-library-2.12.12.jar;"
                f"{javadir}{os.sep}scala-reflect-2.12.12.jar;"
                f"{javadir}{os.sep}scala-parser-combinators_2.12-1.1.2.jar;"
                f"{javadir}{os.sep}twitter-text-2.0.8.jar"
            )

            jpype.startJVM(jvm_path, f"-Djava.class.path={classpath}", "-Dfile.encoding=UTF-8")
            print("JVM started successfully.")
        except Exception as e:
            print(f"Failed to start JVM: {e}")
            raise

# OpenAI API 키 설정
openai.api_key = config('OPENAI_API_KEY')

def load_data(file_path='data.csv'):
    full_path = os.path.join(settings.BASE_DIR, 'conti_recommendation', 'data', file_path)
    encodings = ['utf-8', 'cp949', 'euc-kr', 'iso-8859-1']

    for encoding in encodings:
        try:
            df = pd.read_csv(full_path, encoding=encoding)
            print(f"Successfully loaded the file using {encoding} encoding.")
            return df
        except UnicodeDecodeError:
            continue
        except FileNotFoundError:
            print(f"Error: File {full_path} not found.")
            return None
        except pd.errors.EmptyDataError:
            print(f"Error: File {full_path} is empty.")
            return None
        except pd.errors.ParserError:
            print(f"Error: Unable to parse {full_path}. Make sure it's a valid CSV file.")
            return None

    print("Error: Unable to decode the file with any of the attempted encodings.")
    return None

def load_bible(file_path='bible.txt'):
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
            print(f"Successfully loaded the Bible file using {encoding} encoding.")
            return bible_dict
        except UnicodeDecodeError:
            continue

    print("Error: Unable to decode the Bible file with any of the attempted encodings.")
    return None

def compute_and_cache_tfidf(df, column='lyrics', cache_file=TFIDF_CACHE_FILE, vectorizer_file=TFIDF_VECTORIZER_FILE):
    """
    TF-IDF 벡터를 계산하고 캐싱
    - df: 가사 데이터가 포함된 데이터프레임
    - column: 가사가 저장된 데이터프레임의 컬럼 이름
    - cache_file: TF-IDF 벡터를 저장할 파일 경로
    - vectorizer_file: TF-IDF 벡터라이저를 저장할 파일 경로
    """
    # 캐시 파일이 존재하는지 확인
    if os.path.exists(cache_file) and os.path.exists(vectorizer_file):
        print(f"캐시된 TF-IDF 벡터와 벡터라이저를 {cache_file} 및 {vectorizer_file}에서 불러옵니다...")
        tfidf_matrix = load_npz(cache_file)
        with open(vectorizer_file, 'rb') as f:
            tfidf_vectorizer = pickle.load(f)
        return tfidf_matrix, tfidf_vectorizer

    # TF-IDF 벡터를 새로 계산하고 캐싱
    print("TF-IDF 벡터를 새로 계산하고 캐싱합니다...")
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(df[column])

    # 캐시 파일로 저장
    save_npz(cache_file, tfidf_matrix)
    with open(vectorizer_file, 'wb') as f:
        pickle.dump(tfidf_vectorizer, f)

    return tfidf_matrix, tfidf_vectorizer

def parse_verse(verse):
    match = re.match(r'([가-힣]+)(\d+):(\d+)', verse)
    if match:
        book, chapter, verse = match.groups()
        return book, int(chapter), int(verse)
    else:
        raise ValueError(f"Invalid verse format: {verse}")

def get_bible_verses(bible_dict, start_verse, end_verse):
    start_book, start_chapter, start_num = parse_verse(start_verse)
    end_book, end_chapter, end_num = parse_verse(end_verse)

    if start_book != end_book:
        raise ValueError("Start and end verses must be from the same book")

    verses = []
    for chapter in range(start_chapter, end_chapter + 1):
        start = start_num if chapter == start_chapter else 1
        end = end_num if chapter == end_chapter else 1000  # 임의의 큰 숫자

        for num in range(start, end + 1):
            verse_id = f'{start_book}{chapter}:{num}'
            if verse_id in bible_dict:
                verses.append(bible_dict[verse_id])
            else:
                break  # 해당 장의 마지막 구절에 도달

    return ' '.join(verses)

@lru_cache(maxsize=1000)
def preprocess_korean_text(text, pos_filter=None):
    # JVM 초기화 확인 및 실행
    initialize_jvm()

    okt = Okt()
    pos_tags = okt.pos(text)
    if pos_filter:
        filtered_words = [word for word, pos in pos_tags if pos in pos_filter.split(',')]
    else:
        filtered_words = [word for word, pos in pos_tags]
    return ' '.join(filtered_words)

def extract_keywords(text, top_n=5, pos_filter=None):
    processed_text = preprocess_korean_text(text, pos_filter)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([processed_text])
    feature_names = vectorizer.get_feature_names_out()
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

def train_word2vec_model(sentences, vector_size=100, window=5, min_count=1, workers=4):
    return Word2Vec(sentences, vector_size=vector_size, window=window, min_count=min_count, workers=workers)

def calculate_similarities(df, keywords_tfidf, keywords_w2v, tfidf_matrix, word2vec_model):
    tfidf_similarities = cosine_similarity(keywords_tfidf, tfidf_matrix).flatten()

    w2v_similarities = []
    for text in df['processed_lyrics']:
        words = text.split()
        vectors = [word2vec_model.wv[word] for word in words if word in word2vec_model.wv]
        if vectors:
            text_vector = np.mean(vectors, axis=0)
            if np.any(text_vector) and np.any(keywords_w2v):
                similarity = cosine_similarity([keywords_w2v], [text_vector])[0][0]
            else:
                similarity = 0
        else:
            similarity = 0
        w2v_similarities.append(similarity)
    w2v_similarities = np.array(w2v_similarities)

    return tfidf_similarities, w2v_similarities

def match_songs_with_keywords(df, user_keywords, bible_verse_range, bible_dict, similarity_threshold=0.4):
    # 성경 구절을 기반으로 키워드 추출
    bible_text = get_bible_verses(bible_dict, *bible_verse_range.split('~'))
    bible_keywords = extract_keywords(bible_text)
    all_keywords = user_keywords + bible_keywords

    # lyrics 컬럼이 존재하는지 확인
    if 'lyrics' not in df.columns:
        print("Error: 'lyrics' column not found in the dataframe.")
        return None, None, None

    # processed_lyrics 컬럼을 생성하여 전처리된 가사 데이터 저장
    try:
        df['processed_lyrics'] = df['lyrics'].apply(lambda x: preprocess_korean_text(x) if pd.notnull(x) else "")
        print("Successfully processed and added 'processed_lyrics' column to the dataframe.")
    except Exception as e:
        print(f"Error during processing 'lyrics' to 'processed_lyrics': {e}")
        return None, None, None

    # TF-IDF 벡터 캐싱을 사용하여 처리
    try:
        tfidf_matrix, tfidf_vectorizer = compute_and_cache_tfidf(df, column='processed_lyrics')
    except Exception as e:
        print(f"Error during computing and caching TF-IDF: {e}")
        return None, None, None

    # Word2Vec 모델 학습
    sentences = [text.split() for text in df['processed_lyrics']]
    word2vec_model = train_word2vec_model(sentences)

    # 입력 키워드를 전처리한 후, TF-IDF 및 Word2Vec 벡터 계산
    processed_keywords = preprocess_korean_text(' '.join(all_keywords))
    keywords_tfidf = tfidf_vectorizer.transform([processed_keywords])
    keywords_w2v = safe_mean_vector(word2vec_model, processed_keywords.split())

    # 유사도 계산
    try:
        tfidf_similarities, w2v_similarities = calculate_similarities(df, keywords_tfidf, keywords_w2v, tfidf_matrix, word2vec_model)
    except Exception as e:
        print(f"Error during similarity calculation: {e}")
        return None, None, None

    # 유효한 유사도 계산을 위해 최소값과 최대값이 같을 경우를 처리
    if np.max(tfidf_similarities) != np.min(tfidf_similarities):
        tfidf_similarities = (tfidf_similarities - np.min(tfidf_similarities)) / (np.max(tfidf_similarities) - np.min(tfidf_similarities))
    else:
        tfidf_similarities = np.zeros_like(tfidf_similarities)

    if np.max(w2v_similarities) != np.min(w2v_similarities):
        w2v_similarities = (w2v_similarities - np.min(w2v_similarities)) / (np.max(w2v_similarities) - np.min(w2v_similarities))
    else:
        w2v_similarities = np.zeros_like(w2v_similarities)

    # 최종 유사도 계산 및 매칭된 곡 추출
    final_similarities = 0.5 * tfidf_similarities + 0.5 * w2v_similarities
    df['similarity'] = final_similarities
    matched_songs = df[df['similarity'] > similarity_threshold].sort_values('similarity', ascending=False)

    # Noun 키워드 추출
    noun_keywords = extract_keywords(' '.join(all_keywords), pos_filter='Noun')

    return matched_songs, noun_keywords, bible_text

def recommend_songs(matched_songs, min_recommendations=4, max_recommendations=5):
    num_recommendations = random.randint(min_recommendations, max_recommendations)

    if len(matched_songs) <= num_recommendations:
        return matched_songs

    top_half = matched_songs.head(len(matched_songs)//2)
    recommendations = top_half.sample(min(num_recommendations, len(top_half)))

    if len(recommendations) < num_recommendations:
        remaining = matched_songs.tail(len(matched_songs) - len(matched_songs)//2)
        additional = remaining.sample(num_recommendations - len(recommendations))
        recommendations = pd.concat([recommendations, additional])

    return recommendations.sort_values('similarity', ascending=False)

def generate_gpt4o_mini_conti_title(keywords, bible_verse_range, recommended_songs):
    prompt = f"""
    다음 정보를 바탕으로 창의적이고 매력적인 플레이리스트 제목을 만들어주세요

    키워드: {', '.join(keywords)}
    성경 구절 범위: {bible_verse_range}
    추천 노래: {', '.join([song['title'] for _, song in recommended_songs.iterrows()])}

    제목은 한국어로 작성해 주시고, 플레이리스트의 주제와 내용을 잘 반영하되 창의적이고 흥미로운 표현을 사용해 주세요.
    부제는 생성하지 말고, 단일 제목만 만들어주세요.
    따옴표나 기타 부가적인 텍스트 없이 제목만 반환해주세요.
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
        print(f"An error occurred while generating the title: {e}")
        return "콘티 제목 생성 중 오류 발생"

def generate_gpt4o_mini_conti_description(keywords, bible_verse_range, recommended_songs, bible_text, conti_title):
    prompt = f"""
    다음 정보를 바탕으로 300자 이내로 플레이리스트에 대한 설명을 작성해주세요:

    플레이리스트 제목: {conti_title}
    키워드: {', '.join(keywords)}
    성경 구절 범위: {bible_verse_range}
    성경 구절 내용: {bible_text}
    추천 노래: {', '.join([song['title'] for _, song in recommended_songs.iterrows()])}

    설명은 한국어로 작성해 주시고, 다음 내용을 포함해야 합니다:
    1. 플레이리스트의 주제와 목적
    2. 선택된 성경 구절의 의미와 플레이리스트와의 연관성
    3. 추천된 노래들이 어떻게 주제와 연결되는지
    4. 청취자들에게 이 플레이리스트가 어떤 영감을 줄 수 있는지

    설명은 충분한 정보를 담고 있어야 하며, 영적이고 감동적인 톤으로 작성해 주세요.
    설명은 300자 이내로 작성해주세요.
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
        print(f"An error occurred while generating the description: {e}")
        return "플레이리스트 설명 생성 중 오류 발생"

def create_conti(songs_df, user_keywords, bible_verse_range):
    bible_dict = load_bible('bible.txt')
    if bible_dict is None:
        return {"error": "Failed to load the Bible file."}

    try:
        # 1. 키워드 및 성경 구절 매칭
        matched_songs, noun_keywords, bible_text = match_songs_with_keywords(songs_df, user_keywords, bible_verse_range, bible_dict)
        print(f"Keywords used (nouns only): {', '.join(noun_keywords)}")
        print(f"Found {len(matched_songs)} matching songs.")

        # 2. 4~5개 곡 추천
        recommended_songs = recommend_songs(matched_songs)
        print(f"\nRecommended {len(recommended_songs)} songs:")
        for _, song in recommended_songs.iterrows():
            print(f"Title: {song['title']}, Artist: {song['artist']}, Similarity: {song['similarity']:.4f}")

        # 3. 재생목록 제목 생성
        conti_title = generate_gpt4o_mini_conti_title(noun_keywords, bible_verse_range, recommended_songs)
        print(f"\nConti Title: {conti_title}")

        # 4. 재생목록 설명 생성
        conti_description = generate_gpt4o_mini_conti_description(noun_keywords, bible_verse_range, recommended_songs, bible_text, conti_title)
        print(f"\nConti Description:\n{conti_description}")

        return {
            "title": conti_title,
            "description": conti_description,
            "songs": recommended_songs[['title', 'artist', 'similarity']].to_dict('records')
        }
    except Exception as e:
        print(f"An error occurred: {e}")
        return {"error": "Failed to generate conti."}
