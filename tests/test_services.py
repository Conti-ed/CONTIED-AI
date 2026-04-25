# -*- coding: utf-8 -*-
"""
app.services 내 순수 헬퍼 함수 단위 테스트.

LLM API 호출 함수(generate_title_and_description, extract_search_intent,
get_embedding_async)는 실제 API 키 없이는 동작하지 않으므로 테스트하지 않습니다.
"""
import os
import sys

# services.py 임포트 전 더미 API 키 설정 (genai.Client 초기화 통과용)
os.environ.setdefault("GEMINI_API_KEY", "dummy-key-for-test")

import pytest
import numpy as np

from app.services import (
    _parse_title_description,
    _description_mentions_real_songs,
    parse_verse,
    cosine_similarity_manual,
)


# ---------------------------------------------------------------------------
# _parse_title_description
# ---------------------------------------------------------------------------

class TestParseTitleDescription:
    """AI 응답 텍스트에서 제목/설명을 추출하는 헬퍼 함수 테스트"""

    def test_json_정상응답(self):
        """JSON 형식 정상 응답에서 method='json', title/description 추출"""
        raw = '{"title": "사랑의 주님", "description": "하나님의 사랑을 노래합니다"}'
        title, desc, method = _parse_title_description(raw, "fb_title", "fb_desc")
        assert method == "json"
        assert title == "사랑의 주님"
        assert desc == "하나님의 사랑을 노래합니다"

    def test_json_코드펜스_감싼_응답(self):
        """코드펜스(```json ... ```)로 감싼 응답도 method='json' 으로 파싱"""
        raw = '```json\n{"title": "은혜", "description": "은혜의 시간입니다"}\n```'
        title, desc, method = _parse_title_description(raw, "fb_title", "fb_desc")
        assert method == "json"
        assert title == "은혜"
        assert desc == "은혜의 시간입니다"

    def test_line_한글_패턴(self):
        """'제목: ...\n설명: ...' 한글 줄 패턴 → method='line'"""
        raw = "제목: 찬양의 노래\n설명: 하나님을 찬양하는 시간입니다"
        title, desc, method = _parse_title_description(raw, "fb_title", "fb_desc")
        assert method == "line"
        assert title == "찬양의 노래"
        assert "하나님을 찬양하는 시간" in desc

    def test_line_영문_패턴(self):
        """'Title: ...\nDescription: ...' 영문 줄 패턴 → method='line'"""
        raw = "Title: Praise\nDescription: Praise the Lord with singing"
        title, desc, method = _parse_title_description(raw, "fb_title", "fb_desc")
        assert method == "line"
        assert title == "Praise"
        assert "Praise the Lord" in desc

    def test_line_첫줄_제목_나머지_설명(self):
        """첫 줄을 제목, 나머지 줄을 설명으로 처리 → method='line'"""
        raw = "주님의 사랑\n이 곡은 하나님의 사랑을 노래합니다\n은혜로운 시간이 됩니다"
        title, desc, method = _parse_title_description(raw, "fb_title", "fb_desc")
        assert method == "line"
        assert title == "주님의 사랑"
        assert "하나님의 사랑" in desc

    def test_fallback_빈문자열(self):
        """빈 문자열 입력 → method='fallback', fallback 값 반환"""
        title, desc, method = _parse_title_description("", "fb_title", "fb_desc")
        assert method == "fallback"
        assert title == "fb_title"
        assert desc == "fb_desc"

    def test_fallback_단일줄_무관한_텍스트(self):
        """1줄짜리 무관한 텍스트 → 줄 패턴 불일치, method='fallback'"""
        title, desc, method = _parse_title_description(
            "완전무관한텍스트한줄", "fb_title", "fb_desc"
        )
        assert method == "fallback"
        assert title == "fb_title"
        assert desc == "fb_desc"


# ---------------------------------------------------------------------------
# _description_mentions_real_songs
# ---------------------------------------------------------------------------

class TestDescriptionMentionsRealSongs:
    """description 내 인용 곡명이 실제 DB 곡명과 fuzzy match하는지 검증하는 함수 테스트"""

    SONGS = ["주의 인자하심", "새벽 이슬같은 주의 청년들이", "주님의 은혜"]

    def test_인용없는_설명은_통과(self):
        """인용 부호가 없는 설명 → 환각 위험 없음, True 반환"""
        desc = "이 콘티는 주님을 찬양하는 시간을 담고 있습니다"
        assert _description_mentions_real_songs(desc, self.SONGS) is True

    def test_정확한_곡명_인용은_통과(self):
        """실제 곡명을 정확히 인용한 경우 → True 반환"""
        desc = '찬양 "주의 인자하심"이 이 시간을 이끌어갑니다'
        assert _description_mentions_real_songs(desc, self.SONGS) is True

    def test_유사_곡명_임계값_이상_통과(self):
        """실제 곡명과 유사한(SequenceMatcher ratio >= 0.6) 인용 → True 반환"""
        # "주의 인자하시" → "주의 인자하심" 과 ratio > 0.6
        desc = '찬양 "주의 인자하시"를 부릅니다'
        assert _description_mentions_real_songs(desc, self.SONGS) is True

    def test_완전_무관한_곡명_인용은_실패(self):
        """실제 DB에 없는 완전 무관한 곡명 인용 → False 반환"""
        desc = '노래 "완전히무관한가수의노래제목"을 함께 부릅니다'
        assert _description_mentions_real_songs(desc, self.SONGS) is False


# ---------------------------------------------------------------------------
# parse_verse
# ---------------------------------------------------------------------------

class TestParseVerse:
    """성경 구절 문자열 파싱 함수 테스트"""

    def test_창세기_파싱(self):
        """'창세기1:1' → ('창세기', 1, 1) 튜플 반환"""
        book, chapter, verse = parse_verse("창세기1:1")
        assert book == "창세기"
        assert chapter == 1
        assert verse == 1

    def test_요한복음_파싱(self):
        """'요한복음3:16' → ('요한복음', 3, 16) 튜플 반환"""
        book, chapter, verse = parse_verse("요한복음3:16")
        assert book == "요한복음"
        assert chapter == 3
        assert verse == 16

    def test_잘못된_형식_ValueError(self):
        """올바르지 않은 형식 입력 → ValueError 발생"""
        with pytest.raises(ValueError):
            parse_verse("잘못된형식")


# ---------------------------------------------------------------------------
# cosine_similarity_manual
# ---------------------------------------------------------------------------

class TestCosineSimilarityManual:
    """코사인 유사도 계산 함수 테스트"""

    def test_동일_벡터_유사도_1(self):
        """완전히 동일한 벡터 → 유사도 1.0"""
        v1 = np.array([1.0, 0.0, 0.0])
        v2_matrix = np.array([[1.0, 0.0, 0.0]])
        result = cosine_similarity_manual(v1, v2_matrix)
        assert pytest.approx(result[0], abs=1e-6) == 1.0

    def test_직교_벡터_유사도_0(self):
        """서로 직교하는 벡터 → 유사도 0.0"""
        v1 = np.array([1.0, 0.0, 0.0])
        v2_matrix = np.array([[0.0, 1.0, 0.0]])
        result = cosine_similarity_manual(v1, v2_matrix)
        assert pytest.approx(result[0], abs=1e-6) == 0.0

    def test_영벡터_v1_zeros_반환(self):
        """v1이 영벡터(zero vector) → np.zeros 배열 반환"""
        v1 = np.zeros(3)
        v2_matrix = np.array([[1.0, 0.0, 0.0]])
        result = cosine_similarity_manual(v1, v2_matrix)
        assert isinstance(result, np.ndarray)
        assert np.all(result == 0.0)
