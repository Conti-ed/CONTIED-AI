# -*- coding: utf-8 -*-
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field

class ContiRequest(BaseModel):
    keywords: List[str] = Field(
        ...,
        description="콘티의 주제나 분위기를 나타내는 키워드 리스트",
        examples=[["사랑", "창조", "기쁨"]]
    )
    bible_verse_range: Optional[str] = Field(
        None,
        description="성경 구절 범위 (형식: 권장:절~권장:절)",
        examples=["창세기1:1~창세기1:10"]
    )

class SongItem(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: int
    video_id: Optional[str] = Field(None, alias='video_id')
    title: Optional[str] = None
    artist: Optional[str] = None

class ContiResponse(BaseModel):
    title: str
    description: str
    songs: List[SongItem]
