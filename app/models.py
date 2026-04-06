from typing import List, Optional
from pydantic import BaseModel, Field

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
    id: int

class ContiResponse(BaseModel):
    title: str
    description: str
    songs: List[SongItem]
