
from pydantic import BaseModel
from typing import List

from pydantic import BaseModel, Field
from typing import List

class Author(BaseModel):
    name: str = Field(description="The name of the author")
    org: list[str] = Field(description="The organizations/affiliations of the author")
    misc: list[str] = Field(description="The miscellaneous information of the author")

class AuthorsSchema(BaseModel):
    authors: list[Author] = Field(description="The authors of the paper")

class Keywords(BaseModel):
    keywords: list[str] = Field(description="The keywords of the paper")

class PaperAnalysisSchema(BaseModel):
    scenario: str = Field(description="The scenario information")
    value: str = Field(description="The value information")
    insight: str = Field(description="The insight information")
    keywords: Keywords = Field(description="The keywords of the paper")
    
class SectionMappingSchema(BaseModel):
    parsed_sections: List[str] = Field(description="The parsed sections of the paper")

class TopicResult(BaseModel):
    topic: str = Field(description="The name of the topic/research area")
    keywords: List[str] = Field(description="The keywords of the topic")

class PaperTopicClassificationSchema(BaseModel):
    results: List[TopicResult] = Field(description="List of topics with their assigned paper IDs")

class PaperAttentionScoreSchema(BaseModel):
    score: int = Field(description="The attention score of the paper")

class TopicSchema(BaseModel):
    topics: List[TopicResult] = Field(description="List of topics with their assigned keywords")
    