from langgraph.graph import add_messages
from typing_extensions import Annotated
import operator
from typing import TypedDict, Optional, Dict, Any

class AgenticScienceState(TypedDict):
    messages: Annotated[list, add_messages]
    paper_raw_info: Annotated[list, operator.add]
    paper_processed_info: Annotated[list, operator.add]
    topic_notes: Annotated[list, operator.add]
    stats: Optional[dict] = None
    summary: Annotated[list, operator.add]

class PaperState(TypedDict):
    title: str
    abstract: str
    authors: list[str]
    published_date: str
    source: str
    link: str
    processed_result: Optional[dict]

