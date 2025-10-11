
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
