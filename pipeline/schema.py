
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

class ScenarioAndTask(BaseModel):
    scenario: str = Field(description="Description of the application scenario or domain")
    task: str = Field(description="Specific task or problem being addressed")

class ProblemsAndValue(BaseModel):
    problems: List[str] = Field(description="List of problems addressed by the paper")
    value: str = Field(description="Why these problems are important and valuable to solve")

class ProposedMethods(BaseModel):
    main_method: str = Field(description="Primary method or approach proposed")
    key_components: List[str] = Field(description="Key components of the proposed method")
    technical_details: str = Field(description="Brief description of how the method works")

class Innovations(BaseModel):
    main_innovations: List[str] = Field(description="Main innovations of the paper")
    novelty_description: str = Field(description="What makes this work novel compared to existing approaches")

class ExperimentalSetup(BaseModel):
    datasets: List[str] = Field(description="Datasets used in the experiments")
    models: List[str] = Field(description="Base models used in the experiments")
    baselines: List[str] = Field(description="Baseline methods compared against")
    evaluation_metrics: List[str] = Field(description="Metrics used for evaluation")

class PaperAnalysisSchema(BaseModel):
    scenario_and_task: ScenarioAndTask = Field(description="The scenario and task information")
    problems_and_value: ProblemsAndValue = Field(description="Problems addressed and their value")
    proposed_methods: ProposedMethods = Field(description="Methods proposed by the paper")
    innovations: Innovations = Field(description="Main innovations of the paper")
    experimental_setup: ExperimentalSetup = Field(description="Experimental setup and evaluation")

class SectionMappingSchema(BaseModel):
    parsed_sections: List[str] = Field(description="The parsed sections of the paper")

class TopicResult(BaseModel):
    topic: str = Field(description="The name of the topic/research area")
    ids: List[int] = Field(description="List of paper IDs belonging to this topic")

class PaperTopicClassificationSchema(BaseModel):
    results: List[TopicResult] = Field(description="List of topics with their assigned paper IDs")
