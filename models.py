from typing import Optional
from pydantic import BaseModel, Field


class ActionItem(BaseModel):
    """
    Represents an individual action item extracted from document content.

    Attributes:
        task (str): A clear and concise description of the action or task.
        owner (Optional[str]): The person, team, or entity assigned to execute the task.
        deadline (Optional[str]): Due date, time, or timeframe specified for completion.
        priority (Optional[str]): Urgency level (e.g., High, Medium, Low), if specified or inferred.
        source (str): Contextual excerpt or quote from the source text from which the action item was identified.
    """

    task: str = Field(
        ...,
        description="A clear and concise description of the specific task or action to be completed.",
    )
    owner: Optional[str] = Field(
        default=None,
        description="The individual, team, or entity assigned or responsible for this task.",
    )
    deadline: Optional[str] = Field(
        default=None,
        description="The due date, deadline, or timeframe specified for task completion.",
    )
    priority: Optional[str] = Field(
        default=None,
        description="The priority level of the task (e.g., High, Medium, Low), if mentioned or inferred.",
    )
    source: str = Field(
        ...,
        description="The exact snippet, quote, or sentence from the source document that justifies this action item.",
    )


class ActionItemList(BaseModel):
    """
    Represents a collection of extracted action items.

    Attributes:
        action_items (list[ActionItem]): A list containing all extracted ActionItem instances.
    """

    action_items: list[ActionItem] = Field(
        default_factory=list,
        description="List of action items extracted from the document context.",
    )
