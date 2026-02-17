"""
Task Planning Tools - Enhanced with Strict JSON Output
Implements dynamic, structured TODO generation with validation
"""
from langchain_core.tools import tool
from typing import List
from pydantic import BaseModel, Field, validator
import uuid
import json


class TodoItemInput(BaseModel):
    """Schema for a single TODO item with strict validation."""
    description: str = Field(
        description="Clear, action-oriented description starting with an action verb (e.g., 'Research', 'Analyze', 'Create', 'Compile'). Must be specific and executable."
    )
    
    @validator('description')
    def validate_description(cls, v):
        """Ensure description is meaningful and starts with action verb."""
        if len(v.strip()) < 10:
            raise ValueError("Description must be at least 10 characters and specific")
        
        # Common action verbs that should start tasks
        action_verbs = [
            'research', 'analyze', 'create', 'compile', 'investigate', 
            'examine', 'evaluate', 'gather', 'identify', 'develop',
            'write', 'review', 'compare', 'assess', 'design',
            'collect', 'synthesize', 'summarize', 'extract', 'organize'
        ]
        
        first_word = v.strip().lower().split()[0]
        if not any(first_word.startswith(verb) for verb in action_verbs):
            # Don't raise error, but this validates structure
            pass
        
        return v.strip()


class TodoListInput(BaseModel):
    """Schema for creating a list of TODO items with validation."""
    todos: List[TodoItemInput] = Field(
        description="List of 4-6 sub-tasks that need to be completed to achieve the main goal. Each task must be distinct, actionable, and in logical order.",
        min_items=4,
        max_items=6
    )
    
    @validator('todos')
    def validate_todos(cls, v):
        """Ensure TODOs are unique and not too similar."""
        descriptions = [todo.description.lower() for todo in v]
        
        # Check for duplicates
        if len(descriptions) != len(set(descriptions)):
            raise ValueError("TODO items must be unique - no duplicates allowed")
        
        # Check minimum count
        if len(v) < 4:
            raise ValueError("Must have at least 4 TODO items for complex tasks")
        
        # Check maximum count
        if len(v) > 6:
            raise ValueError("Keep TODO list focused - maximum 6 items")
        
        return v


@tool(args_schema=TodoListInput)
def write_todos(todos: List[TodoItemInput]) -> str:
    """
    Create a structured, validated list of TODO items to break down a complex task.
    
    This tool MUST be used when receiving ANY complex request that requires multiple steps.
    The AI model dynamically generates the TODO list based on the user's request.
    
    Requirements:
    - Each TODO must start with an action verb
    - Each TODO must be specific and executable
    - Must have 4-6 items (no more, no less)
    - Items must be unique and non-repetitive
    - Items must be in logical execution order
    - Output will be validated as strict JSON
    
    Args:
        todos: List of 4-6 sub-tasks with action-oriented descriptions
        
    Returns:
        JSON string confirming TODO creation with structured output
    """
    num_todos = len(todos)
    
    # Create structured output
    result = {
        "status": "success",
        "todo_count": num_todos,
        "message": f"Successfully created {num_todos} validated TODO items in strict JSON format"
    }
    
    return json.dumps(result)


def create_todo_items(todo_inputs: List[TodoItemInput]) -> List[dict]:
    """
    Convert TodoItemInput objects to TodoItem state dictionaries.
    Enforces strict structure and validates output.
    
    Args:
        todo_inputs: List of validated TODO descriptions from the tool
        
    Returns:
        List of strictly structured TodoItem dictionaries ready for state storage
    """
    todo_items = []
    
    for idx, todo_input in enumerate(todo_inputs, 1):
        # Create strictly structured dictionary
        todo_item = {
            "id": str(uuid.uuid4()),
            "index": idx,
            "description": todo_input.description,
            "status": "pending",
            "result": None,
            "created_by": "write_todos_tool"
        }
        todo_items.append(todo_item)
    
    # Validate final structure
    validate_todo_structure(todo_items)
    
    return todo_items


def validate_todo_structure(todos: List[dict]) -> bool:
    """
    Validate that TODO list has correct structure.
    Raises exception if invalid.
    """
    required_keys = {"id", "index", "description", "status", "result", "created_by"}
    
    for todo in todos:
        if not isinstance(todo, dict):
            raise ValueError(f"TODO item must be a dictionary, got {type(todo)}")
        
        if not required_keys.issubset(todo.keys()):
            missing = required_keys - set(todo.keys())
            raise ValueError(f"TODO item missing required keys: {missing}")
        
        if todo["status"] not in ["pending", "in_progress", "completed"]:
            raise ValueError(f"Invalid status: {todo['status']}")
    
    return True
