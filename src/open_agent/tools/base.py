"""Base tool interface and registry for the agent system."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, Callable
from pydantic import BaseModel, Field
import json
import inspect
from enum import Enum


class ToolParameter(BaseModel):
    """Definition of a tool parameter."""
    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None
    enum: Optional[List[Any]] = None


class ToolDefinition(BaseModel):
    """Definition of a tool for LLM consumption."""
    name: str
    description: str
    parameters: List[ToolParameter]
    returns: str = "string"
    
    def to_openai_format(self) -> Dict:
        """Convert to OpenAI function calling format."""
        properties = {}
        required = []
        
        for param in self.parameters:
            prop = {
                "type": param.type,
                "description": param.description
            }
            if param.enum:
                prop["enum"] = param.enum
            properties[param.name] = prop
            
            if param.required:
                required.append(param.name)
        
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }


class ToolResult(BaseModel):
    """Result from a tool execution."""
    success: bool
    output: Any
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Tool(ABC):
    """Base class for all tools."""
    
    def __init__(self):
        self.name = self.__class__.__name__.lower().replace("tool", "")
    
    @abstractmethod
    def get_definition(self) -> ToolDefinition:
        """Return the tool definition for LLM consumption."""
        pass
    
    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given parameters."""
        pass
    
    def validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and convert parameters."""
        definition = self.get_definition()
        validated = {}
        
        for param_def in definition.parameters:
            if param_def.required and param_def.name not in params:
                raise ValueError(f"Missing required parameter: {param_def.name}")
            
            if param_def.name in params:
                value = params[param_def.name]
                # Basic type conversion
                if param_def.type == "integer" and not isinstance(value, int):
                    value = int(value)
                elif param_def.type == "number" and not isinstance(value, (int, float)):
                    value = float(value)
                elif param_def.type == "boolean" and not isinstance(value, bool):
                    value = str(value).lower() == "true"
                elif param_def.type == "array" and isinstance(value, str):
                    value = json.loads(value)
                
                validated[param_def.name] = value
            elif param_def.default is not None:
                validated[param_def.name] = param_def.default
        
        return validated


class ToolRegistry:
    """Registry for managing available tools."""
    
    def __init__(self):
        self._tools: Dict[str, Tool] = {}
    
    def register(self, tool: Tool, name: Optional[str] = None):
        """Register a tool."""
        tool_name = name or tool.name
        self._tools[tool_name] = tool
    
    def register_class(self, tool_class: Type[Tool], name: Optional[str] = None):
        """Register a tool class."""
        tool = tool_class()
        self.register(tool, name)
    
    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(name)
    
    def list(self) -> List[str]:
        """List all available tool names."""
        return list(self._tools.keys())
    
    def get_definitions(self) -> List[ToolDefinition]:
        """Get all tool definitions."""
        return [tool.get_definition() for tool in self._tools.values()]
    
    def get_openai_tools(self) -> List[Dict]:
        """Get tools in OpenAI function calling format."""
        return [tool.get_definition().to_openai_format() for tool in self._tools.values()]
    
    async def execute(self, name: str, **kwargs) -> ToolResult:
        """Execute a tool by name."""
        tool = self.get(name)
        if not tool:
            return ToolResult(
                success=False,
                output=None,
                error=f"Tool '{name}' not found"
            )
        
        try:
            validated_params = tool.validate_params(kwargs)
            return await tool.execute(**validated_params)
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=str(e)
            )


# Global registry instance
registry = ToolRegistry()


def tool(name: Optional[str] = None, description: Optional[str] = None):
    """Decorator to register a function as a tool."""
    def decorator(func: Callable):
        class FunctionTool(Tool):
            def get_definition(self) -> ToolDefinition:
                sig = inspect.signature(func)
                parameters = []
                
                for param_name, param in sig.parameters.items():
                    if param_name == "self":
                        continue
                    
                    param_type = "string"  # Default type
                    if param.annotation != inspect.Parameter.empty:
                        if param.annotation == int:
                            param_type = "integer"
                        elif param.annotation == float:
                            param_type = "number"
                        elif param.annotation == bool:
                            param_type = "boolean"
                        elif param.annotation == list:
                            param_type = "array"
                        elif param.annotation == dict:
                            param_type = "object"
                    
                    parameters.append(ToolParameter(
                        name=param_name,
                        type=param_type,
                        description=f"Parameter {param_name}",
                        required=param.default == inspect.Parameter.empty,
                        default=None if param.default == inspect.Parameter.empty else param.default
                    ))
                
                return ToolDefinition(
                    name=name or func.__name__,
                    description=description or func.__doc__ or f"Function {func.__name__}",
                    parameters=parameters
                )
            
            async def execute(self, **kwargs) -> ToolResult:
                try:
                    if inspect.iscoroutinefunction(func):
                        result = await func(**kwargs)
                    else:
                        result = func(**kwargs)
                    return ToolResult(success=True, output=result)
                except Exception as e:
                    return ToolResult(success=False, output=None, error=str(e))
        
        tool_instance = FunctionTool()
        tool_instance.name = name or func.__name__
        registry.register(tool_instance)
        return func
    
    return decorator
