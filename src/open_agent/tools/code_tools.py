"""Code execution and manipulation tools."""

import ast
import asyncio
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List
import black
import autopep8

from .base import Tool, ToolDefinition, ToolParameter, ToolResult


class ExecuteCodeTool(Tool):
    """Tool for executing code in various languages."""
    
    def __init__(self, timeout: int = 30, sandbox: bool = True):
        super().__init__()
        self.timeout = timeout
        self.sandbox = sandbox
    
    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="execute_code",
            description="Execute code in a specified programming language",
            parameters=[
                ToolParameter(
                    name="code",
                    type="string",
                    description="The code to execute"
                ),
                ToolParameter(
                    name="language",
                    type="string",
                    description="Programming language (python, javascript, bash, etc.)",
                    enum=["python", "javascript", "bash", "ruby", "go", "rust", "java"]
                ),
                ToolParameter(
                    name="stdin",
                    type="string",
                    description="Input to provide to the program",
                    required=False,
                    default=""
                ),
                ToolParameter(
                    name="timeout",
                    type="integer",
                    description="Execution timeout in seconds",
                    required=False,
                    default=30
                )
            ]
        )
    
    async def execute(self, code: str, language: str, 
                     stdin: str = "", timeout: Optional[int] = None) -> ToolResult:
        timeout = timeout or self.timeout
        
        try:
            if language == "python":
                return await self._execute_python(code, stdin, timeout)
            elif language == "javascript":
                return await self._execute_javascript(code, stdin, timeout)
            elif language == "bash":
                return await self._execute_bash(code, stdin, timeout)
            elif language == "ruby":
                return await self._execute_ruby(code, stdin, timeout)
            elif language == "go":
                return await self._execute_go(code, stdin, timeout)
            else:
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Unsupported language: {language}"
                )
        except asyncio.TimeoutError:
            return ToolResult(
                success=False,
                output=None,
                error=f"Execution timed out after {timeout} seconds"
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=str(e)
            )
    
    async def _execute_python(self, code: str, stdin: str, timeout: int) -> ToolResult:
        """Execute Python code."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name
        
        try:
            process = await asyncio.create_subprocess_exec(
                sys.executable, temp_file,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(input=stdin.encode() if stdin else None),
                timeout=timeout
            )
            
            return ToolResult(
                success=process.returncode == 0,
                output=stdout.decode('utf-8'),
                error=stderr.decode('utf-8') if stderr else None,
                metadata={"return_code": process.returncode}
            )
        finally:
            Path(temp_file).unlink(missing_ok=True)
    
    async def _execute_javascript(self, code: str, stdin: str, timeout: int) -> ToolResult:
        """Execute JavaScript code using Node.js."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
            f.write(code)
            temp_file = f.name
        
        try:
            process = await asyncio.create_subprocess_exec(
                'node', temp_file,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(input=stdin.encode() if stdin else None),
                timeout=timeout
            )
            
            return ToolResult(
                success=process.returncode == 0,
                output=stdout.decode('utf-8'),
                error=stderr.decode('utf-8') if stderr else None,
                metadata={"return_code": process.returncode}
            )
        finally:
            Path(temp_file).unlink(missing_ok=True)
    
    async def _execute_bash(self, code: str, stdin: str, timeout: int) -> ToolResult:
        """Execute Bash script."""
        process = await asyncio.create_subprocess_shell(
            code,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await asyncio.wait_for(
            process.communicate(input=stdin.encode() if stdin else None),
            timeout=timeout
        )
        
        return ToolResult(
            success=process.returncode == 0,
            output=stdout.decode('utf-8'),
            error=stderr.decode('utf-8') if stderr else None,
            metadata={"return_code": process.returncode}
        )
    
    async def _execute_ruby(self, code: str, stdin: str, timeout: int) -> ToolResult:
        """Execute Ruby code."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.rb', delete=False) as f:
            f.write(code)
            temp_file = f.name
        
        try:
            process = await asyncio.create_subprocess_exec(
                'ruby', temp_file,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(input=stdin.encode() if stdin else None),
                timeout=timeout
            )
            
            return ToolResult(
                success=process.returncode == 0,
                output=stdout.decode('utf-8'),
                error=stderr.decode('utf-8') if stderr else None,
                metadata={"return_code": process.returncode}
            )
        finally:
            Path(temp_file).unlink(missing_ok=True)
    
    async def _execute_go(self, code: str, stdin: str, timeout: int) -> ToolResult:
        """Execute Go code."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.go', delete=False) as f:
            f.write(code)
            temp_file = f.name
        
        try:
            process = await asyncio.create_subprocess_exec(
                'go', 'run', temp_file,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(input=stdin.encode() if stdin else None),
                timeout=timeout
            )
            
            return ToolResult(
                success=process.returncode == 0,
                output=stdout.decode('utf-8'),
                error=stderr.decode('utf-8') if stderr else None,
                metadata={"return_code": process.returncode}
            )
        finally:
            Path(temp_file).unlink(missing_ok=True)


class RunCommandTool(Tool):
    """Tool for running shell commands."""
    
    def __init__(self, allowed_commands: Optional[List[str]] = None):
        super().__init__()
        self.allowed_commands = allowed_commands
    
    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="run_command",
            description="Run a shell command",
            parameters=[
                ToolParameter(
                    name="command",
                    type="string",
                    description="The command to run"
                ),
                ToolParameter(
                    name="cwd",
                    type="string",
                    description="Working directory for the command",
                    required=False,
                    default=None
                ),
                ToolParameter(
                    name="timeout",
                    type="integer",
                    description="Command timeout in seconds",
                    required=False,
                    default=30
                ),
                ToolParameter(
                    name="shell",
                    type="boolean",
                    description="Whether to run through shell",
                    required=False,
                    default=True
                )
            ]
        )
    
    async def execute(self, command: str, cwd: Optional[str] = None,
                     timeout: int = 30, shell: bool = True) -> ToolResult:
        try:
            # Check if command is allowed
            if self.allowed_commands:
                cmd_parts = command.split()
                if cmd_parts and cmd_parts[0] not in self.allowed_commands:
                    return ToolResult(
                        success=False,
                        output=None,
                        error=f"Command '{cmd_parts[0]}' is not allowed"
                    )
            
            if shell:
                process = await asyncio.create_subprocess_shell(
                    command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=cwd
                )
            else:
                cmd_parts = command.split()
                process = await asyncio.create_subprocess_exec(
                    *cmd_parts,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=cwd
                )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout
            )
            
            return ToolResult(
                success=process.returncode == 0,
                output=stdout.decode('utf-8'),
                error=stderr.decode('utf-8') if stderr else None,
                metadata={
                    "return_code": process.returncode,
                    "command": command,
                    "cwd": cwd
                }
            )
        except asyncio.TimeoutError:
            return ToolResult(
                success=False,
                output=None,
                error=f"Command timed out after {timeout} seconds"
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=str(e)
            )


class FormatCodeTool(Tool):
    """Tool for formatting code."""
    
    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="format_code",
            description="Format code according to language standards",
            parameters=[
                ToolParameter(
                    name="code",
                    type="string",
                    description="The code to format"
                ),
                ToolParameter(
                    name="language",
                    type="string",
                    description="Programming language",
                    enum=["python", "javascript", "json", "yaml"]
                )
            ]
        )
    
    async def execute(self, code: str, language: str) -> ToolResult:
        try:
            if language == "python":
                formatted = black.format_str(code, mode=black.Mode())
                return ToolResult(success=True, output=formatted)
            elif language == "javascript":
                # Would need prettier or similar installed
                return ToolResult(
                    success=False,
                    output=None,
                    error="JavaScript formatting not yet implemented"
                )
            elif language == "json":
                import json
                parsed = json.loads(code)
                formatted = json.dumps(parsed, indent=2)
                return ToolResult(success=True, output=formatted)
            elif language == "yaml":
                import yaml
                parsed = yaml.safe_load(code)
                formatted = yaml.dump(parsed, default_flow_style=False, indent=2)
                return ToolResult(success=True, output=formatted)
            else:
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Unsupported language: {language}"
                )
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=str(e)
            )


class AnalyzeCodeTool(Tool):
    """Tool for analyzing code for issues and patterns."""
    
    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="analyze_code",
            description="Analyze code for syntax errors, complexity, and patterns",
            parameters=[
                ToolParameter(
                    name="code",
                    type="string",
                    description="The code to analyze"
                ),
                ToolParameter(
                    name="language",
                    type="string",
                    description="Programming language",
                    enum=["python", "javascript"]
                )
            ]
        )
    
    async def execute(self, code: str, language: str) -> ToolResult:
        try:
            if language == "python":
                return await self._analyze_python(code)
            elif language == "javascript":
                return await self._analyze_javascript(code)
            else:
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Unsupported language: {language}"
                )
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=str(e)
            )
    
    async def _analyze_python(self, code: str) -> ToolResult:
        """Analyze Python code."""
        analysis = {
            "syntax_valid": False,
            "errors": [],
            "warnings": [],
            "metrics": {}
        }
        
        try:
            # Parse the code
            tree = ast.parse(code)
            analysis["syntax_valid"] = True
            
            # Count various elements
            class CodeAnalyzer(ast.NodeVisitor):
                def __init__(self):
                    self.functions = 0
                    self.classes = 0
                    self.imports = 0
                    self.lines = code.count('\n') + 1
                
                def visit_FunctionDef(self, node):
                    self.functions += 1
                    self.generic_visit(node)
                
                def visit_AsyncFunctionDef(self, node):
                    self.functions += 1
                    self.generic_visit(node)
                
                def visit_ClassDef(self, node):
                    self.classes += 1
                    self.generic_visit(node)
                
                def visit_Import(self, node):
                    self.imports += 1
                    self.generic_visit(node)
                
                def visit_ImportFrom(self, node):
                    self.imports += 1
                    self.generic_visit(node)
            
            analyzer = CodeAnalyzer()
            analyzer.visit(tree)
            
            analysis["metrics"] = {
                "lines": analyzer.lines,
                "functions": analyzer.functions,
                "classes": analyzer.classes,
                "imports": analyzer.imports
            }
            
        except SyntaxError as e:
            analysis["syntax_valid"] = False
            analysis["errors"].append(f"Syntax error at line {e.lineno}: {e.msg}")
        
        return ToolResult(
            success=True,
            output=analysis
        )
    
    async def _analyze_javascript(self, code: str) -> ToolResult:
        """Analyze JavaScript code."""
        # Basic analysis for JavaScript
        analysis = {
            "metrics": {
                "lines": code.count('\n') + 1,
                "functions": code.count('function '),
                "arrow_functions": code.count('=>'),
                "classes": code.count('class ')
            }
        }
        
        return ToolResult(
            success=True,
            output=analysis
        )
