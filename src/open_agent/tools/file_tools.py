"""File operation tools for reading, writing, and manipulating files."""

import os
import shutil
from pathlib import Path
from typing import List, Optional
import aiofiles
import json
import yaml

from .base import Tool, ToolDefinition, ToolParameter, ToolResult


class ReadFileTool(Tool):
    """Tool for reading file contents."""
    
    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="read_file",
            description="Read the contents of a file",
            parameters=[
                ToolParameter(
                    name="path",
                    type="string",
                    description="Path to the file to read"
                ),
                ToolParameter(
                    name="encoding",
                    type="string",
                    description="File encoding",
                    required=False,
                    default="utf-8"
                )
            ]
        )
    
    async def execute(self, path: str, encoding: str = "utf-8") -> ToolResult:
        try:
            file_path = Path(path).expanduser().resolve()
            
            if not file_path.exists():
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"File not found: {path}"
                )
            
            if not file_path.is_file():
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Path is not a file: {path}"
                )
            
            async with aiofiles.open(file_path, mode='r', encoding=encoding) as f:
                content = await f.read()
            
            return ToolResult(
                success=True,
                output=content,
                metadata={
                    "path": str(file_path),
                    "size": file_path.stat().st_size,
                    "lines": content.count('\n') + 1
                }
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=str(e)
            )


class WriteFileTool(Tool):
    """Tool for writing content to a file."""
    
    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="write_file",
            description="Write content to a file (creates or overwrites)",
            parameters=[
                ToolParameter(
                    name="path",
                    type="string",
                    description="Path to the file to write"
                ),
                ToolParameter(
                    name="content",
                    type="string",
                    description="Content to write to the file"
                ),
                ToolParameter(
                    name="encoding",
                    type="string",
                    description="File encoding",
                    required=False,
                    default="utf-8"
                ),
                ToolParameter(
                    name="create_dirs",
                    type="boolean",
                    description="Create parent directories if they don't exist",
                    required=False,
                    default=True
                )
            ]
        )
    
    async def execute(self, path: str, content: str, 
                     encoding: str = "utf-8", create_dirs: bool = True) -> ToolResult:
        try:
            file_path = Path(path).expanduser().resolve()
            
            if create_dirs:
                file_path.parent.mkdir(parents=True, exist_ok=True)
            
            async with aiofiles.open(file_path, mode='w', encoding=encoding) as f:
                await f.write(content)
            
            return ToolResult(
                success=True,
                output=f"File written successfully: {file_path}",
                metadata={
                    "path": str(file_path),
                    "size": len(content.encode(encoding)),
                    "lines": content.count('\n') + 1
                }
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=str(e)
            )


class AppendFileTool(Tool):
    """Tool for appending content to a file."""
    
    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="append_file",
            description="Append content to an existing file",
            parameters=[
                ToolParameter(
                    name="path",
                    type="string",
                    description="Path to the file to append to"
                ),
                ToolParameter(
                    name="content",
                    type="string",
                    description="Content to append to the file"
                ),
                ToolParameter(
                    name="encoding",
                    type="string",
                    description="File encoding",
                    required=False,
                    default="utf-8"
                )
            ]
        )
    
    async def execute(self, path: str, content: str, encoding: str = "utf-8") -> ToolResult:
        try:
            file_path = Path(path).expanduser().resolve()
            
            if not file_path.exists():
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"File not found: {path}"
                )
            
            async with aiofiles.open(file_path, mode='a', encoding=encoding) as f:
                await f.write(content)
            
            return ToolResult(
                success=True,
                output=f"Content appended successfully to: {file_path}",
                metadata={
                    "path": str(file_path),
                    "appended_size": len(content.encode(encoding))
                }
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=str(e)
            )


class ListDirectoryTool(Tool):
    """Tool for listing directory contents."""
    
    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="list_directory",
            description="List contents of a directory",
            parameters=[
                ToolParameter(
                    name="path",
                    type="string",
                    description="Path to the directory",
                    required=False,
                    default="."
                ),
                ToolParameter(
                    name="recursive",
                    type="boolean",
                    description="List recursively",
                    required=False,
                    default=False
                ),
                ToolParameter(
                    name="pattern",
                    type="string",
                    description="Filter pattern (e.g., '*.py')",
                    required=False,
                    default=None
                )
            ]
        )
    
    async def execute(self, path: str = ".", recursive: bool = False, 
                     pattern: Optional[str] = None) -> ToolResult:
        try:
            dir_path = Path(path).expanduser().resolve()
            
            if not dir_path.exists():
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Directory not found: {path}"
                )
            
            if not dir_path.is_dir():
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Path is not a directory: {path}"
                )
            
            files = []
            if recursive:
                if pattern:
                    items = dir_path.rglob(pattern)
                else:
                    items = dir_path.rglob("*")
            else:
                if pattern:
                    items = dir_path.glob(pattern)
                else:
                    items = dir_path.iterdir()
            
            for item in items:
                relative = item.relative_to(dir_path)
                files.append({
                    "path": str(relative),
                    "type": "directory" if item.is_dir() else "file",
                    "size": item.stat().st_size if item.is_file() else None
                })
            
            return ToolResult(
                success=True,
                output=files,
                metadata={
                    "directory": str(dir_path),
                    "count": len(files),
                    "recursive": recursive,
                    "pattern": pattern
                }
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=str(e)
            )


class DeleteFileTool(Tool):
    """Tool for deleting files or directories."""
    
    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="delete_file",
            description="Delete a file or directory",
            parameters=[
                ToolParameter(
                    name="path",
                    type="string",
                    description="Path to the file or directory to delete"
                ),
                ToolParameter(
                    name="force",
                    type="boolean",
                    description="Force delete even if directory is not empty",
                    required=False,
                    default=False
                )
            ]
        )
    
    async def execute(self, path: str, force: bool = False) -> ToolResult:
        try:
            file_path = Path(path).expanduser().resolve()
            
            if not file_path.exists():
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Path not found: {path}"
                )
            
            if file_path.is_file():
                file_path.unlink()
            elif file_path.is_dir():
                if force:
                    shutil.rmtree(file_path)
                else:
                    file_path.rmdir()
            
            return ToolResult(
                success=True,
                output=f"Deleted successfully: {file_path}",
                metadata={"path": str(file_path)}
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=str(e)
            )


class MoveFileTool(Tool):
    """Tool for moving or renaming files."""
    
    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="move_file",
            description="Move or rename a file or directory",
            parameters=[
                ToolParameter(
                    name="source",
                    type="string",
                    description="Source path"
                ),
                ToolParameter(
                    name="destination",
                    type="string",
                    description="Destination path"
                ),
                ToolParameter(
                    name="overwrite",
                    type="boolean",
                    description="Overwrite if destination exists",
                    required=False,
                    default=False
                )
            ]
        )
    
    async def execute(self, source: str, destination: str, 
                     overwrite: bool = False) -> ToolResult:
        try:
            src_path = Path(source).expanduser().resolve()
            dst_path = Path(destination).expanduser().resolve()
            
            if not src_path.exists():
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Source not found: {source}"
                )
            
            if dst_path.exists() and not overwrite:
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Destination already exists: {destination}"
                )
            
            shutil.move(str(src_path), str(dst_path))
            
            return ToolResult(
                success=True,
                output=f"Moved successfully: {src_path} -> {dst_path}",
                metadata={
                    "source": str(src_path),
                    "destination": str(dst_path)
                }
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=str(e)
            )


class CopyFileTool(Tool):
    """Tool for copying files or directories."""
    
    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="copy_file",
            description="Copy a file or directory",
            parameters=[
                ToolParameter(
                    name="source",
                    type="string",
                    description="Source path"
                ),
                ToolParameter(
                    name="destination",
                    type="string",
                    description="Destination path"
                ),
                ToolParameter(
                    name="overwrite",
                    type="boolean",
                    description="Overwrite if destination exists",
                    required=False,
                    default=False
                )
            ]
        )
    
    async def execute(self, source: str, destination: str, 
                     overwrite: bool = False) -> ToolResult:
        try:
            src_path = Path(source).expanduser().resolve()
            dst_path = Path(destination).expanduser().resolve()
            
            if not src_path.exists():
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Source not found: {source}"
                )
            
            if dst_path.exists() and not overwrite:
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Destination already exists: {destination}"
                )
            
            if src_path.is_file():
                shutil.copy2(str(src_path), str(dst_path))
            else:
                shutil.copytree(str(src_path), str(dst_path), dirs_exist_ok=overwrite)
            
            return ToolResult(
                success=True,
                output=f"Copied successfully: {src_path} -> {dst_path}",
                metadata={
                    "source": str(src_path),
                    "destination": str(dst_path)
                }
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=str(e)
            )
