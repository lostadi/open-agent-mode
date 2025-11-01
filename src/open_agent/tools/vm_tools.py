"""VM-integrated tools for agent operations."""

import json

from .base import Tool, ToolDefinition, ToolParameter, ToolResult
from ..environment.vm_manager import VMEnvironment


class VMExecuteCodeTool(Tool):
    """Execute code in the VM environment."""
    
    def __init__(self, vm: VMEnvironment):
        super().__init__()
        self.vm = vm
        self.name = "vm_execute_code"
    
    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="vm_execute_code",
            description="Execute code in the isolated VM environment",
            parameters=[
                ToolParameter(
                    name="code",
                    type="string",
                    description="Code to execute"
                ),
                ToolParameter(
                    name="language",
                    type="string",
                    description="Programming language",
                    enum=["python", "javascript", "bash", "ruby", "go", "rust"]
                )
            ]
        )
    
    async def execute(self, code: str, language: str) -> ToolResult:
        exit_code, stdout, stderr = await self.vm.execute_code(code, language)
        
        return ToolResult(
            success=exit_code == 0,
            output=stdout,
            error=stderr if exit_code != 0 else None,
            metadata={
                "exit_code": exit_code,
                "language": language
            }
        )


class VMRunCommandTool(Tool):
    """Run shell commands in the VM environment."""
    
    def __init__(self, vm: VMEnvironment):
        super().__init__()
        self.vm = vm
        self.name = "vm_run_command"
    
    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="vm_run_command",
            description="Run a shell command in the VM environment",
            parameters=[
                ToolParameter(
                    name="command",
                    type="string",
                    description="Command to execute"
                ),
                ToolParameter(
                    name="timeout",
                    type="integer",
                    description="Timeout in seconds",
                    required=False,
                    default=30
                )
            ]
        )
    
    async def execute(self, command: str, timeout: int = 30) -> ToolResult:
        exit_code, stdout, stderr = await self.vm.execute_command(command, timeout)
        
        return ToolResult(
            success=exit_code == 0,
            output=stdout,
            error=stderr if exit_code != 0 else None,
            metadata={
                "exit_code": exit_code,
                "command": command
            }
        )


class VMReadFileTool(Tool):
    """Read files from the VM environment."""
    
    def __init__(self, vm: VMEnvironment):
        super().__init__()
        self.vm = vm
        self.name = "vm_read_file"
    
    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="vm_read_file",
            description="Read a file from the VM workspace",
            parameters=[
                ToolParameter(
                    name="path",
                    type="string",
                    description="File path (relative to workspace)"
                )
            ]
        )
    
    async def execute(self, path: str) -> ToolResult:
        content = await self.vm.read_file(path)
        
        if content is None:
            return ToolResult(
                success=False,
                output=None,
                error=f"Failed to read file: {path}"
            )
        
        return ToolResult(
            success=True,
            output=content,
            metadata={"path": path, "size": len(content)}
        )


class VMWriteFileTool(Tool):
    """Write files to the VM environment."""
    
    def __init__(self, vm: VMEnvironment):
        super().__init__()
        self.vm = vm
        self.name = "vm_write_file"
    
    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="vm_write_file",
            description="Write a file to the VM workspace",
            parameters=[
                ToolParameter(
                    name="path",
                    type="string",
                    description="File path (relative to workspace)"
                ),
                ToolParameter(
                    name="content",
                    type="string",
                    description="File content"
                )
            ]
        )
    
    async def execute(self, path: str, content: str) -> ToolResult:
        try:
            await self.vm.write_file(path, content)
            return ToolResult(
                success=True,
                output=f"File written: {path}",
                metadata={"path": path, "size": len(content)}
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=str(e)
            )


class VMListFilesTool(Tool):
    """List files in the VM environment."""
    
    def __init__(self, vm: VMEnvironment):
        super().__init__()
        self.vm = vm
        self.name = "vm_list_files"
    
    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="vm_list_files",
            description="List files in a VM workspace directory",
            parameters=[
                ToolParameter(
                    name="path",
                    type="string",
                    description="Directory path (relative to workspace)",
                    required=False,
                    default=""
                )
            ]
        )
    
    async def execute(self, path: str = "") -> ToolResult:
        files = await self.vm.list_files(path)
        
        return ToolResult(
            success=True,
            output=files,
            metadata={"path": path, "count": len(files)}
        )


class VMInstallPackageTool(Tool):
    """Install packages in the VM environment."""
    
    def __init__(self, vm: VMEnvironment):
        super().__init__()
        self.vm = vm
        self.name = "vm_install_package"
    
    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="vm_install_package",
            description="Install a package in the VM environment",
            parameters=[
                ToolParameter(
                    name="package",
                    type="string",
                    description="Package name to install"
                ),
                ToolParameter(
                    name="manager",
                    type="string",
                    description="Package manager to use",
                    enum=["pip", "npm", "gem", "go", "cargo", "apt"],
                    required=False,
                    default="pip"
                )
            ]
        )
    
    async def execute(self, package: str, manager: str = "pip") -> ToolResult:
        exit_code, stdout, stderr = await self.vm.install_package(package, manager)
        
        return ToolResult(
            success=exit_code == 0,
            output=stdout,
            error=stderr if exit_code != 0 else None,
            metadata={
                "package": package,
                "manager": manager
            }
        )


class VMSystemInfoTool(Tool):
    """Get VM system information."""
    
    def __init__(self, vm: VMEnvironment):
        super().__init__()
        self.vm = vm
        self.name = "vm_system_info"
    
    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="vm_system_info",
            description="Get system information about the VM environment",
            parameters=[]
        )
    
    async def execute(self) -> ToolResult:
        info = await self.vm.get_system_info()
        
        return ToolResult(
            success=True,
            output=info
        )


class VMSnapshotTool(Tool):
    """Create VM snapshots."""
    
    def __init__(self, vm: VMEnvironment):
        super().__init__()
        self.vm = vm
        self.name = "vm_snapshot"
    
    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="vm_snapshot",
            description="Create a snapshot of the current VM state",
            parameters=[]
        )
    
    async def execute(self) -> ToolResult:
        snapshot_id = await self.vm.create_snapshot()
        
        return ToolResult(
            success=True,
            output=f"Snapshot created: {snapshot_id}",
            metadata={"snapshot_id": snapshot_id}
        )


class VMRestoreSnapshotTool(Tool):
    """Restore VM from snapshot."""
    
    def __init__(self, vm: VMEnvironment):
        super().__init__()
        self.vm = vm
        self.name = "vm_restore_snapshot"
    
    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="vm_restore_snapshot",
            description="Restore VM from a previous snapshot",
            parameters=[
                ToolParameter(
                    name="snapshot_id",
                    type="string",
                    description="Snapshot ID to restore"
                )
            ]
        )
    
    async def execute(self, snapshot_id: str) -> ToolResult:
        try:
            await self.vm.restore_snapshot(snapshot_id)
            return ToolResult(
                success=True,
                output=f"VM restored from snapshot: {snapshot_id}"
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=str(e)
            )


class VMCreateProjectTool(Tool):
    """Create a new project in the VM."""
    
    def __init__(self, vm: VMEnvironment):
        super().__init__()
        self.vm = vm
        self.name = "vm_create_project"
    
    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="vm_create_project",
            description="Create a new project structure in the VM",
            parameters=[
                ToolParameter(
                    name="name",
                    type="string",
                    description="Project name"
                ),
                ToolParameter(
                    name="type",
                    type="string",
                    description="Project type",
                    enum=["python", "javascript", "web", "data-science", "api"],
                    required=False,
                    default="python"
                )
            ]
        )
    
    async def execute(self, name: str, type: str = "python") -> ToolResult:
        project_path = f"projects/{name}"
        
        # Create project structure based on type
        structures = {
            "python": {
                "dirs": ["src", "tests", "docs", "data"],
                "files": {
                    "README.md": f"# {name}\n\nA Python project.",
                    "requirements.txt": "",
                    "setup.py": f"""from setuptools import setup, find_packages

setup(
    name="{name}",
    version="0.1.0",
    packages=find_packages(),
)""",
                    "src/__init__.py": "",
                    "tests/__init__.py": "",
                    ".gitignore": "*.pyc\n__pycache__/\n.env\nvenv/\n"
                }
            },
            "javascript": {
                "dirs": ["src", "tests", "public", "dist"],
                "files": {
                    "README.md": f"# {name}\n\nA JavaScript project.",
                    "package.json": json.dumps({
                        "name": name,
                        "version": "1.0.0",
                        "description": "",
                        "main": "src/index.js",
                        "scripts": {
                            "test": "jest",
                            "start": "node src/index.js"
                        }
                    }, indent=2),
                    "src/index.js": "// Main entry point\n",
                    ".gitignore": "node_modules/\ndist/\n.env\n"
                }
            },
            "web": {
                "dirs": ["css", "js", "images", "fonts"],
                "files": {
                    "index.html": f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{name}</title>
    <link rel="stylesheet" href="css/style.css">
</head>
<body>
    <h1>Welcome to {name}</h1>
    <script src="js/main.js"></script>
</body>
</html>""",
                    "css/style.css": "/* Styles */\nbody { font-family: Arial, sans-serif; }\n",
                    "js/main.js": "// JavaScript\nconsole.log('Hello World');\n"
                }
            }
        }
        
        structure = structures.get(type, structures["python"])
        
        # Create directories
        for dir_name in structure["dirs"]:
            await self.vm.execute_command(f"mkdir -p {project_path}/{dir_name}")
        
        # Create files
        for file_path, content in structure["files"].items():
            await self.vm.write_file(f"{project_path}/{file_path}", content)
        
        return ToolResult(
            success=True,
            output=f"Project '{name}' created at {project_path}",
            metadata={
                "project_name": name,
                "project_type": type,
                "project_path": project_path
            }
        )


def register_vm_tools(registry, vm: VMEnvironment):
    """Register all VM tools with the registry.
    
    Args:
        registry: Tool registry
        vm: VM environment instance
    """
    tools = [
        VMExecuteCodeTool(vm),
        VMRunCommandTool(vm),
        VMReadFileTool(vm),
        VMWriteFileTool(vm),
        VMListFilesTool(vm),
        VMInstallPackageTool(vm),
        VMSystemInfoTool(vm),
        VMSnapshotTool(vm),
        VMRestoreSnapshotTool(vm),
        VMCreateProjectTool(vm)
    ]
    
    for tool in tools:
        registry.register(tool)
