"""Virtual Machine environment manager for isolated agent execution."""

import asyncio
import docker
import tarfile
import io
import uuid
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
import aiodocker
import logging

logger = logging.getLogger(__name__)


@dataclass
class VMConfig:
    """Configuration for VM environment."""
    image: str = "python:3.11-slim"
    memory_limit: str = "2g"
    cpu_limit: float = 2.0
    timeout: int = 300
    network_enabled: bool = True
    persist_data: bool = True
    work_dir: str = "/workspace"
    additional_packages: List[str] = None
    environment_vars: Dict[str, str] = None


class VMEnvironment:
    """Manages a containerized VM environment for agent operations."""
    
    def __init__(self, config: Optional[VMConfig] = None):
        """Initialize VM environment.
        
        Args:
            config: VM configuration
        """
        self.config = config or VMConfig()
        self.container = None
        self.container_id = None
        self.docker_client = None
        self.async_docker = None
        self.workspace_volume = None
        self.session_id = str(uuid.uuid4())[:8]
        
    async def initialize(self):
        """Initialize the VM environment."""
        try:
            # Create Docker clients
            self.docker_client = docker.from_env()
            self.async_docker = aiodocker.Docker()
            
            # Create or use custom image
            await self._prepare_image()
            
            # Create persistent volume for workspace
            if self.config.persist_data:
                self.workspace_volume = self._create_volume()
            
            # Start container
            await self._start_container()
            
            # Setup workspace
            await self._setup_workspace()
            
            logger.info(f"VM environment initialized: {self.session_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize VM: {e}")
            raise
    
    async def _prepare_image(self):
        """Prepare Docker image with required tools."""
        dockerfile_content = f"""
FROM {self.config.image}

# Install system packages
RUN apt-get update && apt-get install -y \\
    git \\
    curl \\
    wget \\
    vim \\
    build-essential \\
    software-properties-common \\
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir \\
    numpy \\
    pandas \\
    matplotlib \\
    requests \\
    beautifulsoup4 \\
    selenium \\
    pytest \\
    black \\
    pylint \\
    ipython

# Install Node.js
RUN curl -fsSL https://deb.nodesource.com/setup_lts.x | bash - && \\
    apt-get install -y nodejs

# Install additional languages
RUN apt-get update && apt-get install -y \\
    golang-go \\
    ruby \\
    rust-all \\
    && rm -rf /var/lib/apt/lists/*

# Create workspace
WORKDIR {self.config.work_dir}

# Set up user (non-root for security)
RUN useradd -m -s /bin/bash agent && \\
    chown -R agent:agent {self.config.work_dir}

USER agent

CMD ["/bin/bash"]
"""
        
        # Build custom image
        image_name = f"open-agent-vm:{self.session_id}"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='Dockerfile', delete=False) as f:
            f.write(dockerfile_content)
            dockerfile_path = f.name
        
        try:
            # Build image
            self.docker_client.images.build(
                path=str(Path(dockerfile_path).parent),
                dockerfile=Path(dockerfile_path).name,
                tag=image_name,
                rm=True
            )
            self.config.image = image_name
        finally:
            Path(dockerfile_path).unlink()
    
    def _create_volume(self) -> str:
        """Create persistent volume for workspace."""
        volume_name = f"agent-workspace-{self.session_id}"
        self.docker_client.volumes.create(name=volume_name)
        return volume_name
    
    async def _start_container(self):
        """Start the Docker container."""
        container_config = {
            'image': self.config.image,
            'command': '/bin/bash',
            'tty': True,
            'stdin_open': True,
            'detach': True,
            'working_dir': self.config.work_dir,
            'environment': self.config.environment_vars or {},
            'mem_limit': self.config.memory_limit,
            'nano_cpus': int(self.config.cpu_limit * 1e9),
            'network_disabled': not self.config.network_enabled,
            'labels': {
                'agent-session': self.session_id,
                'agent-type': 'vm-environment'
            }
        }
        
        # Add volume mount if persistent
        if self.workspace_volume:
            container_config['volumes'] = {
                self.workspace_volume: {
                    'bind': self.config.work_dir,
                    'mode': 'rw'
                }
            }
        
        # Create and start container
        self.container = self.docker_client.containers.run(
            **container_config
        )
        self.container_id = self.container.id
        
        # Wait for container to be ready
        await asyncio.sleep(1)
    
    async def _setup_workspace(self):
        """Set up initial workspace in container."""
        # Create basic directory structure
        dirs = [
            "projects",
            "data",
            "scripts",
            "notebooks",
            ".config"
        ]
        
        for dir_name in dirs:
            await self.execute_command(f"mkdir -p {self.config.work_dir}/{dir_name}")
        
        # Create welcome file
        welcome_content = f"""# Agent Workspace
Session ID: {self.session_id}
Working Directory: {self.config.work_dir}

This is your isolated development environment.
All files and code execution happens within this containerized space.
"""
        await self.write_file("README.md", welcome_content)
    
    async def execute_command(self, command: str, 
                            timeout: Optional[int] = None) -> Tuple[int, str, str]:
        """Execute command in the VM.
        
        Args:
            command: Command to execute
            timeout: Execution timeout in seconds
        
        Returns:
            Tuple of (exit_code, stdout, stderr)
        """
        timeout = timeout or self.config.timeout
        
        try:
            # Execute command in container
            result = self.container.exec_run(
                command,
                tty=True,
                stdin=True,
                stdout=True,
                stderr=True,
                demux=True,
                workdir=self.config.work_dir
            )
            
            stdout = result.output[0].decode('utf-8') if result.output[0] else ""
            stderr = result.output[1].decode('utf-8') if result.output[1] else ""
            
            return result.exit_code, stdout, stderr
            
        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            return 1, "", str(e)
    
    async def execute_code(self, code: str, language: str = "python") -> Tuple[int, str, str]:
        """Execute code in the VM.
        
        Args:
            code: Code to execute
            language: Programming language
        
        Returns:
            Tuple of (exit_code, stdout, stderr)
        """
        # Map language to interpreter
        interpreters = {
            "python": "python3",
            "javascript": "node",
            "ruby": "ruby",
            "go": "go run",
            "rust": "rustc && ./",
            "bash": "bash",
            "sh": "sh"
        }
        
        if language not in interpreters:
            return 1, "", f"Unsupported language: {language}"
        
        # Create temporary file with code
        file_ext = {
            "python": "py",
            "javascript": "js",
            "ruby": "rb",
            "go": "go",
            "rust": "rs",
            "bash": "sh",
            "sh": "sh"
        }.get(language, "txt")
        
        temp_file = f"/tmp/code_{uuid.uuid4().hex[:8]}.{file_ext}"
        
        # Write code to file
        await self.write_file(temp_file, code)
        
        # Execute code
        interpreter = interpreters[language]
        if language == "rust":
            # Special handling for Rust
            binary = temp_file.replace(".rs", "")
            await self.execute_command(f"rustc {temp_file} -o {binary}")
            return await self.execute_command(binary)
        else:
            return await self.execute_command(f"{interpreter} {temp_file}")
    
    async def write_file(self, path: str, content: str):
        """Write file to the VM.
        
        Args:
            path: File path (relative to work_dir)
            content: File content
        """
        if not path.startswith('/'):
            path = f"{self.config.work_dir}/{path}"
        
        # Create tar archive with file
        tar_stream = io.BytesIO()
        tar = tarfile.open(fileobj=tar_stream, mode='w')
        
        # Add file to tar
        file_data = content.encode('utf-8')
        tarinfo = tarfile.TarInfo(name=Path(path).name)
        tarinfo.size = len(file_data)
        tar.addfile(tarinfo, io.BytesIO(file_data))
        tar.close()
        
        # Put file in container
        tar_stream.seek(0)
        self.container.put_archive(
            str(Path(path).parent),
            tar_stream.read()
        )
    
    async def read_file(self, path: str) -> Optional[str]:
        """Read file from the VM.
        
        Args:
            path: File path (relative to work_dir)
        
        Returns:
            File content or None if error
        """
        if not path.startswith('/'):
            path = f"{self.config.work_dir}/{path}"
        
        try:
            # Get file from container
            bits, stat = self.container.get_archive(path)
            
            # Extract from tar
            tar_stream = io.BytesIO()
            for chunk in bits:
                tar_stream.write(chunk)
            tar_stream.seek(0)
            
            tar = tarfile.open(fileobj=tar_stream)
            for member in tar.getmembers():
                f = tar.extractfile(member)
                if f:
                    return f.read().decode('utf-8')
            
        except Exception as e:
            logger.error(f"Failed to read file {path}: {e}")
            return None
    
    async def list_files(self, path: str = "") -> List[Dict[str, Any]]:
        """List files in directory.
        
        Args:
            path: Directory path (relative to work_dir)
        
        Returns:
            List of file information
        """
        if not path:
            path = self.config.work_dir
        elif not path.startswith('/'):
            path = f"{self.config.work_dir}/{path}"
        
        # List directory contents
        exit_code, stdout, _ = await self.execute_command(
            f"ls -la --time-style=long-iso {path}"
        )
        
        if exit_code != 0:
            return []
        
        files = []
        for line in stdout.strip().split('\n')[1:]:  # Skip total line
            parts = line.split(None, 8)
            if len(parts) >= 9:
                files.append({
                    'permissions': parts[0],
                    'size': int(parts[4]),
                    'date': f"{parts[5]} {parts[6]}",
                    'name': parts[8],
                    'type': 'directory' if parts[0].startswith('d') else 'file'
                })
        
        return files
    
    async def install_package(self, package: str, manager: str = "pip"):
        """Install a package in the VM.
        
        Args:
            package: Package name
            manager: Package manager (pip, npm, gem, go, cargo)
        """
        commands = {
            "pip": f"pip install {package}",
            "npm": f"npm install {package}",
            "gem": f"gem install {package}",
            "go": f"go get {package}",
            "cargo": f"cargo install {package}",
            "apt": f"apt-get update && apt-get install -y {package}"
        }
        
        if manager not in commands:
            raise ValueError(f"Unknown package manager: {manager}")
        
        return await self.execute_command(commands[manager])
    
    async def get_system_info(self) -> Dict[str, Any]:
        """Get VM system information."""
        info = {}
        
        # OS info
        exit_code, stdout, _ = await self.execute_command("uname -a")
        info['os'] = stdout.strip() if exit_code == 0 else "Unknown"
        
        # Python version
        exit_code, stdout, _ = await self.execute_command("python3 --version")
        info['python'] = stdout.strip() if exit_code == 0 else "Not installed"
        
        # Node version
        exit_code, stdout, _ = await self.execute_command("node --version")
        info['node'] = stdout.strip() if exit_code == 0 else "Not installed"
        
        # Disk usage
        exit_code, stdout, _ = await self.execute_command(f"df -h {self.config.work_dir}")
        info['disk'] = stdout.strip() if exit_code == 0 else "Unknown"
        
        # Memory info
        exit_code, stdout, _ = await self.execute_command("free -h")
        info['memory'] = stdout.strip() if exit_code == 0 else "Unknown"
        
        return info
    
    async def create_snapshot(self) -> str:
        """Create a snapshot of the current VM state.
        
        Returns:
            Snapshot ID
        """
        snapshot_id = f"snapshot-{uuid.uuid4().hex[:8]}"
        
        # Commit container to image
        self.container.commit(
            repository="agent-snapshot",
            tag=snapshot_id
        )
        
        return snapshot_id
    
    async def restore_snapshot(self, snapshot_id: str):
        """Restore VM from snapshot.
        
        Args:
            snapshot_id: Snapshot ID to restore
        """
        # Stop current container
        await self.cleanup()
        
        # Start new container from snapshot
        self.config.image = f"agent-snapshot:{snapshot_id}"
        await self._start_container()
    
    async def cleanup(self):
        """Clean up VM resources."""
        try:
            if self.container:
                self.container.stop(timeout=5)
                self.container.remove()
                self.container = None
            
            if self.async_docker:
                await self.async_docker.close()
            
            logger.info(f"VM environment cleaned up: {self.session_id}")
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()
