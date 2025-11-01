"""Enhanced agent with integrated VM environment."""

import logging
from typing import Optional, Dict, Any

from .agent import Agent, AgentConfig, Message
from .environment.vm_manager import VMEnvironment, VMConfig
from .tools.vm_tools import register_vm_tools
from .providers.base import LLMProvider


logger = logging.getLogger(__name__)


class VMAgent(Agent):
    """Agent with integrated VM environment for isolated execution."""
    
    def __init__(self, 
                 provider: Optional[LLMProvider] = None,
                 config: Optional[AgentConfig] = None,
                 vm_config: Optional[VMConfig] = None,
                 auto_start_vm: bool = True):
        """Initialize VM-enabled agent.
        
        Args:
            provider: LLM provider instance
            config: Agent configuration
            vm_config: VM configuration
            auto_start_vm: Whether to automatically start VM
        """
        super().__init__(provider, config)
        
        self.vm_config = vm_config or VMConfig()
        self.vm: Optional[VMEnvironment] = None
        self.auto_start_vm = auto_start_vm
        self.vm_initialized = False
        
        # VM session tracking
        self.vm_session_id = None
        self.vm_snapshots = []
        
    async def initialize_vm(self):
        """Initialize the VM environment."""
        if self.vm_initialized:
            return
        
        try:
            logger.info("Initializing VM environment...")
            
            # Create VM environment
            self.vm = VMEnvironment(config=self.vm_config)
            await self.vm.initialize()
            
            self.vm_session_id = self.vm.session_id
            
            # Register VM tools
            register_vm_tools(self.tool_registry, self.vm)
            
            # Add system message about VM
            vm_info = await self.vm.get_system_info()
            system_message = f"""You now have access to an isolated VM environment.
Session ID: {self.vm_session_id}
Working directory: {self.vm_config.work_dir}

Available VM tools:
- vm_execute_code: Execute code in any supported language
- vm_run_command: Run shell commands
- vm_read_file/vm_write_file: File operations
- vm_list_files: Browse directories
- vm_install_package: Install packages
- vm_create_project: Create project structures
- vm_snapshot/vm_restore_snapshot: State management

All code execution and file operations happen within this isolated environment.
The VM has Python, Node.js, Go, Ruby, and Rust installed."""
            
            self.conversation.add_message(Message(
                role="system",
                content=system_message,
                metadata={"vm_info": vm_info}
            ))
            
            self.vm_initialized = True
            logger.info(f"VM environment ready: {self.vm_session_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize VM: {e}")
            raise
    
    async def process_message(self, message: str, 
                             system_prompt: Optional[str] = None) -> str:
        """Process message with VM environment.
        
        Args:
            message: User message
            system_prompt: Optional system prompt
            
        Returns:
            Assistant's response
        """
        # Auto-start VM if enabled
        if self.auto_start_vm and not self.vm_initialized:
            await self.initialize_vm()
        
        # Enhanced system prompt for VM context
        if system_prompt:
            system_prompt = f"""{system_prompt}

You have access to a fully-featured VM environment where you can:
- Execute code in multiple languages
- Create and manage files and projects  
- Install packages and dependencies
- Run system commands
- Take snapshots and restore states

Use the vm_* tools to interact with this environment."""
        
        return await super().process_message(message, system_prompt)
    
    async def create_checkpoint(self, name: Optional[str] = None) -> str:
        """Create a VM checkpoint.
        
        Args:
            name: Optional checkpoint name
            
        Returns:
            Checkpoint ID
        """
        if not self.vm:
            raise RuntimeError("VM not initialized")
        
        snapshot_id = await self.vm.create_snapshot()
        
        checkpoint = {
            "id": snapshot_id,
            "name": name or f"checkpoint_{len(self.vm_snapshots)}",
            "conversation_state": len(self.conversation.messages)
        }
        
        self.vm_snapshots.append(checkpoint)
        
        logger.info(f"Created checkpoint: {checkpoint['name']} ({snapshot_id})")
        return snapshot_id
    
    async def restore_checkpoint(self, checkpoint_id: str):
        """Restore from a checkpoint.
        
        Args:
            checkpoint_id: Checkpoint ID to restore
        """
        if not self.vm:
            raise RuntimeError("VM not initialized")
        
        # Find checkpoint
        checkpoint = None
        for cp in self.vm_snapshots:
            if cp["id"] == checkpoint_id or cp["name"] == checkpoint_id:
                checkpoint = cp
                break
        
        if not checkpoint:
            raise ValueError(f"Checkpoint not found: {checkpoint_id}")
        
        # Restore VM state
        await self.vm.restore_snapshot(checkpoint["id"])
        
        # Restore conversation to checkpoint state
        if checkpoint["conversation_state"]:
            self.conversation.messages = self.conversation.messages[:checkpoint["conversation_state"]]
        
        logger.info(f"Restored checkpoint: {checkpoint['name']}")
    
    async def reset_vm(self):
        """Reset the VM to a clean state."""
        if self.vm:
            await self.vm.cleanup()
            self.vm_initialized = False
        
        await self.initialize_vm()
    
    async def get_vm_status(self) -> Dict[str, Any]:
        """Get current VM status.
        
        Returns:
            VM status information
        """
        if not self.vm:
            return {"status": "not_initialized"}
        
        info = await self.vm.get_system_info()
        files = await self.vm.list_files()
        
        return {
            "status": "running",
            "session_id": self.vm_session_id,
            "system_info": info,
            "workspace_files": len(files),
            "snapshots": len(self.vm_snapshots),
            "config": {
                "memory_limit": self.vm_config.memory_limit,
                "cpu_limit": self.vm_config.cpu_limit,
                "network_enabled": self.vm_config.network_enabled,
                "persist_data": self.vm_config.persist_data
            }
        }
    
    async def execute_in_vm(self, code: str, language: str = "python") -> Dict[str, Any]:
        """Direct code execution in VM.
        
        Args:
            code: Code to execute
            language: Programming language
            
        Returns:
            Execution result
        """
        if not self.vm:
            await self.initialize_vm()
        
        exit_code, stdout, stderr = await self.vm.execute_code(code, language)
        
        return {
            "success": exit_code == 0,
            "exit_code": exit_code,
            "stdout": stdout,
            "stderr": stderr
        }
    
    async def install_in_vm(self, packages: list, manager: str = "pip"):
        """Install packages in VM.
        
        Args:
            packages: List of packages to install
            manager: Package manager to use
        """
        if not self.vm:
            await self.initialize_vm()
        
        results = []
        for package in packages:
            exit_code, stdout, stderr = await self.vm.install_package(package, manager)
            results.append({
                "package": package,
                "success": exit_code == 0,
                "output": stdout if exit_code == 0 else stderr
            })
        
        return results
    
    async def cleanup(self):
        """Clean up resources."""
        if self.vm:
            await self.vm.cleanup()
            self.vm = None
            self.vm_initialized = False
        
        logger.info("VM Agent cleaned up")
    
    async def __aenter__(self):
        """Async context manager entry."""
        if self.auto_start_vm:
            await self.initialize_vm()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()


class VMAgentSession:
    """Manages a VM agent session with lifecycle."""
    
    def __init__(self, 
                 provider: LLMProvider,
                 agent_config: Optional[AgentConfig] = None,
                 vm_config: Optional[VMConfig] = None):
        """Initialize VM agent session.
        
        Args:
            provider: LLM provider
            agent_config: Agent configuration
            vm_config: VM configuration
        """
        self.provider = provider
        self.agent_config = agent_config or AgentConfig()
        self.vm_config = vm_config or VMConfig()
        self.agent: Optional[VMAgent] = None
        
    async def start(self):
        """Start the agent session."""
        self.agent = VMAgent(
            provider=self.provider,
            config=self.agent_config,
            vm_config=self.vm_config,
            auto_start_vm=True
        )
        await self.agent.initialize_vm()
        return self.agent
    
    async def restart(self):
        """Restart the agent session."""
        if self.agent:
            await self.agent.cleanup()
        
        return await self.start()
    
    async def stop(self):
        """Stop the agent session."""
        if self.agent:
            await self.agent.cleanup()
            self.agent = None
    
    async def __aenter__(self):
        """Context manager entry."""
        return await self.start()
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        await self.stop()
