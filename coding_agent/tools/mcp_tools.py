"""MCP (Model Context Protocol) client integration for the coding agent.

This module provides tools to connect to MCP servers and expose their tools
to the LLM agent, allowing dynamic tool discovery and execution.
"""

import asyncio
import logging
from typing import Any, Optional

from .base import Tool, ToolResult, ToolRegistry

logger = logging.getLogger(__name__)


class MCPToolWrapper(Tool):
    """Wrapper that adapts an MCP tool to the agent's Tool interface.
    
    Attributes:
        mcp_tool: The original MCP tool definition.
        mcp_client: The MCP client used to call the tool.
    """
    
    def __init__(self, mcp_tool: Any, mcp_client: Any) -> None:
        """Initialize the MCP tool wrapper.
        
        Args:
            mcp_tool: MCP tool object with name, description, inputSchema.
            mcp_client: MCP client instance for calling the tool.
        """
        self._mcp_tool = mcp_tool
        self._mcp_client = mcp_client
        self._name = mcp_tool.name
        self._description = mcp_tool.description or f"Call the {mcp_tool.name} tool"
        self._schema = mcp_tool.inputSchema
    
    @property
    def name(self) -> str:
        """Return the tool name."""
        return self._name
    
    @property
    def description(self) -> str:
        """Return the tool description."""
        return self._description
    
    @property
    def schema(self) -> dict:
        """Return the JSON schema for tool parameters."""
        return self._schema
    
    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the MCP tool with given parameters.
        
        Args:
            **kwargs: Tool-specific parameters.
            
        Returns:
            ToolResult with execution outcome.
        """
        try:
            # Use nest_asyncio to run async code in sync context
            import nest_asyncio
            try:
                nest_asyncio.apply()
            except RuntimeError:
                # Already applied, continue
                pass
            
            loop = asyncio.get_event_loop()
            result = loop.run_until_complete(
                self._async_call_tool(kwargs)
            )
            
            return self._parse_result(result)
            
        except asyncio.TimeoutError:
            logger.error(f"MCP tool '{self._name}' timed out")
            return ToolResult(
                success=False,
                error="Tool execution timed out"
            )
        except Exception as e:
            logger.error(f"MCP tool '{self._name}' failed: {e}")
            return ToolResult(
                success=False,
                error=f"MCP tool execution error: {e}"
            )
    
    async def _async_call_tool(self, kwargs: dict) -> Any:
        """Internal async method to call the MCP tool."""
        return await self._mcp_client.call_tool(self._name, kwargs)
    
    def _parse_result(self, result: Any) -> ToolResult:
        """Parse the MCP tool result into a ToolResult."""
        if hasattr(result, 'content') and result.content:
            # Extract text content from CallToolResult
            output_parts = []
            for item in result.content:
                if hasattr(item, 'text'):
                    output_parts.append(str(item.text))
                else:
                    output_parts.append(str(item))
            output = "\n".join(output_parts)
        elif hasattr(result, 'data') and result.data is not None:
            output = str(result.data)
        else:
            output = str(result)
        
        # Check for errors
        is_error = getattr(result, 'is_error', False)
        
        return ToolResult(
            success=not is_error,
            output=output,
            error=None if not is_error else "MCP tool returned an error"
        )
    
    async def async_execute(self, **kwargs: Any) -> ToolResult:
        """Execute the MCP tool asynchronously.
        
        Args:
            **kwargs: Tool-specific parameters.
            
        Returns:
            ToolResult with execution outcome.
        """
        try:
            result = await self._mcp_client.call_tool(self._name, kwargs)
            
            # Parse the result
            if hasattr(result, 'content') and result.content:
                output_parts = []
                for item in result.content:
                    if hasattr(item, 'text'):
                        output_parts.append(str(item.text))
                    else:
                        output_parts.append(str(item))
                output = "\n".join(output_parts)
            elif hasattr(result, 'data') and result.data is not None:
                output = str(result.data)
            else:
                output = str(result)
            
            is_error = getattr(result, 'is_error', False)
            
            return ToolResult(
                success=not is_error,
                output=output,
                error=None if not is_error else "MCP tool returned an error"
            )
            
        except Exception as e:
            logger.error(f"MCP tool '{self._name}' failed: {e}")
            return ToolResult(
                success=False,
                error=f"MCP tool execution error: {e}"
            )


class MCPClientManager:
    """Manages MCP client connections and tool registration.
    
    This class handles connecting to MCP servers, discovering available tools,
    and registering them with the agent's tool registry.
    
    Example:
        ```python
        from fastmcp import FastMCP
        from coding_agent.tools.mcp_tools import MCPClientManager
        
        # Create an MCP server (or connect to existing one)
        mcp_server = FastMCP("MyServer")
        
        @mcp_server.tool()
        def my_custom_tool(param: str) -> str:
            return f"Processed: {param}"
        
        # Manager will connect and register tools
        manager = MCPClientManager()
        async with manager.connect(mcp_server) as tools:
            # tools are now registered in the registry
            for tool in tools:
                print(f"Registered: {tool.name}")
        ```
    """
    
    def __init__(self, registry: Optional[ToolRegistry] = None) -> None:
        """Initialize the MCP client manager.
        
        Args:
            registry: Tool registry to register MCP tools with.
                     Creates new registry if not provided.
        """
        self._registry = registry or ToolRegistry()
        self._clients: dict[str, Any] = {}  # server_name -> client
        self._wrapped_tools: dict[str, MCPToolWrapper] = {}  # tool_name -> wrapper
    
    @property
    def registry(self) -> ToolRegistry:
        """Get the tool registry."""
        return self._registry
    
    async def connect(
        self,
        transport: Any,
        server_name: Optional[str] = None,
        prefix: Optional[str] = None,
    ) -> list[MCPToolWrapper]:
        """Connect to an MCP server and register its tools.
        
        Args:
            transport: MCP transport (FastMCP instance, URL, Path, etc.).
            server_name: Optional name for the server (for logging).
            prefix: Optional prefix to add to tool names to avoid conflicts.
            
        Returns:
            List of wrapped MCP tools that were registered.
            
        Raises:
            ImportError: If fastmcp is not installed.
            Exception: If connection fails.
        """
        try:
            from fastmcp.client import Client
        except ImportError:
            raise ImportError(
                "fastmcp is required for MCP support. "
                "Install it with: pip install fastmcp"
            )
        
        # Determine server name for tracking
        if server_name is None:
            if hasattr(transport, 'name'):
                server_name = transport.name
            else:
                server_name = f"mcp_server_{id(transport)}"
        
        logger.info(f"Connecting to MCP server: {server_name}")
        
        # Create client
        client = Client(transport)
        
        # Connect and initialize
        await client.__aenter__()
        self._clients[server_name] = client
        
        # Discover tools
        mcp_tools = await client.list_tools()
        logger.info(f"Discovered {len(mcp_tools)} tools from {server_name}")
        
        # Wrap and register each tool
        wrapped_tools = []
        for mcp_tool in mcp_tools:
            tool_name = f"{prefix}_{mcp_tool.name}" if prefix else mcp_tool.name
            
            # Create wrapper
            wrapper = MCPToolWrapper(mcp_tool, client)
            
            # Register with registry
            try:
                self._registry.register(wrapper)
                self._wrapped_tools[tool_name] = wrapper
                wrapped_tools.append(wrapper)
                logger.info(f"Registered MCP tool: {tool_name}")
            except ValueError as e:
                logger.warning(f"Could not register tool '{tool_name}': {e}")
        
        return wrapped_tools
    
    async def disconnect(self, server_name: str) -> None:
        """Disconnect from an MCP server and unregister its tools.
        
        Args:
            server_name: Name of the server to disconnect from.
        """
        if server_name not in self._clients:
            logger.warning(f"No connection found for server: {server_name}")
            return
        
        client = self._clients[server_name]
        
        # Unregister tools from this server
        tools_to_remove = [
            name for name, wrapper in self._wrapped_tools.items()
            if wrapper._mcp_client is client
        ]
        
        for tool_name in tools_to_remove:
            try:
                self._registry.unregister(tool_name)
                del self._wrapped_tools[tool_name]
                logger.info(f"Unregistered MCP tool: {tool_name}")
            except KeyError:
                pass
        
        # Close client connection
        try:
            await client.__aexit__(None, None, None)
            logger.info(f"Disconnected from MCP server: {server_name}")
        except Exception as e:
            logger.error(f"Error disconnecting from {server_name}: {e}")
        
        del self._clients[server_name]
    
    async def disconnect_all(self) -> None:
        """Disconnect from all MCP servers and cleanup."""
        server_names = list(self._clients.keys())
        for name in server_names:
            await self.disconnect(name)
    
    def get_registered_tools(self) -> list[str]:
        """Get list of currently registered MCP tool names."""
        return list(self._wrapped_tools.keys())


# Convenience function for async context manager usage
async def connect_mcp(
    transport: Any,
    registry: Optional[ToolRegistry] = None,
    prefix: Optional[str] = None,
) -> MCPClientManager:
    """Connect to an MCP server and register its tools.
    
    This is a convenience function that creates a manager, connects to the
    server, and returns the manager for later cleanup.
    
    Args:
        transport: MCP transport (FastMCP instance, URL, Path, etc.).
        registry: Tool registry to register tools with.
        prefix: Optional prefix for tool names.
        
    Returns:
        MCPClientManager instance (call disconnect_all() when done).
        
    Example:
        ```python
        manager = await connect_mcp(mcp_server, registry=my_registry)
        try:
            # Use tools...
            result = registry.execute("my_tool", param="value")
        finally:
            await manager.disconnect_all()
        ```
    """
    manager = MCPClientManager(registry)
    await manager.connect(transport, prefix=prefix)
    return manager
