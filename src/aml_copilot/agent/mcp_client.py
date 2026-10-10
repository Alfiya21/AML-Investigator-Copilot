import json
import os
import sys
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class ToolBox:
    """Starts the MCP server as a subprocess and exposes its tools in OpenAI's format."""

    def __init__(self):
        self._stack = AsyncExitStack()
        self.session = None
        self.openai_tools = []

    async def __aenter__(self):
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "aml_copilot.mcp_server.server"],
            env=dict(os.environ),
        )
        read, write = await self._stack.enter_async_context(stdio_client(params))
        self.session = await self._stack.enter_async_context(ClientSession(read, write))
        await self.session.initialize()

        listing = await self.session.list_tools()
        self.openai_tools = [
            {"type": "function", "function": {
                "name": t.name,
                "description": t.description or "",
                "parameters": t.inputSchema,
            }}
            for t in listing.tools
        ]
        return self

    async def __aexit__(self, *exc):
        await self._stack.aclose()

    async def call(self, name, args):
        result = await self.session.call_tool(name, args)
        text = "".join(c.text for c in result.content if getattr(c, "type", "") == "text")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"raw": text}
