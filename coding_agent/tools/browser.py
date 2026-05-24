"""Browser automation tools for the coding agent.

This module provides browser automation capabilities using Playwright.
Playwright is chosen for its modern API, cross-browser support, and reliability.
"""

import asyncio
from typing import Any

from playwright.async_api import Browser, BrowserContext, Page, Playwright, async_playwright

from .base import Tool, ToolResult


class BrowserSession:
    """Manages a browser session with context and page."""

    def __init__(self) -> None:
        """Initialize browser session."""
        self.playwright: Playwright | None = None
        self.browser: Browser | None = None
        self.context: BrowserContext | None = None
        self.page: Page | None = None
        self._initialized = False

    async def initialize(self, headless: bool = True) -> None:
        """Initialize the browser.

        Args:
            headless: Whether to run in headless mode.
        """
        if self._initialized:
            return

        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=headless)
        self.context = await self.browser.new_context(
            viewport={"width": 1280, "height": 720}
        )
        self.page = await self.context.new_page()
        self._initialized = True

    async def close(self) -> None:
        """Close the browser session."""
        if self.page:
            await self.page.close()
            self.page = None
        if self.context:
            await self.context.close()
            self.context = None
        if self.browser:
            await self.browser.close()
            self.browser = None
        if self.playwright:
            await self.playwright.stop()
            self.playwright = None
        self._initialized = False


# Global browser session manager
_browser_session: BrowserSession | None = None


def _get_session() -> BrowserSession:
    """Get or create the global browser session."""
    global _browser_session
    if _browser_session is None:
        _browser_session = BrowserSession()
    return _browser_session


async def _run_browser_task(coro):
    """Run an async browser task in a sync context."""
    session = _get_session()
    if not session._initialized:
        await session.initialize()
    return await coro


class BrowserTools:
    """Collection of browser automation tools.

    Provides web automation capabilities using Playwright.
    """

    def __init__(self) -> None:
        """Initialize browser tools."""
        pass


class BrowserNavigateTool(Tool):
    """Tool for navigating to a URL."""

    def __init__(self) -> None:
        """Initialize the navigate tool."""
        pass

    @property
    def name(self) -> str:
        """Return tool name."""
        return "browser_navigate"

    @property
    def description(self) -> str:
        """Return tool description."""
        return (
            "Navigate to a URL in the browser. Opens the specified webpage "
            "and waits for it to load completely."
        )

    @property
    def schema(self) -> dict:
        """Return tool parameter schema."""
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "URL to navigate to (include http:// or https://)",
                },
                "wait_until": {
                    "type": "string",
                    "description": "When to consider navigation complete: 'load', 'domcontentloaded', 'networkidle', 'commit' (default: 'load')",
                    "enum": ["load", "domcontentloaded", "networkidle", "commit"],
                },
            },
            "required": ["url"],
        }

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the navigate operation.

        Args:
            **kwargs: Must contain 'url' key, optionally 'wait_until'.

        Returns:
            ToolResult with navigation status or error.
        """
        try:
            url = kwargs.get("url")
            wait_until = kwargs.get("wait_until", "load")

            if not url:
                return ToolResult(success=False, error="Missing required parameter: url")

            async def navigate():
                session = _get_session()
                await session.initialize()
                response = await session.page.goto(url, wait_until=wait_until)
                title = await session.page.title()
                return f"Navigated to {url}\nPage title: {title}\nStatus: {response.status}"

            result = asyncio.run(navigate())
            return ToolResult(success=True, output=result)

        except Exception as e:
            return ToolResult(success=False, error=str(e))


class BrowserClickTool(Tool):
    """Tool for clicking elements on a page."""

    def __init__(self) -> None:
        """Initialize the click tool."""
        pass

    @property
    def name(self) -> str:
        """Return tool name."""
        return "browser_click"

    @property
    def description(self) -> str:
        """Return tool description."""
        return (
            "Click an element on the page identified by a CSS selector. "
            "Use for interacting with buttons, links, and other clickable elements."
        )

    @property
    def schema(self) -> dict:
        """Return tool parameter schema."""
        return {
            "type": "object",
            "properties": {
                "selector": {
                    "type": "string",
                    "description": "CSS selector for the element to click (e.g., 'button.submit', '#login-btn')",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in milliseconds to wait for element (default: 30000)",
                },
            },
            "required": ["selector"],
        }

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the click operation.

        Args:
            **kwargs: Must contain 'selector' key, optionally 'timeout'.

        Returns:
            ToolResult with click status or error.
        """
        try:
            selector = kwargs.get("selector")
            timeout = kwargs.get("timeout", 30000)

            if not selector:
                return ToolResult(success=False, error="Missing required parameter: selector")

            async def click():
                session = _get_session()
                await session.initialize()
                await session.page.click(selector, timeout=timeout)
                return f"Successfully clicked element: {selector}"

            result = asyncio.run(click())
            return ToolResult(success=True, output=result)

        except Exception as e:
            return ToolResult(success=False, error=str(e))


class BrowserFillTool(Tool):
    """Tool for filling input fields."""

    def __init__(self) -> None:
        """Initialize the fill tool."""
        pass

    @property
    def name(self) -> str:
        """Return tool name."""
        return "browser_fill"

    @property
    def description(self) -> str:
        """Return tool description."""
        return (
            "Fill an input field with text. Use for entering data into forms, "
            "search boxes, and other text inputs."
        )

    @property
    def schema(self) -> dict:
        """Return tool parameter schema."""
        return {
            "type": "object",
            "properties": {
                "selector": {
                    "type": "string",
                    "description": "CSS selector for the input field",
                },
                "value": {
                    "type": "string",
                    "description": "Text value to enter",
                },
            },
            "required": ["selector", "value"],
        }

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the fill operation.

        Args:
            **kwargs: Must contain 'selector' and 'value' keys.

        Returns:
            ToolResult with fill status or error.
        """
        try:
            selector = kwargs.get("selector")
            value = kwargs.get("value")

            if not selector:
                return ToolResult(success=False, error="Missing required parameter: selector")
            if value is None:
                return ToolResult(success=False, error="Missing required parameter: value")

            async def fill():
                session = _get_session()
                await session.initialize()
                await session.page.fill(selector, value)
                return f"Successfully filled '{selector}' with value"

            result = asyncio.run(fill())
            return ToolResult(success=True, output=result)

        except Exception as e:
            return ToolResult(success=False, error=str(e))


class BrowserGetContentTool(Tool):
    """Tool for getting page content."""

    def __init__(self) -> None:
        """Initialize the get content tool."""
        pass

    @property
    def name(self) -> str:
        """Return tool name."""
        return "browser_get_content"

    @property
    def description(self) -> str:
        """Return tool description."""
        return (
            "Get the text content of the current page or a specific element. "
            "Use for extracting information from webpages."
        )

    @property
    def schema(self) -> dict:
        """Return tool parameter schema."""
        return {
            "type": "object",
            "properties": {
                "selector": {
                    "type": "string",
                    "description": "CSS selector for specific element (optional, returns full page text if not provided)",
                },
            },
            "required": [],
        }

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the get content operation.

        Args:
            **kwargs: Optionally contains 'selector' key.

        Returns:
            ToolResult with page/element content or error.
        """
        try:
            selector = kwargs.get("selector")

            async def get_content():
                session = _get_session()
                await session.initialize()
                if selector:
                    content = await session.page.inner_text(selector)
                    return f"Content of '{selector}':\n{content}"
                else:
                    content = await session.page.inner_text("body")
                    title = await session.page.title()
                    return f"Page title: {title}\n\nPage content:\n{content[:5000]}"

            result = asyncio.run(get_content())
            return ToolResult(success=True, output=result)

        except Exception as e:
            return ToolResult(success=False, error=str(e))


class BrowserScreenshotTool(Tool):
    """Tool for taking screenshots."""

    def __init__(self) -> None:
        """Initialize the screenshot tool."""
        pass

    @property
    def name(self) -> str:
        """Return tool name."""
        return "browser_screenshot"

    @property
    def description(self) -> str:
        """Return tool description."""
        return (
            "Take a screenshot of the current page or a specific element. "
            "Saves to the specified path or a default location."
        )

    @property
    def schema(self) -> dict:
        """Return tool parameter schema."""
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path to save screenshot (default: screenshot.png)",
                },
                "selector": {
                    "type": "string",
                    "description": "CSS selector for specific element (optional, captures full page if not provided)",
                },
                "full_page": {
                    "type": "boolean",
                    "description": "Capture full scrollable page (default: true)",
                },
            },
            "required": [],
        }

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the screenshot operation.

        Args:
            **kwargs: Optionally contains 'path', 'selector', 'full_page'.

        Returns:
            ToolResult with screenshot path or error.
        """
        try:
            path = kwargs.get("path", "screenshot.png")
            selector = kwargs.get("selector")
            full_page = kwargs.get("full_page", True)

            async def take_screenshot():
                session = _get_session()
                await session.initialize()
                if selector:
                    await session.page.locator(selector).screenshot(path=path)
                else:
                    await session.page.screenshot(path=path, full_page=full_page)
                return f"Screenshot saved to: {path}"

            result = asyncio.run(take_screenshot())
            return ToolResult(success=True, output=result)

        except Exception as e:
            return ToolResult(success=False, error=str(e))


class BrowserEvaluateTool(Tool):
    """Tool for executing JavaScript in the browser."""

    def __init__(self) -> None:
        """Initialize the evaluate tool."""
        pass

    @property
    def name(self) -> str:
        """Return tool name."""
        return "browser_evaluate"

    @property
    def description(self) -> str:
        """Return tool description."""
        return (
            "Execute JavaScript code in the browser context. "
            "Use for custom interactions, data extraction, or complex operations."
        )

    @property
    def schema(self) -> dict:
        """Return tool parameter schema."""
        return {
            "type": "object",
            "properties": {
                "script": {
                    "type": "string",
                    "description": "JavaScript code to execute",
                },
            },
            "required": ["script"],
        }

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the evaluate operation.

        Args:
            **kwargs: Must contain 'script' key.

        Returns:
            ToolResult with JavaScript execution result or error.
        """
        try:
            script = kwargs.get("script")

            if not script:
                return ToolResult(success=False, error="Missing required parameter: script")

            async def evaluate():
                session = _get_session()
                await session.initialize()
                result = await session.page.evaluate(script)
                return f"JavaScript result: {result}"

            result = asyncio.run(evaluate())
            return ToolResult(success=True, output=result)

        except Exception as e:
            return ToolResult(success=False, error=str(e))


class BrowserWaitTool(Tool):
    """Tool for waiting for conditions."""

    def __init__(self) -> None:
        """Initialize the wait tool."""
        pass

    @property
    def name(self) -> str:
        """Return tool name."""
        return "browser_wait"

    @property
    def description(self) -> str:
        """Return tool description."""
        return (
            "Wait for a condition before proceeding. Supports waiting for selectors, "
            "time delays, or network idle states."
        )

    @property
    def schema(self) -> dict:
        """Return tool parameter schema."""
        return {
            "type": "object",
            "properties": {
                "type": {
                    "type": "string",
                    "description": "Type of wait: 'selector', 'time', 'load'",
                    "enum": ["selector", "time", "load"],
                },
                "value": {
                    "type": "string",
                    "description": "Value for the wait (selector string, milliseconds for time, or load state)",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in milliseconds (default: 30000)",
                },
            },
            "required": ["type", "value"],
        }

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the wait operation.

        Args:
            **kwargs: Must contain 'type' and 'value' keys, optionally 'timeout'.

        Returns:
            ToolResult with wait status or error.
        """
        try:
            wait_type = kwargs.get("type")
            value = kwargs.get("value")
            timeout = kwargs.get("timeout", 30000)

            if not wait_type:
                return ToolResult(success=False, error="Missing required parameter: type")
            if value is None:
                return ToolResult(success=False, error="Missing required parameter: value")

            async def wait():
                session = _get_session()
                await session.initialize()

                if wait_type == "selector":
                    await session.page.wait_for_selector(value, timeout=timeout)
                    return f"Element appeared: {value}"
                elif wait_type == "time":
                    await session.page.wait_for_timeout(float(value))
                    return f"Waited for {value}ms"
                elif wait_type == "load":
                    await session.page.wait_for_load_state(value, timeout=timeout)
                    return f"Load state reached: {value}"
                else:
                    raise ValueError(f"Unknown wait type: {wait_type}")

            result = asyncio.run(wait())
            return ToolResult(success=True, output=result)

        except Exception as e:
            return ToolResult(success=False, error=str(e))


class BrowserCloseTool(Tool):
    """Tool for closing the browser."""

    def __init__(self) -> None:
        """Initialize the close tool."""
        pass

    @property
    def name(self) -> str:
        """Return tool name."""
        return "browser_close"

    @property
    def description(self) -> str:
        """Return tool description."""
        return "Close the browser session and release resources."

    @property
    def schema(self) -> dict:
        """Return tool parameter schema."""
        return {
            "type": "object",
            "properties": {},
            "required": [],
        }

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the close operation.

        Args:
            **kwargs: No parameters required.

        Returns:
            ToolResult with close status or error.
        """
        try:
            async def close():
                session = _get_session()
                if session._initialized:
                    await session.close()
                    return "Browser closed successfully"
                return "Browser was not initialized"

            result = asyncio.run(close())
            return ToolResult(success=True, output=result)

        except Exception as e:
            return ToolResult(success=False, error=str(e))
