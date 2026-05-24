"""Browser automation tools using Playwright for the coding agent."""

import asyncio
from typing import Any, Optional
from pathlib import Path

from .base import Tool, ToolResult


class BrowserTools:
    """Collection of browser automation tools using Playwright.
    
    Provides capabilities for web scraping, testing, and browser automation.
    Requires playwright to be installed: pip install playwright
    Then run: playwright install chromium
    """

    def __init__(self, workspace_root: str = ".", headless: bool = True) -> None:
        """Initialize browser tools.
        
        Args:
            workspace_root: Root directory for browser operations.
            headless: Whether to run browser in headless mode.
        """
        self.workspace_root = Path(workspace_root).resolve()
        self.headless = headless
        self._browser = None
        self._context = None
        self._page = None
        self._playwright = None

    async def _get_page(self):
        """Get or create a browser page."""
        if self._page is None:
            await self._ensure_browser()
        return self._page

    async def _ensure_browser(self):
        """Ensure browser context and page are available."""
        from playwright.async_api import async_playwright
        
        if self._browser is None:
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(headless=self.headless)
        
        if self._context is None:
            self._context = await self._browser.new_context(
                viewport={"width": 1280, "height": 720}
            )
            self._page = await self._context.new_page()

    async def close(self):
        """Close browser resources."""
        if self._context:
            await self._context.close()
            self._context = None
        if self._browser:
            await self._browser.close()
            self._browser = None
        if self._playwright:
            await self._playwright.stop()
            self._playwright = None
        self._page = None


class BrowserNavigateTool(Tool):
    """Tool for navigating to a URL."""

    def __init__(self, workspace_root: str = ".", headless: bool = True) -> None:
        """Initialize the browser navigate tool.
        
        Args:
            workspace_root: Working directory.
            headless: Whether to run browser in headless mode.
        """
        self._browser_tools = BrowserTools(workspace_root, headless)

    @property
    def name(self) -> str:
        """Return tool name."""
        return "browser_navigate"

    @property
    def description(self) -> str:
        """Return tool description."""
        return (
            "Navigate to a URL in the browser. Opens the specified URL and waits for it to load. "
            "Use this to start browser automation sessions or visit web pages."
        )

    @property
    def schema(self) -> dict:
        """Return tool parameter schema."""
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "URL to navigate to (must include http:// or https://)",
                },
                "wait_until": {
                    "type": "string",
                    "description": "When to consider navigation successful: 'load', 'domcontentloaded', 'networkidle', 'commit'",
                    "enum": ["load", "domcontentloaded", "networkidle", "commit"],
                },
                "timeout": {
                    "type": "integer",
                    "description": "Navigation timeout in milliseconds (default: 30000)",
                },
            },
            "required": ["url"],
        }

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the browser navigate operation.
        
        Args:
            **kwargs: Must contain 'url' key, optionally 'wait_until' and 'timeout'.
            
        Returns:
            ToolResult with navigation status or error.
        """
        try:
            url = kwargs.get("url")
            wait_until = kwargs.get("wait_until", "load")
            timeout = kwargs.get("timeout", 30000)

            if not url:
                return ToolResult(success=False, error="Missing required parameter: url")

            # Validate URL format
            if not url.startswith(("http://", "https://")):
                return ToolResult(success=False, error="URL must start with http:// or https://")

            result = asyncio.run(self._navigate(url, wait_until, timeout))
            return result

        except Exception as e:
            return ToolResult(success=False, error=str(e))

    async def _navigate(self, url: str, wait_until: str, timeout: int) -> ToolResult:
        """Async navigate to URL."""
        try:
            await self._browser_tools._ensure_browser()
            page = await self._browser_tools._get_page()
            
            response = await page.goto(url, wait_until=wait_until, timeout=timeout)
            
            title = await page.title()
            current_url = page.url
            
            output = (
                f"Successfully navigated to: {current_url}\n"
                f"Page title: {title}\n"
                f"Status: {response.status} {response.status_text}"
            )
            
            return ToolResult(success=True, output=output)
            
        except Exception as e:
            return ToolResult(success=False, error=f"Navigation failed: {str(e)}")


class BrowserClickTool(Tool):
    """Tool for clicking elements on a page."""

    def __init__(self, workspace_root: str = ".", headless: bool = True) -> None:
        """Initialize the browser click tool.
        
        Args:
            workspace_root: Working directory.
            headless: Whether to run browser in headless mode.
        """
        self._browser_tools = BrowserTools(workspace_root, headless)

    @property
    def name(self) -> str:
        """Return tool name."""
        return "browser_click"

    @property
    def description(self) -> str:
        """Return tool description."""
        return (
            "Click an element on the page. Uses CSS selector or text content to find the element. "
            "Use this to interact with buttons, links, and other clickable elements."
        )

    @property
    def schema(self) -> dict:
        """Return tool parameter schema."""
        return {
            "type": "object",
            "properties": {
                "selector": {
                    "type": "string",
                    "description": "CSS selector or text to click (e.g., 'button#submit', 'text=Click Me')",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in milliseconds to wait for element (default: 5000)",
                },
            },
            "required": ["selector"],
        }

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the browser click operation.
        
        Args:
            **kwargs: Must contain 'selector' key, optionally 'timeout'.
            
        Returns:
            ToolResult with click status or error.
        """
        try:
            selector = kwargs.get("selector")
            timeout = kwargs.get("timeout", 5000)

            if not selector:
                return ToolResult(success=False, error="Missing required parameter: selector")

            result = asyncio.run(self._click(selector, timeout))
            return result

        except Exception as e:
            return ToolResult(success=False, error=str(e))

    async def _click(self, selector: str, timeout: int) -> ToolResult:
        """Async click on element."""
        try:
            await self._browser_tools._ensure_browser()
            page = await self._browser_tools._get_page()
            
            # Try different selector strategies
            try:
                # Try as CSS selector first
                await page.click(selector, timeout=timeout)
            except Exception:
                # Try as text content
                await page.get_by_text(selector).first.click(timeout=timeout)
            
            output = f"Successfully clicked: {selector}"
            return ToolResult(success=True, output=output)
            
        except Exception as e:
            return ToolResult(success=False, error=f"Click failed: {str(e)}")


class BrowserFillTool(Tool):
    """Tool for filling input fields."""

    def __init__(self, workspace_root: str = ".", headless: bool = True) -> None:
        """Initialize the browser fill tool.
        
        Args:
            workspace_root: Working directory.
            headless: Whether to run browser in headless mode.
        """
        self._browser_tools = BrowserTools(workspace_root, headless)

    @property
    def name(self) -> str:
        """Return tool name."""
        return "browser_fill"

    @property
    def description(self) -> str:
        """Return tool description."""
        return (
            "Fill an input field with text. Uses CSS selector to find the input element. "
            "Use this to enter text into forms, search boxes, and other input fields."
        )

    @property
    def schema(self) -> dict:
        """Return tool parameter schema."""
        return {
            "type": "object",
            "properties": {
                "selector": {
                    "type": "string",
                    "description": "CSS selector for the input field (e.g., 'input#email', 'input[name=\"username\"]')",
                },
                "value": {
                    "type": "string",
                    "description": "Text value to fill in the input",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in milliseconds to wait for element (default: 5000)",
                },
            },
            "required": ["selector", "value"],
        }

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the browser fill operation.
        
        Args:
            **kwargs: Must contain 'selector' and 'value' keys, optionally 'timeout'.
            
        Returns:
            ToolResult with fill status or error.
        """
        try:
            selector = kwargs.get("selector")
            value = kwargs.get("value")
            timeout = kwargs.get("timeout", 5000)

            if not selector:
                return ToolResult(success=False, error="Missing required parameter: selector")
            if value is None:
                return ToolResult(success=False, error="Missing required parameter: value")

            result = asyncio.run(self._fill(selector, value, timeout))
            return result

        except Exception as e:
            return ToolResult(success=False, error=str(e))

    async def _fill(self, selector: str, value: str, timeout: int) -> ToolResult:
        """Async fill input field."""
        try:
            await self._browser_tools._ensure_browser()
            page = await self._browser_tools._get_page()
            
            await page.fill(selector, value, timeout=timeout)
            
            output = f"Successfully filled '{selector}' with value: {value}"
            return ToolResult(success=True, output=output)
            
        except Exception as e:
            return ToolResult(success=False, error=f"Fill failed: {str(e)}")


class BrowserScreenshotTool(Tool):
    """Tool for taking screenshots of the page."""

    def __init__(self, workspace_root: str = ".", headless: bool = True) -> None:
        """Initialize the browser screenshot tool.
        
        Args:
            workspace_root: Working directory.
            headless: Whether to run browser in headless mode.
        """
        self._browser_tools = BrowserTools(workspace_root, headless)

    @property
    def name(self) -> str:
        """Return tool name."""
        return "browser_screenshot"

    @property
    def description(self) -> str:
        """Return tool description."""
        return (
            "Take a screenshot of the current page. Saves to a file in the workspace directory. "
            "Use this to capture the visual state of the page for debugging or documentation."
        )

    @property
    def schema(self) -> dict:
        """Return tool parameter schema."""
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path to save the screenshot (default: screenshot.png)",
                },
                "full_page": {
                    "type": "boolean",
                    "description": "If true, capture the entire scrollable page (default: false)",
                },
            },
        }

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the browser screenshot operation.
        
        Args:
            **kwargs: Optionally 'path' and 'full_page'.
            
        Returns:
            ToolResult with screenshot path or error.
        """
        try:
            path = kwargs.get("path", "screenshot.png")
            full_page = kwargs.get("full_page", False)

            result = asyncio.run(self._screenshot(path, full_page))
            return result

        except Exception as e:
            return ToolResult(success=False, error=str(e))

    async def _screenshot(self, path: str, full_page: bool) -> ToolResult:
        """Async take screenshot."""
        try:
            await self._browser_tools._ensure_browser()
            page = await self._browser_tools._get_page()
            
            # Ensure path is within workspace
            screenshot_path = self._browser_tools.workspace_root / path
            screenshot_path.parent.mkdir(parents=True, exist_ok=True)
            
            await page.screenshot(path=str(screenshot_path), full_page=full_page)
            
            output = f"Screenshot saved to: {screenshot_path}"
            return ToolResult(success=True, output=output)
            
        except Exception as e:
            return ToolResult(success=False, error=f"Screenshot failed: {str(e)}")


class BrowserGetContentTool(Tool):
    """Tool for getting page content."""

    def __init__(self, workspace_root: str = ".", headless: bool = True) -> None:
        """Initialize the browser get content tool.
        
        Args:
            workspace_root: Working directory.
            headless: Whether to run browser in headless mode.
        """
        self._browser_tools = BrowserTools(workspace_root, headless)

    @property
    def name(self) -> str:
        """Return tool name."""
        return "browser_get_content"

    @property
    def description(self) -> str:
        """Return tool description."""
        return (
            "Get the content of the current page. Returns HTML, text, or specific element content. "
            "Use this to extract data from web pages for scraping or analysis."
        )

    @property
    def schema(self) -> dict:
        """Return tool parameter schema."""
        return {
            "type": "object",
            "properties": {
                "selector": {
                    "type": "string",
                    "description": "Optional CSS selector to get content from specific element. If omitted, returns full page content.",
                },
                "content_type": {
                    "type": "string",
                    "description": "Type of content to retrieve: 'html', 'text', 'inner_text'",
                    "enum": ["html", "text", "inner_text"],
                },
            },
        }

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the browser get content operation.
        
        Args:
            **kwargs: Optionally 'selector' and 'content_type'.
            
        Returns:
            ToolResult with page content or error.
        """
        try:
            selector = kwargs.get("selector")
            content_type = kwargs.get("content_type", "text")

            result = asyncio.run(self._get_content(selector, content_type))
            return result

        except Exception as e:
            return ToolResult(success=False, error=str(e))

    async def _get_content(self, selector: Optional[str], content_type: str) -> ToolResult:
        """Async get page content."""
        try:
            await self._browser_tools._ensure_browser()
            page = await self._browser_tools._get_page()
            
            if selector:
                element = page.locator(selector).first
                if content_type == "html":
                    content = await element.inner_html()
                elif content_type == "inner_text":
                    content = await element.inner_text()
                else:
                    content = await element.text_content()
            else:
                if content_type == "html":
                    content = await page.content()
                elif content_type == "inner_text":
                    content = await page.evaluate("document.body.innerText")
                else:
                    content = await page.evaluate("document.body.textContent")
            
            # Truncate if too long
            max_length = 10000
            if len(content) > max_length:
                content = content[:max_length] + "\n... (truncated)"
            
            output = f"Page content ({content_type}):\n{content}"
            return ToolResult(success=True, output=output)
            
        except Exception as e:
            return ToolResult(success=False, error=f"Get content failed: {str(e)}")


class BrowserEvaluateTool(Tool):
    """Tool for executing JavaScript in the browser."""

    def __init__(self, workspace_root: str = ".", headless: bool = True) -> None:
        """Initialize the browser evaluate tool.
        
        Args:
            workspace_root: Working directory.
            headless: Whether to run browser in headless mode.
        """
        self._browser_tools = BrowserTools(workspace_root, headless)

    @property
    def name(self) -> str:
        """Return tool name."""
        return "browser_evaluate"

    @property
    def description(self) -> str:
        """Return tool description."""
        return (
            "Execute JavaScript code in the browser context and return the result. "
            "Use this for advanced interactions, data extraction, or custom automation logic."
        )

    @property
    def schema(self) -> dict:
        """Return tool parameter schema."""
        return {
            "type": "object",
            "properties": {
                "javascript": {
                    "type": "string",
                    "description": "JavaScript code to execute in the browser",
                },
            },
            "required": ["javascript"],
        }

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the browser evaluate operation.
        
        Args:
            **kwargs: Must contain 'javascript' key.
            
        Returns:
            ToolResult with evaluation result or error.
        """
        try:
            javascript = kwargs.get("javascript")

            if not javascript:
                return ToolResult(success=False, error="Missing required parameter: javascript")

            result = asyncio.run(self._evaluate(javascript))
            return result

        except Exception as e:
            return ToolResult(success=False, error=str(e))

    async def _evaluate(self, javascript: str) -> ToolResult:
        """Async evaluate JavaScript."""
        try:
            await self._browser_tools._ensure_browser()
            page = await self._browser_tools._get_page()
            
            result = await page.evaluate(javascript)
            
            output = f"JavaScript execution result:\n{result}"
            return ToolResult(success=True, output=output)
            
        except Exception as e:
            return ToolResult(success=False, error=f"Evaluation failed: {str(e)}")


class BrowserCloseTool(Tool):
    """Tool for closing the browser."""

    def __init__(self, workspace_root: str = ".", headless: bool = True) -> None:
        """Initialize the browser close tool.
        
        Args:
            workspace_root: Working directory.
            headless: Whether to run browser in headless mode.
        """
        self._browser_tools = BrowserTools(workspace_root, headless)

    @property
    def name(self) -> str:
        """Return tool name."""
        return "browser_close"

    @property
    def description(self) -> str:
        """Return tool description."""
        return (
            "Close the browser and release all resources. Use this to clean up after browser automation tasks."
        )

    @property
    def schema(self) -> dict:
        """Return tool parameter schema."""
        return {
            "type": "object",
            "properties": {},
        }

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the browser close operation.
        
        Args:
            **kwargs: No parameters required.
            
        Returns:
            ToolResult with close status or error.
        """
        try:
            result = asyncio.run(self._close())
            return result

        except Exception as e:
            return ToolResult(success=False, error=str(e))

    async def _close(self) -> ToolResult:
        """Async close browser."""
        try:
            await self._browser_tools.close()
            output = "Browser closed successfully"
            return ToolResult(success=True, output=output)
            
        except Exception as e:
            return ToolResult(success=False, error=f"Close failed: {str(e)}")
