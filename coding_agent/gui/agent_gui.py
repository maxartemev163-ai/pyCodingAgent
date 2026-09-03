#!/usr/bin/env python3
"""GUI application for the coding agent using PyQt5."""

import sys
import os
import json
import threading
from pathlib import Path
from typing import Optional

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTextEdit,
    QLineEdit,
    QPushButton,
    QLabel,
    QFileDialog,
    QMessageBox,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QStatusBar,
    QMenuBar,
    QMenu,
    QAction,
    QToolBar,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QComboBox,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QFont, QIcon, QTextCursor

from coding_agent.config import ModelConfig, Settings
from coding_agent.core import CodingAgent
from coding_agent.tools import (
    ListDirTool,
    ReadFileTool,
    RunCommandTool,
    RunPythonTool,
    SearchFilesTool,
    WriteFileTool,
    VulnerabilityScannerTool,
)


class AgentWorker(QThread):
    """Worker thread for running agent tasks without blocking the UI."""
    
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    tool_output = pyqtSignal(str)
    
    def __init__(self, agent: CodingAgent, user_input: str):
        super().__init__()
        self.agent = agent
        self.user_input = user_input
    
    def run(self):
        """Run the agent task in a separate thread."""
        try:
            # Capture print output from tool execution
            import io
            from contextlib import redirect_stdout
            
            f = io.StringIO()
            with redirect_stdout(f):
                response = self.agent.run(self.user_input)
            
            tool_output = f.getvalue()
            if tool_output:
                self.tool_output.emit(tool_output)
            
            self.finished.emit(response or "No response generated.")
        except Exception as e:
            self.error.emit(str(e))


class FileExplorer(QWidget):
    """File explorer widget showing workspace directory structure."""
    
    file_selected = pyqtSignal(str)
    
    def __init__(self, workspace_dir: str, parent=None):
        super().__init__(parent)
        self.workspace_dir = Path(workspace_dir)
        self.init_ui()
        self.refresh_tree()
    
    def init_ui(self):
        """Initialize the file explorer UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Header
        header = QLabel("📁 Workspace Files")
        header.setStyleSheet("font-weight: bold; padding: 5px;")
        layout.addWidget(header)
        
        # Tree widget
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Name", "Type"])
        self.tree.setColumnWidth(0, 200)
        self.tree.itemDoubleClicked.connect(self.on_item_double_clicked)
        layout.addWidget(self.tree)
        
        # Refresh button
        refresh_btn = QPushButton("🔄 Refresh")
        refresh_btn.clicked.connect(self.refresh_tree)
        layout.addWidget(refresh_btn)
    
    def refresh_tree(self):
        """Refresh the file tree."""
        self.tree.clear()
        self._populate_tree(self.workspace_dir, self.tree.invisibleRootItem())
    
    def _populate_tree(self, path: Path, parent_item: QTreeWidgetItem, depth: int = 0):
        """Recursively populate the tree with files and directories."""
        if depth > 3:  # Limit depth to avoid performance issues
            return
        
        try:
            items = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
            for item_path in items:
                if item_path.name.startswith('.'):
                    continue
                
                if item_path.is_dir():
                    item = QTreeWidgetItem([item_path.name, "Folder"])
                    item.setIcon(0, self.style().standardIcon(10))  # Dir icon
                    parent_item.addChild(item)
                    
                    # Recursively add subdirectories (limited depth)
                    self._populate_tree(item_path, item, depth + 1)
                else:
                    ext = item_path.suffix.lower()
                    file_type = "File"
                    if ext == '.py':
                        file_type = "Python"
                    elif ext in ['.js', '.ts', '.jsx', '.tsx']:
                        file_type = "JavaScript"
                    elif ext in ['.html', '.css']:
                        file_type = "Web"
                    elif ext in ['.md', '.txt', '.rst']:
                        file_type = "Text"
                    elif ext in ['.json', '.yaml', '.yml', '.toml']:
                        file_type = "Config"
                    
                    item = QTreeWidgetItem([item_path.name, file_type])
                    item.setIcon(0, self.style().standardIcon(6))  # File icon
                    parent_item.addChild(item)
        except PermissionError:
            pass
    
    def on_item_double_clicked(self, item: QTreeWidgetItem, column: int):
        """Handle double-click on a file item."""
        # Get the full path by traversing up the tree
        path_parts = []
        current = item
        while current:
            path_parts.insert(0, current.text(0))
            current = current.parent()
        
        if len(path_parts) > 0:
            file_path = self.workspace_dir / '/'.join(path_parts)
            if file_path.exists() and file_path.is_file():
                self.file_selected.emit(str(file_path))


class ChatMessage(QTextEdit):
    """Custom text edit for displaying chat messages."""
    
    def __init__(self, is_user: bool, parent=None):
        super().__init__(parent)
        self.is_user = is_user
        self.setReadOnly(True)
        self.setStyleSheet(self._get_style())
        self.setFont(QFont("Consolas", 10))
    
    def _get_style(self) -> str:
        """Get stylesheet based on message type."""
        if self.is_user:
            return """
                QTextEdit {
                    background-color: #e3f2fd;
                    border-radius: 10px;
                    padding: 10px;
                    margin: 5px;
                }
            """
        else:
            return """
                QTextEdit {
                    background-color: #f5f5f5;
                    border-radius: 10px;
                    padding: 10px;
                    margin: 5px;
                }
            """


class SettingsDialog(QDialog):
    """Dialog for configuring agent settings."""
    
    def __init__(self, model_config: ModelConfig, workspace_dir: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(400)
        
        self.model_config = model_config
        self.workspace_dir = workspace_dir
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize the settings dialog UI."""
        layout = QFormLayout(self)
        
        # Workspace directory
        self.workspace_edit = QLineEdit(str(self.workspace_dir))
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_workspace)
        
        workspace_layout = QHBoxLayout()
        workspace_layout.addWidget(self.workspace_edit)
        workspace_layout.addWidget(browse_btn)
        layout.addRow("Workspace:", workspace_layout)
        
        # Base URL
        self.base_url_edit = QLineEdit(self.model_config.base_url)
        layout.addRow("Base URL:", self.base_url_edit)
        
        # API Key
        self.api_key_edit = QLineEdit(self.model_config.api_key)
        layout.addRow("API Key:", self.api_key_edit)
        
        # Model name
        self.model_edit = QLineEdit(self.model_config.model_name)
        layout.addRow("Model:", self.model_edit)
        
        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)
    
    def browse_workspace(self):
        """Open directory browser."""
        directory = QFileDialog.getExistingDirectory(
            self, "Select Workspace Directory", str(self.workspace_dir)
        )
        if directory:
            self.workspace_edit.setText(directory)
    
    def get_model_config(self) -> ModelConfig:
        """Get the updated model configuration."""
        return ModelConfig(
            base_url=self.base_url_edit.text(),
            api_key=self.api_key_edit.text(),
            model_name=self.model_edit.text(),
        )
    
    def get_workspace_dir(self) -> str:
        """Get the updated workspace directory."""
        return self.workspace_edit.text()


class CodingAgentGUI(QMainWindow):
    """Main window for the coding agent GUI."""
    
    def __init__(self):
        super().__init__()
        
        # Default configuration
        self.workspace_dir = str(Path.home() / "coding_workspace")
        self.model_config = ModelConfig(
            base_url=os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"),
            api_key=os.getenv("LLM_API_KEY", "ollama"),
            model_name=os.getenv("LLM_MODEL", "qwen2.5-coder:7b"),
        )
        self.agent: Optional[CodingAgent] = None
        self.worker: Optional[AgentWorker] = None
        
        self.init_ui()
        self.init_menu()
        self.init_status_bar()
        
        # Try to create workspace directory if it doesn't exist
        Path(self.workspace_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize agent
        self.initialize_agent()
    
    def init_ui(self):
        """Initialize the main window UI."""
        self.setWindowTitle("Coding Agent - AI Programming Assistant")
        self.setMinimumSize(1200, 800)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel - File explorer
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        self.file_explorer = FileExplorer(self.workspace_dir)
        self.file_explorer.file_selected.connect(self.on_file_selected)
        left_layout.addWidget(self.file_explorer)
        
        splitter.addWidget(left_panel)
        
        # Right panel - Chat interface
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        # Chat display area
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setFont(QFont("Consolas", 10))
        self.chat_display.setStyleSheet("""
            QTextEdit {
                background-color: #ffffff;
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 10px;
            }
        """)
        right_layout.addWidget(self.chat_display)
        
        # Input area
        input_layout = QHBoxLayout()
        
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Type your request here... (Press Enter to send)")
        self.input_field.setFont(QFont("Consolas", 10))
        self.input_field.returnPressed.connect(self.send_message)
        input_layout.addWidget(self.input_field)
        
        send_btn = QPushButton("Send ➤")
        send_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """)
        send_btn.clicked.connect(self.send_message)
        input_layout.addWidget(send_btn)
        
        right_layout.addLayout(input_layout)
        
        # Quick commands toolbar
        commands_layout = QHBoxLayout()
        commands_layout.addWidget(QLabel("Quick commands:"))
        
        self.plan_btn = QPushButton("📋 Plan")
        self.plan_btn.clicked.connect(lambda: self.send_command("/plan"))
        commands_layout.addWidget(self.plan_btn)
        
        self.scan_btn = QPushButton("🔍 Scan")
        self.scan_btn.clicked.connect(lambda: self.send_command("/scan"))
        commands_layout.addWidget(self.scan_btn)
        
        self.clear_btn = QPushButton("🗑️ Clear")
        self.clear_btn.clicked.connect(self.clear_chat)
        commands_layout.addWidget(self.clear_btn)
        
        self.retry_btn = QPushButton("🔄 Retry")
        self.retry_btn.clicked.connect(self.retry_last_task)
        commands_layout.addWidget(self.retry_btn)
        
        commands_layout.addStretch()
        
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: gray;")
        commands_layout.addWidget(self.status_label)
        
        right_layout.addLayout(commands_layout)
        
        splitter.addWidget(right_panel)
        
        # Set initial sizes (25% left, 75% right)
        splitter.setSizes([300, 900])
        
        main_layout.addWidget(splitter)
    
    def init_menu(self):
        """Initialize the menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("&File")
        
        open_workspace_action = QAction("&Open Workspace...", self)
        open_workspace_action.setShortcut("Ctrl+O")
        open_workspace_action.triggered.connect(self.open_workspace)
        file_menu.addAction(open_workspace_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Edit menu
        edit_menu = menubar.addMenu("&Edit")
        
        clear_action = QAction("&Clear Chat", self)
        clear_action.setShortcut("Ctrl+L")
        clear_action.triggered.connect(self.clear_chat)
        edit_menu.addAction(clear_action)
        
        edit_menu.addSeparator()
        
        retry_action = QAction("&Retry Last Task", self)
        retry_action.setShortcut("Ctrl+R")
        retry_action.triggered.connect(self.retry_last_task)
        edit_menu.addAction(retry_action)
        
        # Tools menu
        tools_menu = menubar.addMenu("&Tools")
        
        plan_action = QAction("&Generate Plan", self)
        plan_action.setShortcut("Ctrl+P")
        plan_action.triggered.connect(lambda: self.send_command("/plan"))
        tools_menu.addAction(plan_action)
        
        scan_action = QAction("&Vulnerability Scan", self)
        scan_action.setShortcut("Ctrl+S")
        scan_action.triggered.connect(lambda: self.send_command("/scan"))
        tools_menu.addAction(scan_action)
        
        tools_menu.addSeparator()
        
        show_tools_action = QAction("Show Available &Tools", self)
        show_tools_action.triggered.connect(self.show_tools_list)
        tools_menu.addAction(show_tools_action)
        
        # Settings menu
        settings_menu = menubar.addMenu("&Settings")
        
        settings_action = QAction("&Configure...", self)
        settings_action.setShortcut("Ctrl+,")
        settings_action.triggered.connect(self.show_settings)
        settings_menu.addAction(settings_action)
        
        # Help menu
        help_menu = menubar.addMenu("&Help")
        
        about_action = QAction("&About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
        # Toolbar
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        
        toolbar.addAction(open_workspace_action)
        toolbar.addSeparator()
        toolbar.addAction(clear_action)
        toolbar.addSeparator()
        toolbar.addAction(plan_action)
        toolbar.addSeparator()
        toolbar.addAction(scan_action)
    
    def init_status_bar(self):
        """Initialize the status bar."""
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        self.statusbar.showMessage("Ready")
    
    def initialize_agent(self):
        """Initialize the coding agent with current settings."""
        try:
            settings = Settings(
                workspace_dir=self.workspace_dir,
                max_iterations=50,
                log_level="INFO",
            )
            
            self.agent = CodingAgent(
                settings=settings,
                model_config=self.model_config,
                prepare_context=True,
            )
            
            # Register tools
            self.agent.register_tool(ReadFileTool(workspace_root=self.workspace_dir))
            self.agent.register_tool(WriteFileTool(workspace_root=self.workspace_dir))
            self.agent.register_tool(ListDirTool(workspace_root=self.workspace_dir))
            self.agent.register_tool(SearchFilesTool(workspace_root=self.workspace_dir))
            self.agent.register_tool(RunCommandTool(workspace_root=self.workspace_dir))
            self.agent.register_tool(RunPythonTool(workspace_root=self.workspace_dir))
            self.agent.register_tool(VulnerabilityScannerTool(workspace_root=self.workspace_dir))
            
            self.statusbar.showMessage(f"Agent initialized - Workspace: {self.workspace_dir}")
            self.add_system_message(f"✓ Agent initialized successfully.\nWorkspace: {self.workspace_dir}\nModel: {self.model_config.model_name}")
            
        except Exception as e:
            QMessageBox.critical(self, "Initialization Error", f"Failed to initialize agent:\n{e}")
            self.statusbar.showMessage("Initialization failed")
    
    def add_user_message(self, text: str):
        """Add a user message to the chat display."""
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.End)
        
        # Add user label
        cursor.insertHtml('<div style="background-color: #e3f2fd; border-radius: 10px; padding: 10px; margin: 5px;">')
        cursor.insertHtml('<b>👤 You:</b><br>')
        cursor.insertPlainText(text)
        cursor.insertHtml('</div><br>')
        
        self.chat_display.setTextCursor(cursor)
        self.chat_display.scrollToBottom()
    
    def add_assistant_message(self, text: str):
        """Add an assistant message to the chat display."""
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.End)
        
        # Add assistant label
        cursor.insertHtml('<div style="background-color: #f5f5f5; border-radius: 10px; padding: 10px; margin: 5px;">')
        cursor.insertHtml('<b>🤖 Agent:</b><br>')
        # Convert markdown-like formatting to HTML
        formatted_text = text.replace('\n', '<br>').replace('**', '<b>').replace('`', '<code>')
        cursor.insertHtml(formatted_text)
        cursor.insertHtml('</div><br>')
        
        self.chat_display.setTextCursor(cursor)
        self.chat_display.scrollToBottom()
    
    def add_system_message(self, text: str):
        """Add a system message to the chat display."""
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.End)
        
        cursor.insertHtml(f'<div style="color: gray; font-style: italic; padding: 5px;">{text}</div><br>')
        
        self.chat_display.setTextCursor(cursor)
        self.chat_display.scrollToBottom()
    
    def add_tool_output(self, text: str):
        """Add tool output to the chat display."""
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.End)
        
        cursor.insertHtml('<div style="background-color: #fff3e0; border-left: 3px solid #ff9800; padding: 10px; margin: 5px; font-family: monospace;">')
        cursor.insertHtml('<b>🔧 Tool Output:</b><br>')
        cursor.insertPlainText(text)
        cursor.insertHtml('</div><br>')
        
        self.chat_display.setTextCursor(cursor)
        self.chat_display.scrollToBottom()
    
    def send_message(self):
        """Send the current input to the agent."""
        user_input = self.input_field.text().strip()
        if not user_input:
            return
        
        if self.agent is None:
            QMessageBox.warning(self, "Not Initialized", "Agent is not initialized. Please check settings.")
            return
        
        # Disable input during processing
        self.input_field.setEnabled(False)
        self.status_label.setText("Processing...")
        self.statusbar.showMessage("Agent is thinking...")
        
        # Add user message to chat
        self.add_user_message(user_input)
        self.input_field.clear()
        
        # Create and start worker thread
        self.worker = AgentWorker(self.agent, user_input)
        self.worker.finished.connect(self.on_agent_response)
        self.worker.error.connect(self.on_agent_error)
        self.worker.tool_output.connect(self.add_tool_output)
        self.worker.start()
    
    def send_command(self, command: str):
        """Send a slash command to the agent."""
        if command == "/plan":
            self.input_field.setText("/plan ")
            self.input_field.setFocus()
        elif command == "/scan":
            self.input_field.setText("/scan ")
            self.input_field.setFocus()
    
    def on_agent_response(self, response: str):
        """Handle agent response."""
        self.add_assistant_message(response)
        self.input_field.setEnabled(True)
        self.input_field.setFocus()
        self.status_label.setText("Ready")
        self.statusbar.showMessage("Ready")
    
    def on_agent_error(self, error: str):
        """Handle agent error."""
        self.add_system_message(f"❌ Error: {error}")
        self.input_field.setEnabled(True)
        self.input_field.setFocus()
        self.status_label.setText("Error")
        self.statusbar.showMessage("Error occurred")
    
    def clear_chat(self):
        """Clear the chat display and agent context."""
        self.chat_display.clear()
        if self.agent:
            self.agent.clear_context()
        self.add_system_message("Chat history cleared.")
    
    def retry_last_task(self):
        """Retry the last task."""
        if self.agent is None:
            return
        
        last_task = self.agent.retry_last_task()
        if last_task:
            self.add_system_message(f"Retrying: {last_task}")
            self.send_message_from_text(last_task)
        else:
            self.add_system_message("No previous task found to retry.")
    
    def send_message_from_text(self, text: str):
        """Send a message programmatically."""
        if not text:
            return
        
        self.input_field.setEnabled(False)
        self.status_label.setText("Processing...")
        self.statusbar.showMessage("Agent is thinking...")
        
        self.add_user_message(text)
        
        self.worker = AgentWorker(self.agent, text)
        self.worker.finished.connect(self.on_agent_response)
        self.worker.error.connect(self.on_agent_error)
        self.worker.tool_output.connect(self.add_tool_output)
        self.worker.start()
    
    def on_file_selected(self, file_path: str):
        """Handle file selection from the file explorer."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            self.add_system_message(f"📄 Opened: {file_path}")
            self.add_assistant_message(f"<pre>{content}</pre>")
        except Exception as e:
            self.add_system_message(f"❌ Error opening file: {e}")
    
    def open_workspace(self):
        """Open a different workspace directory."""
        directory = QFileDialog.getExistingDirectory(
            self, "Select Workspace Directory", self.workspace_dir
        )
        if directory:
            self.workspace_dir = directory
            self.file_explorer.workspace_dir = Path(directory)
            self.file_explorer.refresh_tree()
            self.initialize_agent()
    
    def show_settings(self):
        """Show the settings dialog."""
        dialog = SettingsDialog(self.model_config, self.workspace_dir, self)
        if dialog.exec_() == QDialog.Accepted:
            self.model_config = dialog.get_model_config()
            self.workspace_dir = dialog.get_workspace_dir()
            self.file_explorer.workspace_dir = Path(self.workspace_dir)
            self.file_explorer.refresh_tree()
            self.initialize_agent()
    
    def show_tools_list(self):
        """Show the list of available tools."""
        if self.agent is None:
            QMessageBox.warning(self, "Not Initialized", "Agent is not initialized.")
            return
        
        tools = self.agent.tool_registry.list_tools()
        tools_str = "\n".join([f"• {tool}" for tool in tools])
        QMessageBox.information(self, "Available Tools", f"Available tools:\n\n{tools_str}")
    
    def show_about(self):
        """Show the about dialog."""
        QMessageBox.about(
            self,
            "About Coding Agent",
            "<h2>Coding Agent GUI</h2>"
            "<p>An AI-powered programming assistant.</p>"
            "<p>Built with PyQt5</p>"
            "<p>Version 1.0.0</p>"
        )
    
    def closeEvent(self, event):
        """Handle window close event."""
        if self.agent:
            self.agent.close()
        event.accept()


def main():
    """Main entry point for the GUI application."""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    # Set application info
    app.setApplicationName("Coding Agent")
    app.setOrganizationName("CodingAgent")
    app.setApplicationVersion("1.0.0")
    
    # Create and show main window
    window = CodingAgentGUI()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
