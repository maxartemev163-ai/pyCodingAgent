"""Skills and rules loader for the coding agent.

This module provides functionality to load skills and rules from Markdown files
in the workspace directory and pass them to the LLM as context.
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class SkillsLoader:
    """Loads skills and rules from Markdown files in the workspace.
    
    This class scans for .md files in designated directories (skills/, rules/, .agent/)
    and loads their content to be passed as context to the LLM.
    
    Attributes:
        workspace_dir: Root directory to scan for skill/rule files.
        skills_dirs: List of directory names to scan for skills/rules.
    """
    
    def __init__(
        self,
        workspace_dir: str = ".",
        skills_dirs: list[str] | None = None,
        max_files: int = 10,  # Limit number of skill files to reduce context
    ) -> None:
        """Initialize the skills loader.
        
        Args:
            workspace_dir: Root directory to scan for skill/rule files.
            skills_dirs: List of directory names to scan. Defaults to 
                ['skills', 'rules', '.agent'].
            max_files: Maximum number of skill files to load (prevents context bloat).
        """
        self.workspace_dir = Path(workspace_dir).resolve()
        self.skills_dirs = skills_dirs or ['skills', 'rules', '.agent']
        self.max_files = max_files
    
    def find_skill_files(self) -> list[Path]:
        """Find all Markdown files containing skills/rules.
        
        Searches in:
        - Root workspace directory (*.md files)
        - Subdirectories named 'skills', 'rules', or '.agent'
        
        Returns:
            List of paths to skill/rule Markdown files (limited by max_files).
        """
        skill_files = []
        
        # Check root directory for .md files (excluding common non-skill files)
        excluded_root_files = {
            'README.md', 'CHANGELOG.md', 'CONTRIBUTING.md', 
            'LICENSE.md', 'requirements.txt', 'setup.py',
            'pyproject.toml', '.gitignore'
        }
        
        for md_file in self.workspace_dir.glob('*.md'):
            if md_file.name not in excluded_root_files:
                # Check if file might contain skills/rules based on name
                name_lower = md_file.name.lower()
                if any(keyword in name_lower for keyword in ['skill', 'rule', 'guide', 'standard', 'practice', 'convention']):
                    skill_files.append(md_file)
        
        # Check designated subdirectories
        for dir_name in self.skills_dirs:
            skills_dir = self.workspace_dir / dir_name
            if skills_dir.exists() and skills_dir.is_dir():
                for md_file in skills_dir.glob('*.md'):
                    # Exclude README files in subdirectories too
                    if md_file.name.upper() == 'README.MD':
                        continue
                    skill_files.append(md_file)
                # Also check subdirectories within skills dirs
                for subdir in skills_dir.iterdir():
                    if subdir.is_dir():
                        for md_file in subdir.glob('*.md'):
                            skill_files.append(md_file)
        
        # Limit number of files to prevent context bloat
        if len(skill_files) > self.max_files:
            logger.warning(
                f"Found {len(skill_files)} skill files, limiting to {self.max_files}. "
                "Prioritizing files by name relevance."
            )
            # Prioritize files with 'rule' or 'skill' in name
            priority_files = [f for f in skill_files if 'rule' in f.name.lower() or 'skill' in f.name.lower()]
            other_files = [f for f in skill_files if f not in priority_files]
            skill_files = priority_files[:self.max_files] if len(priority_files) >= self.max_files else priority_files + other_files[:self.max_files - len(priority_files)]
        
        return sorted(set(skill_files))
    
    def load_skill_content(self, file_path: Path) -> str:
        """Load content from a single skill/rule file.
        
        Args:
            file_path: Path to the Markdown file.
            
        Returns:
            Content of the file as a string.
        """
        try:
            content = file_path.read_text(encoding='utf-8')
            logger.info(f"Loaded skill/rule file: {file_path.relative_to(self.workspace_dir)}")
            return content.strip()
        except Exception as e:
            logger.warning(f"Failed to load skill file {file_path}: {e}")
            return ""
    
    def load_all_skills(self) -> dict[str, str]:
        """Load all skills and rules from Markdown files.
        
        Returns:
            Dictionary mapping file paths (relative to workspace) to their content.
        """
        skills = {}
        skill_files = self.find_skill_files()
        
        for file_path in skill_files:
            relative_path = str(file_path.relative_to(self.workspace_dir))
            content = self.load_skill_content(file_path)
            if content:
                skills[relative_path] = content
        
        logger.info(f"Loaded {len(skills)} skill/rule file(s)")
        return skills
    
    def format_for_context(self, skills: dict[str, str] | None = None) -> str:
        """Format loaded skills for inclusion in LLM context.
        
        Args:
            skills: Dictionary of skills to format. If None, loads all skills.
            
        Returns:
            Formatted string suitable for passing to LLM as context.
        """
        if skills is None:
            skills = self.load_all_skills()
        
        if not skills:
            return ""
        
        sections = []
        sections.append("=" * 60)
        sections.append("SKILLS AND RULES CONTEXT")
        sections.append("=" * 60)
        sections.append("")
        sections.append("The following skills and rules have been loaded from Markdown files")
        sections.append("in the workspace directory. Please follow these guidelines when")
        sections.append("assisting with coding tasks:")
        sections.append("")
        
        for file_path, content in skills.items():
            sections.append("-" * 40)
            sections.append(f"Source: {file_path}")
            sections.append("-" * 40)
            sections.append(content)
            sections.append("")
        
        sections.append("=" * 60)
        sections.append("END OF SKILLS AND RULES")
        sections.append("=" * 60)
        
        return "\n".join(sections)
    
    def get_skills_context(self) -> Optional[str]:
        """Get formatted skills context for LLM.
        
        Convenience method that loads and formats all skills in one call.
        
        Returns:
            Formatted skills context string, or None if no skills found.
        """
        skills = self.load_all_skills()
        if not skills:
            return None
        return self.format_for_context(skills)


def load_skills_context(workspace_dir: str = ".", max_files: int = 10) -> Optional[str]:
    """Load and format skills/rules context from workspace directory.
    
    This is a convenience function that creates a SkillsLoader and returns
    the formatted context string.
    
    Args:
        workspace_dir: Root directory to scan for skill/rule files.
        max_files: Maximum number of skill files to load (prevents context bloat).
        
    Returns:
        Formatted skills context string, or None if no skills found.
    """
    loader = SkillsLoader(workspace_dir=workspace_dir, max_files=max_files)
    return loader.get_skills_context()
