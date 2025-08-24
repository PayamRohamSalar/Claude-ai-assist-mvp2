"""
Legal Structure Parser

Role: Lightly parse structure: chapters (فصل), articles (ماده N), notes (تبصره N), and clauses (بند …).
"""

import re
import sys
from dataclasses import dataclass
from typing import List
from pathlib import Path

# Add parent directory to path for shared_utils imports
sys.path.append(str(Path(__file__).parent.parent))

from shared_utils.logger import get_logger
from shared_utils.constants import persian_to_english_digits

logger = get_logger(__name__)


@dataclass
class Article:
    """Represents a legal article with its content, notes, and clauses."""
    number: str
    text: str
    notes: List[str]
    clauses: List[str]


@dataclass
class Chapter:
    """Represents a legal chapter containing articles."""
    title: str
    articles: List[Article]


@dataclass
class ParsedStructure:
    """Container for the complete parsed legal document structure."""
    chapters: List[Chapter]
    total_chapters: int
    total_articles: int
    total_notes: int


class LegalStructureParser:
    """Parser for Persian legal document structures."""
    
    def __init__(self):
        # Regex patterns for different structural elements
        self.chapter_pattern = re.compile(r'^فصل\s+(.+)', re.MULTILINE)
        self.article_pattern = re.compile(r'^ماده\s+([۰-۹]+|\d+)', re.MULTILINE)
        self.note_pattern = re.compile(r'^تبصره\s*([۰-۹]*|\d*)\s*[-:]?\s*(.+)', re.MULTILINE)
        self.clause_pattern = re.compile(r'^بند\s+([الف-ی])\s*[-:]?\s*(.+)', re.MULTILINE)
    
    def parse(self, text: str) -> ParsedStructure:
        """
        Parse the legal document text into structured components.
        
        Args:
            text: The legal document text to parse
            
        Returns:
            ParsedStructure containing parsed chapters, articles, notes, and clauses
        """
        logger.info("Starting legal structure parsing")
        
        # Convert Persian digits to ASCII for consistency
        normalized_text = persian_to_english_digits(text)
        
        # Find chapters
        chapters = self._extract_chapters(normalized_text)
        
        # Calculate totals
        total_chapters = len(chapters)
        total_articles = sum(len(chapter.articles) for chapter in chapters)
        total_notes = sum(
            len(article.notes) 
            for chapter in chapters 
            for article in chapter.articles
        )
        
        logger.info(f"Parsed {total_chapters} chapters, {total_articles} articles, {total_notes} notes")
        
        return ParsedStructure(
            chapters=chapters,
            total_chapters=total_chapters,
            total_articles=total_articles,
            total_notes=total_notes
        )
    
    def _extract_chapters(self, text: str) -> List[Chapter]:
        """Extract chapters from the text."""
        chapter_matches = list(self.chapter_pattern.finditer(text))
        
        if not chapter_matches:
            # No explicit chapters, create a pseudo-chapter
            logger.info("No chapters found, creating pseudo-chapter 'فصل واحد'")
            articles = self._extract_articles(text)
            return [Chapter(title="فصل واحد", articles=articles)]
        
        chapters = []
        for i, match in enumerate(chapter_matches):
            chapter_title = match.group(1).strip()
            
            # Determine chapter text boundaries
            start_pos = match.start()
            end_pos = chapter_matches[i + 1].start() if i + 1 < len(chapter_matches) else len(text)
            chapter_text = text[start_pos:end_pos]
            
            # Extract articles within this chapter
            articles = self._extract_articles(chapter_text)
            chapters.append(Chapter(title=chapter_title, articles=articles))
            
            logger.debug(f"Extracted chapter: {chapter_title} with {len(articles)} articles")
        
        return chapters
    
    def _extract_articles(self, text: str) -> List[Article]:
        """Extract articles from the given text."""
        article_matches = list(self.article_pattern.finditer(text))
        
        if not article_matches:
            # No explicit articles, create one article with full text
            logger.info("No articles found, creating single article '1' with full text")
            notes = self._extract_notes(text)
            clauses = self._extract_clauses(text)
            return [Article(number="1", text=text.strip(), notes=notes, clauses=clauses)]
        
        articles = []
        for i, match in enumerate(article_matches):
            article_number = match.group(1)
            
            # Determine article text boundaries
            start_pos = match.start()
            end_pos = article_matches[i + 1].start() if i + 1 < len(article_matches) else len(text)
            article_text = text[start_pos:end_pos]
            
            # Extract notes and clauses within this article
            notes = self._extract_notes(article_text)
            clauses = self._extract_clauses(article_text)
            
            # Clean article text by removing the article header
            clean_text = re.sub(r'^ماده\s+\d+\s*[-:]?\s*', '', article_text, flags=re.MULTILINE)
            
            articles.append(Article(
                number=article_number,
                text=clean_text.strip(),
                notes=notes,
                clauses=clauses
            ))
            
            logger.debug(f"Extracted article {article_number} with {len(notes)} notes and {len(clauses)} clauses")
        
        return articles
    
    def _extract_notes(self, text: str) -> List[str]:
        """Extract notes (تبصره) from the text."""
        notes = []
        for match in self.note_pattern.finditer(text):
            note_number = match.group(1) if match.group(1) else ""
            note_content = match.group(2).strip()
            
            # Format note with number if present
            if note_number:
                note_label = f"تبصره {note_number}"
            else:
                note_label = "تبصره"
            
            notes.append(f"{note_label}: {note_content}")
        
        return notes
    
    def _extract_clauses(self, text: str) -> List[str]:
        """Extract clauses (بند) from the text."""
        clauses = []
        for match in self.clause_pattern.finditer(text):
            clause_letter = match.group(1)
            clause_content = match.group(2).strip()
            clauses.append(f"بند {clause_letter}: {clause_content}")
        
        return clauses
