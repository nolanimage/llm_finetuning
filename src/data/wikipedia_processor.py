"""
Chinese Wikipedia data processor for fine-tuning.
Handles downloading and processing Chinese Wikipedia dumps.
"""

import os
import re
import bz2
import xml.etree.ElementTree as ET
import logging
from typing import List, Dict, Iterator, Optional
from pathlib import Path
import requests
from tqdm import tqdm
import json

logger = logging.getLogger(__name__)

class ChineseWikipediaProcessor:
    """Processor for Chinese Wikipedia dumps."""
    
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Chinese text cleaning patterns
        self.chinese_patterns = {
            'remove_refs': re.compile(r'<ref[^>]*>.*?</ref>', re.DOTALL),
            'remove_tags': re.compile(r'<[^>]+>'),
            'remove_templates': re.compile(r'\{\{[^}]*\}\}'),
            'remove_categories': re.compile(r'\[\[Category:[^\]]*\]\]'),
            'remove_links': re.compile(r'\[\[([^|\]]*?)(?:\|[^]]*?)?\]\]'),
            'remove_external_links': re.compile(r'\[https?://[^\s\]]+\s+([^\]]+)\]'),
            'remove_headers': re.compile(r'^=+\s*([^=]+)\s*=+$', re.MULTILINE),
            'remove_whitespace': re.compile(r'\s+'),
            'remove_punctuation': re.compile(r'[^\u4e00-\u9fff\w\s，。！？；：""''（）【】]'),
        }
    
    def download_wikipedia_dump(self, date: str = "20250320", 
                               file_type: str = "pages-articles") -> str:
        """
        Download Chinese Wikipedia dump.
        
        Args:
            date: Dump date in YYYYMMDD format
            file_type: Type of dump to download
            
        Returns:
            str: Path to downloaded file
        """
        base_url = "https://dumps.wikimedia.org/zhwiki"
        filename = f"zhwiki-{date}-{file_type}.xml.bz2"
        url = f"{base_url}/{date}/{filename}"
        
        filepath = self.data_dir / filename
        
        if filepath.exists():
            logger.info(f"File already exists: {filepath}")
            return str(filepath)
        
        logger.info(f"Downloading {url}...")
        
        # Download with progress bar
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filepath, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        logger.info(f"Downloaded to: {filepath}")
        return str(filepath)
    
    def extract_articles(self, dump_file: str, 
                        min_length: int = 100,
                        max_length: int = 2000) -> Iterator[Dict[str, str]]:
        """
        Extract articles from Wikipedia dump.
        
        Args:
            dump_file: Path to the dump file
            min_length: Minimum article length (characters)
            max_length: Maximum article length (characters)
            
        Yields:
            Dict with 'title' and 'text' keys
        """
        logger.info(f"Extracting articles from {dump_file}...")
        
        # Open bz2 compressed file
        with bz2.open(dump_file, 'rt', encoding='utf-8') as f:
            # Parse XML
            context = ET.iterparse(f, events=('start', 'end'))
            
            article_count = 0
            for event, elem in tqdm(context, desc="Processing articles"):
                if event == 'end' and elem.tag == '{http://www.mediawiki.org/xml/export-0.10/}page':
                    # Check if it's a regular article (not a template, category, etc.)
                    ns = elem.find('{http://www.mediawiki.org/xml/export-0.10/}ns')
                    if ns is not None and ns.text == '0':  # Main namespace
                        title_elem = elem.find('{http://www.mediawiki.org/xml/export-0.10/}title')
                        revision = elem.find('.//{http://www.mediawiki.org/xml/export-0.10/}revision')
                        
                        if title_elem is not None and revision is not None:
                            text_elem = revision.find('{http://www.mediawiki.org/xml/export-0.10/}text')
                            
                            if text_elem is not None and text_elem.text:
                                title = title_elem.text
                                text = text_elem.text
                                
                                # Clean the text
                                cleaned_text = self.clean_chinese_text(text)
                                
                                # Filter by length
                                if min_length <= len(cleaned_text) <= max_length:
                                    article_count += 1
                                    yield {
                                        'title': title,
                                        'text': cleaned_text,
                                        'length': len(cleaned_text)
                                    }
                                    
                                    if article_count % 1000 == 0:
                                        logger.info(f"Processed {article_count} articles")
                
                # Clear element to free memory
                elem.clear()
        
        logger.info(f"Total articles extracted: {article_count}")
    
    def clean_chinese_text(self, text: str) -> str:
        """
        Clean Chinese Wikipedia text.
        
        Args:
            text: Raw Wikipedia text
            
        Returns:
            str: Cleaned text
        """
        if not text:
            return ""
        
        # Remove Wikipedia markup
        text = self.chinese_patterns['remove_refs'].sub('', text)
        text = self.chinese_patterns['remove_tags'].sub('', text)
        text = self.chinese_patterns['remove_templates'].sub('', text)
        text = self.chinese_patterns['remove_categories'].sub('', text)
        
        # Handle links - keep the text, remove the markup
        text = self.chinese_patterns['remove_links'].sub(r'\1', text)
        text = self.chinese_patterns['remove_external_links'].sub(r'\1', text)
        
        # Remove headers
        text = self.chinese_patterns['remove_headers'].sub(r'\1', text)
        
        # Remove excessive whitespace
        text = self.chinese_patterns['remove_whitespace'].sub(' ', text)
        
        # Remove non-Chinese characters and punctuation (keep basic punctuation)
        text = self.chinese_patterns['remove_punctuation'].sub('', text)
        
        # Final cleanup
        text = text.strip()
        
        return text
    
    def save_articles_to_jsonl(self, articles: Iterator[Dict[str, str]], 
                              output_file: str) -> None:
        """
        Save articles to JSONL format.
        
        Args:
            articles: Iterator of article dictionaries
            output_file: Output file path
        """
        logger.info(f"Saving articles to {output_file}...")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for article in articles:
                f.write(json.dumps(article, ensure_ascii=False) + '\n')
        
        logger.info(f"Articles saved to {output_file}")
    
    def create_training_dataset(self, dump_file: str, 
                               output_file: str = "chinese_wiki_articles.jsonl",
                               min_length: int = 100,
                               max_length: int = 2000,
                               max_articles: Optional[int] = None) -> str:
        """
        Create a training dataset from Wikipedia dump.
        
        Args:
            dump_file: Path to Wikipedia dump file
            output_file: Output file name
            min_length: Minimum article length
            max_length: Maximum article length
            max_articles: Maximum number of articles to process
            
        Returns:
            str: Path to created dataset
        """
        output_path = self.data_dir / output_file
        
        articles = self.extract_articles(dump_file, min_length, max_length)
        
        if max_articles:
            # Limit number of articles
            limited_articles = []
            for i, article in enumerate(articles):
                if i >= max_articles:
                    break
                limited_articles.append(article)
            articles = limited_articles
        
        self.save_articles_to_jsonl(articles, str(output_path))
        
        return str(output_path)
    
    def load_articles_from_jsonl(self, file_path: str) -> List[Dict[str, str]]:
        """
        Load articles from JSONL file.
        
        Args:
            file_path: Path to JSONL file
            
        Returns:
            List of article dictionaries
        """
        articles = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                articles.append(json.loads(line.strip()))
        
        return articles
    
    def get_dataset_stats(self, file_path: str) -> Dict[str, any]:
        """
        Get statistics about the dataset.
        
        Args:
            file_path: Path to JSONL file
            
        Returns:
            Dictionary with dataset statistics
        """
        articles = self.load_articles_from_jsonl(file_path)
        
        lengths = [article['length'] for article in articles]
        
        stats = {
            'total_articles': len(articles),
            'avg_length': sum(lengths) / len(lengths) if lengths else 0,
            'min_length': min(lengths) if lengths else 0,
            'max_length': max(lengths) if lengths else 0,
            'total_characters': sum(lengths),
            'sample_titles': [article['title'] for article in articles[:5]]
        }
        
        return stats 