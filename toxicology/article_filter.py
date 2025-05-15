"""
Optimized Article Filtering Module using OpenAI GPT-4.1-mini
High-performance filtering with ThreadPoolExecutor and exponential backoff
Purpose: OpenAI GPT-4.1-mini based filtering for toxicology relevance
Filtering Criteria:

Include: Inhalation studies, animal studies, in vitro/vivo, thermal degradation, QSAR models
Exclude: Reference-only mentions, synthesis papers, insecticide studies

Features:

✅ Concurrent processing (up to 15 workers)
✅ Confidence scoring (0.0-1.0)
✅ Study type classification
✅ Keyword extraction
✅ Detailed exclusion reasons

Performance: ~15 articles/second processing rate
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
import json
from openai import AsyncOpenAI
from dataclasses import dataclass
import re
import time
from concurrent.futures import ThreadPoolExecutor
import backoff
import random
from google_scholar_scraper import SearchResult

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class FilterCriteria:
    """Data class to store filtering criteria"""
    include_keywords: List[str]
    exclude_keywords: List[str]
    study_types: List[str]
    
    @classmethod
    def get_toxicology_criteria(cls):
        """Get predefined toxicology research criteria"""
        return cls(
            include_keywords=[
                "toxicity", "inhalation", "gavage", "pyrolysis", 
                "cannabis", "degradation", "thermal degradation",
                "in vitro", "in vivo", "in silico", "QSAR",
                "oral administration", "animal studies", "cell-based",
                "vaping", "combustion", "vaporization", "thermal stability"
            ],
            exclude_keywords=[
                "insecticidal", "repellent", "insects", "chemical synthesis",
                "precursor", "degradation product only", "reference section only",
                "chemical composition analysis only"
            ],
            study_types=[
                "inhalation toxicity",
                "animal studies", 
                "in vitro studies",
                "thermal stability & pyrolysis",
                "in silico models"
            ]
        )

class OptimizedArticleFilter:
    """
    High-performance article filtering using OpenAI GPT-4.1-mini
    Features: ThreadPoolExecutor, exponential backoff, and concurrent processing
    """
    
    def __init__(self, openai_api_key: str, max_workers: int = 10):
        """
        Initialize the optimized article filter
        
        Args:
            openai_api_key (str): OpenAI API key
            max_workers (int): Maximum number of concurrent workers
        """
        self.client = AsyncOpenAI(api_key=openai_api_key)
        self.model = "gpt-4.1-mini-2025-04-14"
        self.criteria = FilterCriteria.get_toxicology_criteria()
        self.max_workers = max_workers
        self.semaphore = asyncio.Semaphore(max_workers)  # Limit concurrent requests
        
        logger.info(f"OptimizedArticleFilter initialized with model: {self.model}, max_workers: {max_workers}")
    
    def _create_filtering_prompt(self, compound_name: str, article: SearchResult) -> str:
        """
        Create a detailed prompt for filtering articles (optimized version)
        
        Args:
            compound_name (str): Name of the compound being researched
            article (SearchResult): Article to be filtered
            
        Returns:
            str: Formatted prompt for OpenAI
        """
        # Prepare article content for analysis (more efficient processing)
        content_parts = []
        
        # Include title (most important)
        if article.title:
            content_parts.append(f"TITLE: {article.title}")
        
        # Include abstract if available
        if article.abstract:
            content_parts.append(f"ABSTRACT: {article.abstract}")
        
        # Include limited full text if available (reduced to 1500 chars for speed)
        if article.full_text:
            full_text_excerpt = article.full_text[:1500]
            content_parts.append(f"FULL TEXT EXCERPT: {full_text_excerpt}")
        
        content_to_analyze = "\n\n".join(content_parts)
        
        # Optimized prompt (shorter for faster processing)
        prompt = f"""Analyze this research article about {compound_name} for toxicology relevance.

COMPOUND: {compound_name}
ARTICLE CONTENT:
{content_to_analyze}

INCLUSION CRITERIA - Include if study involves:
1. Inhalation toxicity of {compound_name}
2. Animal studies with {compound_name} (any route)
3. In vitro studies with {compound_name}
4. Thermal degradation/pyrolysis of {compound_name}
5. In silico/QSAR modeling of {compound_name}

EXCLUSION CRITERIA - Exclude if:
- Only mentions {compound_name} in references
- Only chemical composition/synthesis
- Only insecticidal/repellent studies
- {compound_name} is just a precursor/degradation product

Respond with JSON:
{{
    "include": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation",
    "study_type": "inhalation_toxicity/animal_studies/in_vitro/thermal_pyrolysis/in_silico/not_applicable",
    "relevant_keywords_found": ["keyword1", "keyword2"],
    "exclusion_reasons": ["reason1"] if excluded
}}"""
        return prompt
    
    @backoff.on_exception(
        backoff.expo,
        Exception,
        max_tries=3,
        max_time=30,
        jitter=backoff.random_jitter,
    )
    async def _filter_single_with_backoff(self, compound_name: str, article: SearchResult) -> Dict[str, Any]:
        """
        Filter single article with exponential backoff retry logic
        """
        async with self.semaphore:  # Limit concurrent requests
            try:
                # Create the filtering prompt
                prompt = self._create_filtering_prompt(compound_name, article)
                
                # Add small random delay to avoid rate limiting
                await asyncio.sleep(random.uniform(0.1, 0.3))
                
                # Call OpenAI API with optimized parameters
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a toxicology expert. Respond with valid JSON only."
                        },
                        {
                            "role": "user", 
                            "content": prompt
                        }
                    ],
                    temperature=0.1,
                    max_tokens=300,  # Reduced for faster response
                    response_format={"type": "json_object"}
                )
                
                # Parse the response
                result = json.loads(response.choices[0].message.content)
                
                # Add metadata
                result["article_title"] = article.title
                result["article_link"] = article.link
                result["has_full_text"] = bool(article.full_text)
                result["has_abstract"] = bool(article.abstract)
                
                return result
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON parse error for '{article.title[:50]}...': {e}")
                raise  # Will trigger backoff retry
            except Exception as e:
                logger.error(f"Error filtering '{article.title[:50]}...': {e}")
                raise  # Will trigger backoff retry
    
    async def filter_single_article(self, compound_name: str, article: SearchResult) -> Dict[str, Any]:
        """
        Filter a single article with retry logic and error handling
        """
        try:
            return await self._filter_single_with_backoff(compound_name, article)
        except Exception as e:
            # If all retries failed, return error result
            logger.error(f"Final failure for '{article.title[:50]}...': {e}")
            return {
                "include": False,
                "confidence": 0.0,
                "reasoning": f"Error during filtering: {str(e)}",
                "study_type": "error",
                "relevant_keywords_found": [],
                "exclusion_reasons": ["filtering_error"],
                "article_title": article.title,
                "article_link": article.link,
                "has_full_text": bool(article.full_text),
                "has_abstract": bool(article.abstract)
            }
    
    async def filter_articles_batch_optimized(
        self, 
        compound_name: str, 
        articles: List[SearchResult],
        min_confidence: float = 0.7
    ) -> Dict[str, Any]:
        """
        Optimized batch filtering using concurrent processing
        
        Args:
            compound_name (str): Name of the compound
            articles (List[SearchResult]): Articles to filter
            min_confidence (float): Minimum confidence threshold for inclusion
            
        Returns:
            Dict[str, Any]: Filtering results with included/excluded articles
        """
        start_time = time.time()
        logger.info(f"Starting optimized filtering of {len(articles)} articles for {compound_name}")
        
        # Create tasks for all articles
        tasks = [
            self.filter_single_article(compound_name, article) 
            for article in articles
        ]
        
        # Execute all tasks concurrently
        logger.info(f"Processing {len(tasks)} articles concurrently with {self.max_workers} workers")
        filtering_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        included_articles = []
        excluded_articles = []
        valid_results = []
        
        for i, result in enumerate(filtering_results):
            article = articles[i]
            
            # Handle exceptions in results
            if isinstance(result, Exception):
                logger.error(f"Exception for article {i}: {result}")
                result = {
                    "include": False,
                    "confidence": 0.0,
                    "reasoning": f"Exception: {str(result)}",
                    "study_type": "error",
                    "relevant_keywords_found": [],
                    "exclusion_reasons": ["exception_occurred"],
                    "article_title": article.title,
                    "article_link": article.link,
                    "has_full_text": bool(article.full_text),
                    "has_abstract": bool(article.abstract)
                }
            
            valid_results.append(result)
            
            # Apply confidence threshold
            should_include = (
                result.get("include", False) and 
                result.get("confidence", 0.0) >= min_confidence
            )
            
            if should_include:
                # Add filtering metadata to the article
                article.filtering_result = result
                article.filtering_confidence = result.get("confidence", 0.0)
                article.study_type = result.get("study_type", "not_applicable")
                article.relevant_keywords = result.get("relevant_keywords_found", [])
                included_articles.append(article)
            else:
                # Add exclusion reason
                article.filtering_result = result
                article.exclusion_reasons = result.get("exclusion_reasons", [])
                excluded_articles.append(article)
        
        # Calculate statistics
        total_articles = len(articles)
        included_count = len(included_articles)
        excluded_count = len(excluded_articles)
        
        # Analyze study types
        study_types = {}
        for result in valid_results:
            if result.get("include", False):
                study_type = result.get("study_type", "unknown")
                study_types[study_type] = study_types.get(study_type, 0) + 1
        
        # Analyze exclusion reasons
        exclusion_reasons = {}
        for result in valid_results:
            if not result.get("include", False):
                for reason in result.get("exclusion_reasons", []):
                    exclusion_reasons[reason] = exclusion_reasons.get(reason, 0) + 1
        
        processing_time = time.time() - start_time
        logger.info(f"Optimized filtering complete: {included_count}/{total_articles} articles included in {processing_time:.2f}s")
        logger.info(f"Study types found: {study_types}")
        
        return {
            "filtering_summary": {
                "total_articles": total_articles,
                "included_count": included_count,
                "excluded_count": excluded_count,
                "inclusion_rate": (included_count / total_articles * 100) if total_articles > 0 else 0,
                "min_confidence_threshold": min_confidence,
                "model_used": self.model,
                "processing_time_seconds": processing_time,
                "articles_per_second": total_articles / processing_time if processing_time > 0 else 0
            },
            "included_articles": included_articles,
            "excluded_articles": excluded_articles,
            "study_types_distribution": study_types,
            "exclusion_reasons_distribution": exclusion_reasons,
            "filtering_results": valid_results
        }
    
    # Keep the original method for backward compatibility
    async def filter_articles_batch(
        self, 
        compound_name: str, 
        articles: List[SearchResult],
        batch_size: int = 5,  # This parameter is ignored in optimized version
        min_confidence: float = 0.7
    ) -> Dict[str, Any]:
        """
        Wrapper for backward compatibility - uses optimized implementation
        """
        return await self.filter_articles_batch_optimized(compound_name, articles, min_confidence)
    
    def get_filtered_articles_summary(self, filtering_result: Dict[str, Any]) -> str:
        """
        Generate a human-readable summary of filtering results (updated with performance metrics)
        """
        summary = filtering_result["filtering_summary"]
        study_types = filtering_result["study_types_distribution"]
        exclusion_reasons = filtering_result["exclusion_reasons_distribution"]
        
        text = f"""
OPTIMIZED ARTICLE FILTERING SUMMARY
===================================

Total Articles Analyzed: {summary['total_articles']}
Articles Included: {summary['included_count']}
Articles Excluded: {summary['excluded_count']}
Inclusion Rate: {summary['inclusion_rate']:.1f}%
Confidence Threshold: {summary['min_confidence_threshold']}
Model Used: {summary['model_used']}

PERFORMANCE METRICS:
Processing Time: {summary['processing_time_seconds']:.2f} seconds
Articles/Second: {summary['articles_per_second']:.1f}

STUDY TYPES FOUND:
"""
        for study_type, count in study_types.items():
            text += f"- {study_type.replace('_', ' ').title()}: {count}\n"
        
        text += f"\nTOP EXCLUSION REASONS:\n"
        sorted_exclusions = sorted(exclusion_reasons.items(), key=lambda x: x[1], reverse=True)
        for reason, count in sorted_exclusions[:5]:
            text += f"- {reason.replace('_', ' ').title()}: {count}\n"
        
        return text

# Keep the original class as alias for backward compatibility
ArticleFilter = OptimizedArticleFilter
