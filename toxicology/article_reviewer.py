"""
Optimized Article Review Module using OpenAI GPT-4.1-mini
High-performance comprehensive analysis with ThreadPoolExecutor and exponential backoff
Purpose: Detailed toxicological analysis using OpenAI for endpoint extraction
Analysis Includes:
Toxicology Endpoints: NOAEL, LOAEL, EC50, LD50, NOEL
Study Design: Species, routes, duration, concentrations
Key Findings: Main results and clinical significance
Quality Assessment: Reliability scoring
Regulatory Info: Cramer class, guidelines mentioned

Features:

✅ Concurrent processing (up to 10 workers)
✅ Structured JSON output
✅ Error handling with fallbacks
✅ Comprehensive metadata extraction
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
import json
from openai import AsyncOpenAI
from dataclasses import dataclass
import re
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import backoff
import random
from google_scholar_scraper import SearchResult

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class ToxicologyEndpoints:
    """Data class for toxicology endpoints"""
    ec50: Optional[str] = None
    noael: Optional[str] = None
    noel: Optional[str] = None
    loeal: Optional[str] = None
    loael: Optional[str] = None
    ld50: Optional[str] = None
    other_endpoints: Optional[Dict[str, str]] = None

@dataclass
class StudyDesign:
    """Data class for study design information"""
    species: Optional[str] = None
    cell_type: Optional[str] = None
    route_of_exposure: Optional[str] = None
    duration: Optional[str] = None
    highest_concentration: Optional[str] = None
    test_system: Optional[str] = None

@dataclass
class ArticleReview:
    """Comprehensive article review data"""
    title: str
    doi: Optional[str] = None
    full_access: bool = False
    paywall_status: Optional[str] = None
    key_findings: Optional[str] = None
    study_summary: Optional[str] = None
    conclusion: Optional[str] = None
    citation: Optional[str] = None
    toxicology_endpoints: Optional[ToxicologyEndpoints] = None
    study_design: Optional[StudyDesign] = None
    cramer_class: Optional[str] = None
    toxic_effects: Optional[str] = None
    in_vitro_results: Optional[str] = None
    recommendations: Optional[str] = None

class OptimizedArticleReviewer:
    """
    High-performance article reviewer using OpenAI GPT-4.1-mini
    Features: ThreadPoolExecutor, exponential backoff, and concurrent processing
    """
    
    def __init__(self, openai_api_key: str, max_workers: int = 8):
        """
        Initialize the optimized article reviewer
        
        Args:
            openai_api_key (str): OpenAI API key
            max_workers (int): Maximum number of concurrent workers (reduced for review complexity)
        """
        self.client = AsyncOpenAI(api_key=openai_api_key)
        self.model = "gpt-4.1-mini-2025-04-14"
        self.max_workers = max_workers
        self.semaphore = asyncio.Semaphore(max_workers)  # Limit concurrent requests
        
        logger.info(f"OptimizedArticleReviewer initialized with model: {self.model}, max_workers: {max_workers}")
    
    def _create_review_prompt(self, compound_name: str, article: SearchResult) -> str:
        """
        Create optimized review prompt for toxicology analysis
        """
        # Prepare content for analysis (more efficient)
        content_parts = []
        
        # Include title
        if article.title:
            content_parts.append(f"TITLE: {article.title}")
        
        # Include authors (limit to first 3)
        if article.authors:
            content_parts.append(f"AUTHORS: {', '.join(article.authors[:3])}")
        
        # Include DOI if available
        if hasattr(article, 'doi') and article.doi:
            content_parts.append(f"DOI: {article.doi}")
        
        # Include abstract
        if article.abstract:
            content_parts.append(f"ABSTRACT: {article.abstract}")
        
        # Include limited full text (reduced to 3000 chars for speed)
        if article.full_text:
            full_text_excerpt = article.full_text[:3000]
            content_parts.append(f"FULL TEXT EXCERPT: {full_text_excerpt}")
        
        content_for_review = "\n\n".join(content_parts)
        
        # Optimized prompt (more concise)
        prompt = f"""Review this toxicology research about {compound_name}.

COMPOUND: {compound_name}
ARTICLE:
{content_for_review}

Provide a comprehensive toxicological analysis in JSON format:

{{
    "access_analysis": {{
        "full_paper_access": {bool(article.full_text)},
        "paywall_status": "full_access/abstract_only/behind_paywall",
        "content_completeness": "Brief description"
    }},
    "key_findings": {{
        "main_findings": "Key toxicological findings",
        "significance": "Clinical/regulatory significance"
    }},
    "study_summary": {{
        "objective": "Study objective",
        "methods_brief": "Methods used",
        "conclusion": "Main conclusion",
        "citation": "Author, Year, Title, Journal"
    }},
    "toxicology_endpoints": {{
        "ec50": "Effect concentration with units",
        "noael": "NOAEL with units and species",
        "noel": "NOEL with units and species", 
        "loeal": "LOEAL with units and species",
        "loael": "LOAEL with units and species",
        "ld50": "LD50 with units and species",
        "endpoints_available": true/false
    }},
    "study_design": {{
        "species": "Species used",
        "cell_type": "Cell line if in vitro",
        "route_of_exposure": "inhalation/oral/dermal/injection/in vitro",
        "duration": "Study duration",
        "highest_concentration": "Max concentration with units",
        "test_system": "in vivo/in vitro/in silico"
    }},
    "toxicity_assessment": {{
        "in_vitro_results": "Results with concentrations",
        "in_vivo_effects": "Observed toxic effects",
        "target_organs": "Affected organs/systems"
    }},
    "regulatory_information": {{
        "cramer_class": "TTC Cramer Class if applicable",
        "regulatory_guidelines": "Guidelines mentioned"
    }},
    "quality_assessment": {{
        "reliability": "High/Medium/Low",
        "limitations": "Study limitations"
    }},
    "recommendations": {{
        "for_toxicologists": "Key takeaways",
        "data_gaps": "Additional studies needed"
    }}
}}

Extract specific numerical values with units. If information is not available, state "Not available".
"""
        return prompt
    
    @backoff.on_exception(
        backoff.expo,
        Exception,
        max_tries=3,
        max_time=60,
        jitter=backoff.random_jitter,
    )
    async def _review_single_with_backoff(self, compound_name: str, article: SearchResult) -> Dict[str, Any]:
        """
        Review single article with exponential backoff retry logic
        """
        async with self.semaphore:  # Limit concurrent requests
            try:
                # Create the review prompt
                prompt = self._create_review_prompt(compound_name, article)
                
                # Add small random delay to avoid rate limiting
                await asyncio.sleep(random.uniform(0.2, 0.5))
                
                # Call OpenAI API with optimized parameters
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a senior toxicologist. Provide comprehensive analysis in valid JSON format. Extract specific numerical values with units."
                        },
                        {
                            "role": "user", 
                            "content": prompt
                        }
                    ],
                    temperature=0.1,
                    max_tokens=1500,  # Reduced for faster processing
                    response_format={"type": "json_object"}
                )
                
                # Parse the response
                review_result = json.loads(response.choices[0].message.content)
                
                # Add metadata
                review_result["review_metadata"] = {
                    "article_title": article.title,
                    "article_link": article.link,
                    "compound_analyzed": compound_name,
                    "review_timestamp": datetime.now().isoformat(),
                    "model_used": self.model,
                    "content_availability": "full_text" if article.full_text else "abstract_only"
                }
                
                # Add source information
                review_result["source_information"] = {
                    "doi": getattr(article, 'doi', 'Not available'),
                    "journal": "Not available",  # Could be extracted if needed
                    "publication_year": article.year,
                    "accessible_link": article.link
                }
                
                # Ensure all expected fields exist
                self._validate_review_structure(review_result)
                
                return review_result
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON parse error for '{article.title[:50]}...': {e}")
                raise  # Will trigger backoff retry
            except Exception as e:
                logger.error(f"Error reviewing '{article.title[:50]}...': {e}")
                raise  # Will trigger backoff retry
    
    def _validate_review_structure(self, review_result: Dict[str, Any]) -> None:
        """Ensure all expected fields exist in the review response"""
        required_sections = [
            "access_analysis", "key_findings", "study_summary", 
            "toxicology_endpoints", "study_design", "toxicity_assessment",
            "regulatory_information", "quality_assessment", "recommendations"
        ]
        
        for section in required_sections:
            if section not in review_result:
                review_result[section] = {}
                logger.warning(f"Missing section '{section}' in review response")
    
    async def review_article(self, compound_name: str, article: SearchResult) -> Dict[str, Any]:
        """
        Review a single article with retry logic and error handling
        """
        try:
            return await self._review_single_with_backoff(compound_name, article)
        except Exception as e:
            # If all retries failed, return error result
            logger.error(f"Final review failure for '{article.title[:50]}...': {e}")
            return self._create_error_review_response(article, compound_name, str(e))
    
    def _create_error_review_response(self, article: SearchResult, compound_name: str, error_msg: str) -> Dict[str, Any]:
        """Create error response with basic article information"""
        return {
            "access_analysis": {
                "full_paper_access": bool(article.full_text),
                "paywall_status": "error_in_analysis",
                "content_completeness": "Unable to analyze due to error"
            },
            "key_findings": {
                "main_findings": f"Error during analysis: {error_msg}",
                "significance": "Unable to determine"
            },
            "study_summary": {
                "objective": "Not available",
                "methods_brief": "Not available",
                "conclusion": "Analysis failed",
                "citation": "Unable to format citation"
            },
            "toxicology_endpoints": {
                "endpoints_available": False,
                "error": error_msg
            },
            "study_design": {
                "test_system": "Unable to determine",
                "error": error_msg
            },
            "toxicity_assessment": {
                "error": error_msg
            },
            "regulatory_information": {
                "error": error_msg
            },
            "quality_assessment": {
                "reliability": "Unable to assess due to error"
            },
            "recommendations": {
                "for_toxicologists": "Review failed - manual analysis required"
            },
            "source_information": {
                "doi": getattr(article, 'doi', 'Not available'),
                "accessible_link": article.link,
                "error": error_msg
            },
            "review_metadata": {
                "article_title": article.title,
                "compound_analyzed": compound_name,
                "review_timestamp": datetime.now().isoformat(),
                "error": error_msg
            }
        }
    
    async def review_articles_batch_optimized(
        self, 
        compound_name: str, 
        articles: List[SearchResult]
    ) -> Dict[str, Any]:
        """
        Optimized batch review using concurrent processing
        
        Args:
            compound_name (str): Name of the compound
            articles (List[SearchResult]): Articles to review
            
        Returns:
            Dict[str, Any]: Comprehensive batch review results
        """
        start_time = time.time()
        logger.info(f"Starting optimized toxicological review of {len(articles)} articles for {compound_name}")
        
        # Create tasks for all articles
        tasks = [
            self.review_article(compound_name, article) 
            for article in articles
        ]
        
        # Execute all tasks concurrently
        logger.info(f"Processing {len(tasks)} reviews concurrently with {self.max_workers} workers")
        review_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        reviewed_articles = []
        
        for i, review_result in enumerate(review_results):
            article = articles[i]
            
            # Handle exceptions in results
            if isinstance(review_result, Exception):
                logger.error(f"Exception for article {i}: {review_result}")
                review_result = self._create_error_review_response(article, compound_name, str(review_result))
            
            # Add review to article
            article.toxicology_review = review_result
            reviewed_articles.append(article)
        
        # Generate summary statistics
        review_summary = self._generate_review_summary(reviewed_articles, compound_name)
        
        processing_time = time.time() - start_time
        review_summary["processing_time_seconds"] = processing_time
        review_summary["articles_per_second"] = len(articles) / processing_time if processing_time > 0 else 0
        
        logger.info(f"Optimized review complete: {len(reviewed_articles)} articles in {processing_time:.2f}s")
        logger.info(f"Review rate: {review_summary['articles_per_second']:.1f} articles/second")
        
        return {
            "review_summary": review_summary,
            "reviewed_articles": reviewed_articles,
            "total_articles_reviewed": len(reviewed_articles),
            "compound": compound_name,
            "review_timestamp": datetime.now().isoformat()
        }
    
    # Keep the original method for backward compatibility
    async def review_articles_batch(
        self, 
        compound_name: str, 
        articles: List[SearchResult],
        batch_size: int = 3  # This parameter is ignored in optimized version
    ) -> Dict[str, Any]:
        """
        Wrapper for backward compatibility - uses optimized implementation
        """
        return await self.review_articles_batch_optimized(compound_name, articles)
    
    def _generate_review_summary(self, reviewed_articles: List[SearchResult], compound_name: str) -> Dict[str, Any]:
        """Generate summary statistics from reviewed articles (optimized)"""
        total_articles = len(reviewed_articles)
        
        # Access analysis
        full_access_count = sum(1 for a in reviewed_articles 
                               if a.toxicology_review.get("access_analysis", {}).get("full_paper_access", False))
        
        # Study types
        test_systems = {}
        routes_of_exposure = {}
        species_used = {}
        
        # Endpoints availability
        articles_with_endpoints = 0
        available_endpoints = {
            "EC50": 0, "NOAEL": 0, "NOEL": 0, "LOEAL": 0, "LOAEL": 0, "LD50": 0
        }
        
        # Quality assessment
        reliability_scores = {"High": 0, "Medium": 0, "Low": 0, "Unknown": 0}
        
        for article in reviewed_articles:
            review = article.toxicology_review
            
            # Study design analysis
            study_design = review.get("study_design", {})
            test_system = study_design.get("test_system", "Unknown")
            test_systems[test_system] = test_systems.get(test_system, 0) + 1
            
            route = study_design.get("route_of_exposure", "Unknown")
            routes_of_exposure[route] = routes_of_exposure.get(route, 0) + 1
            
            species = study_design.get("species", "Unknown")
            species_used[species] = species_used.get(species, 0) + 1
            
            # Endpoints analysis
            endpoints = review.get("toxicology_endpoints", {})
            if endpoints.get("endpoints_available", False):
                articles_with_endpoints += 1
                for endpoint in available_endpoints.keys():
                    endpoint_value = endpoints.get(endpoint.lower())
                    if endpoint_value and endpoint_value != "Not available":
                        available_endpoints[endpoint] += 1
            
            # Quality assessment
            quality = review.get("quality_assessment", {})
            reliability = quality.get("reliability", "Unknown")
            reliability_scores[reliability] = reliability_scores.get(reliability, 0) + 1
        
        return {
            "total_articles": total_articles,
            "access_summary": {
                "full_access": full_access_count,
                "abstract_only": total_articles - full_access_count,
                "full_access_percentage": (full_access_count / total_articles * 100) if total_articles > 0 else 0
            },
            "study_design_distribution": {
                "test_systems": test_systems,
                "routes_of_exposure": routes_of_exposure,
                "species_used": species_used
            },
            "endpoints_summary": {
                "articles_with_endpoints": articles_with_endpoints,
                "endpoint_availability": available_endpoints,
                "endpoint_coverage_percentage": (articles_with_endpoints / total_articles * 100) if total_articles > 0 else 0
            },
            "quality_assessment": {
                "reliability_distribution": reliability_scores,
                "high_quality_percentage": (reliability_scores["High"] / total_articles * 100) if total_articles > 0 else 0
            },
            "compound": compound_name
        }
    
    def export_comprehensive_reviews(self, review_results: Dict[str, Any], output_file: str = None) -> str:
        """
        Export comprehensive review results to JSON file
        """
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            compound = review_results.get("compound", "unknown")
            output_file = f"toxicology_review_{compound}_{timestamp}.json"
        
        # Format export data
        export_data = {
            "export_metadata": {
                "export_timestamp": datetime.now().isoformat(),
                "model_used": self.model,
                "review_type": "optimized_toxicology_analysis",
                "version": "2.0"
            },
            "summary": review_results["review_summary"],
            "detailed_reviews": [
                {
                    "article_title": article.title,
                    "doi": getattr(article, 'doi', None),
                    "link": article.link,
                    "toxicology_review": article.toxicology_review,
                    "filtering_info": {
                        "confidence": getattr(article, 'filtering_confidence', None),
                        "study_type": getattr(article, 'study_type', None),
                        "relevant_keywords": getattr(article, 'relevant_keywords', None)
                    }
                }
                for article in review_results["reviewed_articles"]
            ]
        }
        
        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Optimized toxicology reviews exported to {output_file}")
        return output_file

# Keep the original class as alias for backward compatibility
ArticleReviewer = OptimizedArticleReviewer