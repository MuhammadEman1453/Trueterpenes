"""
Enhanced Google Scholar Scraper with Article Content Extraction
Uses Marco Gullo's scraper + Apify's Article Extractor Smart
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional, Set
import json
from datetime import datetime
import os
from apify_client import ApifyClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class CompoundInfo:
    """Data class to store compound information"""
    name: str
    cas_number: str
    synonyms: List[str] = None
    
    def __post_init__(self):
        if self.synonyms is None:
            self.synonyms = []

@dataclass
class SearchResult:
    """Data class for search results with full content"""
    title: str
    authors: List[str]
    abstract: str
    link: str
    pdf_link: Optional[str]
    citation_count: int
    year: int
    search_term: str
    full_text: Optional[str] = None  # Added for extracted content
    content_extracted: bool = False  # Flag to track extraction status
    extraction_method: Optional[str] = None  # Track how content was extracted
    
class GoogleScholarScraper:
    """
    Enhanced Google Scholar scraper with full content extraction
    Uses Marco Gullo's scraper + Apify's Article Extractor
    """
    
    # Toxicology-specific keywords
    TOXICOLOGY_KEYWORDS = [
        'toxicity',
        'inhalation',
        'gavage',
        'degradation',
        'pyrolysis',
        'vape',
        'cannabis'
    ]
    
    def __init__(self, apify_token: str):
        """
        Initialize the scraper with Apify credentials
        
        Args:
            apify_token (str): Apify API token
        """
        self.client = ApifyClient(apify_token)
        self.results = []
        
        # Using Marco Gullo's Google Scholar scraper
        self.scholar_actor_id = "marco.gullo/google-scholar-scraper"
        
        # Using Apify's Smart Article Extractor
        self.extractor_actor_id = "lukaskrivka/article-extractor-smart"
        
        logger.info("Enhanced GoogleScholarScraper initialized")
        logger.info(f"Scholar Actor: {self.scholar_actor_id}")
        logger.info(f"Extractor Actor: {self.extractor_actor_id}")
    
    def generate_search_queries(self, compound: CompoundInfo) -> List[str]:
        """Generate targeted search queries for a compound"""
        queries = []
        
        # Get all identifiers (name, CAS, synonyms)
        identifiers = [compound.name, compound.cas_number] + compound.synonyms
        
        # Generate queries: identifier + keyword combinations
        for identifier in identifiers:
            for keyword in self.TOXICOLOGY_KEYWORDS:
                # Primary queries
                queries.append(f'"{identifier}" {keyword}')
        
        # Remove duplicates while preserving order
        seen = set()
        unique_queries = []
        for query in queries:
            if query not in seen:
                seen.add(query)
                unique_queries.append(query)
        
        logger.info(f"Generated {len(unique_queries)} search queries")
        return unique_queries[:15]  # Limit to 15 queries for performance
    
    async def run_search(self, query: str, max_results: int = 20) -> List[Dict]:
        """
        Run a single search query using Marco Gullo's actor
        
        Args:
            query (str): Search query
            max_results (int): Maximum results per query
            
        Returns:
            List[Dict]: Search results
        """
        try:
            logger.info(f"Running search: {query}")
            
            # Configure input for Marco Gullo's actor
            run_input = {
                "keyword": query,
                "maxResults": min(max_results, 50),
                "proxyOptions": {
                    "useApifyProxy": True,
                    "apifyProxyGroups": ["RESIDENTIAL"]
                },
                "includeAbstract": True,
                "includePdf": True,
                "includeAuthors": True,
                "includeCitations": True
            }
            
            # Run the actor
            run = self.client.actor(self.scholar_actor_id).call(run_input=run_input)
            
            # Fetch results
            results = []
            for item in self.client.dataset(run["defaultDatasetId"]).iterate_items():
                # Add search term to each result for tracking
                item['search_term'] = query
                results.append(item)
            
            logger.info(f"Found {len(results)} results for query: {query}")
            return results
            
        except Exception as e:
            logger.error(f"Error in search for query '{query}': {e}")
            return []
    
    async def extract_article_content(self, urls: List[str]) -> Dict[str, Dict]:
        """
        Extract content from article URLs using Apify's Smart Article Extractor
        
        Args:
            urls (List[str]): List of URLs to extract content from
            
        Returns:
            Dict[str, Dict]: Mapping of URL to extracted content
        """
        if not urls:
            return {}
        
        try:
            logger.info(f"Extracting content from {len(urls)} URLs")
            
            # Configure the article extractor
            run_input = {
                "urls": urls,
                "outputFormat": "json",
                "includeMetadata": True,
                "timeout": 30,
                "waitFor": 3,  # Wait 3 seconds for page to load
                "skipExternalImages": True
            }
            
            # Run the article extractor
            run = self.client.actor(self.extractor_actor_id).call(run_input=run_input)
            
            # Process results
            content_map = {}
            for item in self.client.dataset(run["defaultDatasetId"]).iterate_items():
                url = item.get("url")
                if url:
                    content_map[url] = {
                        "title": item.get("title", ""),
                        "text": item.get("text", ""),
                        "abstract": item.get("excerpt", "") or item.get("description", ""),
                        "extracted": True,
                        "length": len(item.get("text", ""))
                    }
            
            logger.info(f"Successfully extracted content from {len(content_map)} URLs")
            return content_map
            
        except Exception as e:
            logger.error(f"Error extracting article content: {e}")
            return {}
    
    async def search_compound(self, compound: CompoundInfo, max_results_per_query: int = 15) -> List[SearchResult]:
        """
        Complete workflow: Search Google Scholar and extract full content
        
        Args:
            compound (CompoundInfo): Compound to search for
            max_results_per_query (int): Max results per individual query
            
        Returns:
            List[SearchResult]: Search results with extracted content
        """
        logger.info(f"Starting enhanced search for compound: {compound.name} (CAS: {compound.cas_number})")
        
        # Phase 1: Search Google Scholar
        logger.info("Phase 1: Searching Google Scholar...")
        queries = self.generate_search_queries(compound)
        
        all_results = []
        batch_size = 3  # Small batches for stability
        
        for i in range(0, len(queries), batch_size):
            batch = queries[i:i + batch_size]
            
            # Run batch in parallel
            batch_tasks = [self.run_search(query, max_results_per_query) for query in batch]
            batch_results = await asyncio.gather(*batch_tasks)
            
            # Combine results
            for results in batch_results:
                all_results.extend(results)
            
            # Rate limiting: wait between batches
            await asyncio.sleep(5)
            
            # Progress update
            logger.info(f"Completed {min(i + batch_size, len(queries))}/{len(queries)} query batches")
        
        # Convert to SearchResult objects and deduplicate
        search_results = self._process_results(all_results)
        
        # Phase 2: Extract full content from articles
        logger.info("Phase 2: Extracting full article content...")
        search_results = await self._extract_content_from_results(search_results)
        
        logger.info(f"Pipeline complete! Found {len(search_results)} unique papers")
        return search_results
    
    def _process_results(self, raw_results: List[Dict]) -> List[SearchResult]:
        """Process and deduplicate search results"""
        seen_titles = set()
        processed_results = []
        
        for result in raw_results:
            try:
                # Extract data with fallbacks for Marco's format
                title = result.get('title', '') or result.get('Title', '')
                title = title.strip()
                
                # Skip if we've seen this title or if title is empty
                if not title or title in seen_titles:
                    continue
                
                seen_titles.add(title)
                
                # Extract authors
                authors = result.get('authors', []) or result.get('Authors', [])
                if isinstance(authors, str):
                    authors = [authors]
                
                # Get links
                link = (result.get('url', '') or 
                       result.get('URL', '') or 
                       result.get('link', '') or 
                       result.get('Link', ''))
                
                pdf_link = (result.get('pdfUrl', None) or 
                           result.get('PdfUrl', None) or 
                           result.get('pdf_link', None) or 
                           result.get('PdfLink', None))
                
                # Create SearchResult object
                search_result = SearchResult(
                    title=title,
                    authors=authors,
                    abstract=result.get('abstract', '') or result.get('Abstract', ''),
                    link=link,
                    pdf_link=pdf_link,
                    citation_count=int(result.get('citations', 0) or result.get('Citations', 0) or 0),
                    year=int(result.get('year', 0) or result.get('Year', 0) or 0),
                    search_term=result.get('search_term', '')
                )
                
                processed_results.append(search_result)
                
            except Exception as e:
                logger.warning(f"Error processing result: {e}")
                continue
        
        return processed_results
    
    async def _extract_content_from_results(self, search_results: List[SearchResult]) -> List[SearchResult]:
        """
        Extract full content from search results
        
        Args:
            search_results (List[SearchResult]): Search results with links
            
        Returns:
            List[SearchResult]: Search results with extracted content
        """
        # Collect all URLs to extract content from
        urls_to_extract = []
        url_to_result_map = {}
        
        for result in search_results:
            # Prefer PDF links, then regular links
            urls_to_try = [result.pdf_link, result.link]
            
            for url in urls_to_try:
                if url and url not in url_to_result_map:
                    urls_to_extract.append(url)
                    if url not in url_to_result_map:
                        url_to_result_map[url] = []
                    url_to_result_map[url].append(result)
                    break  # Use first available URL
        
        # Extract content in batches
        batch_size = 10  # Process 10 URLs at a time
        content_map = {}
        
        for i in range(0, len(urls_to_extract), batch_size):
            batch_urls = urls_to_extract[i:i + batch_size]
            
            logger.info(f"Extracting content from batch {i//batch_size + 1} ({len(batch_urls)} URLs)")
            
            batch_content = await self.extract_article_content(batch_urls)
            content_map.update(batch_content)
            
            # Rate limiting between batches
            if i + batch_size < len(urls_to_extract):
                await asyncio.sleep(3)
        
        # Update search results with extracted content
        for url, content in content_map.items():
            if url in url_to_result_map:
                for result in url_to_result_map[url]:
                    result.full_text = content.get("text", "")
                    result.content_extracted = True
                    
                    # Update abstract if we got a better one
                    extracted_abstract = content.get("abstract", "")
                    if extracted_abstract and len(extracted_abstract) > len(result.abstract):
                        result.abstract = extracted_abstract
                    
                    # Determine extraction method
                    if url == result.pdf_link:
                        result.extraction_method = "pdf"
                    else:
                        result.extraction_method = "web_page"
        
        # Log extraction statistics
        total_extracted = sum(1 for r in search_results if r.content_extracted)
        logger.info(f"Successfully extracted content from {total_extracted}/{len(search_results)} papers")
        
        return search_results
    
    def save_results(self, results: List[SearchResult], compound_name: str, output_dir: str = "output"):
        """Save search results with extracted content to JSON file"""
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert results to dict format
        results_dict = {
            'compound': compound_name,
            'timestamp': datetime.now().isoformat(),
            'total_results': len(results),
            'papers_with_full_text': len([r for r in results if r.content_extracted]),
            'actors_used': {
                'scholar_search': self.scholar_actor_id,
                'content_extraction': self.extractor_actor_id
            },
            'results': [
                {
                    'title': r.title,
                    'authors': r.authors,
                    'abstract': r.abstract,
                    'link': r.link,
                    'pdf_link': r.pdf_link,
                    'citation_count': r.citation_count,
                    'year': r.year,
                    'search_term': r.search_term,
                    'full_text': r.full_text,
                    'content_extracted': r.content_extracted,
                    'extraction_method': r.extraction_method,
                    'text_length': len(r.full_text) if r.full_text else 0
                }
                for r in results
            ]
        }
        
        # Save to file
        filename = f"{output_dir}/{compound_name.replace(' ', '_')}_complete_results.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Complete results saved to {filename}")
    
    def get_summary_stats(self, results: List[SearchResult]) -> Dict:
        """Get comprehensive summary statistics for search results"""
        if not results:
            return {}
        
        years = [r.year for r in results if r.year > 0]
        citations = [r.citation_count for r in results]
        text_lengths = [len(r.full_text) for r in results if r.full_text]
        
        return {
            'total_papers': len(results),
            'papers_with_pdfs': len([r for r in results if r.pdf_link]),
            'papers_with_full_text': len([r for r in results if r.content_extracted]),
            'extraction_success_rate': len([r for r in results if r.content_extracted]) / len(results) * 100,
            'average_year': sum(years) / len(years) if years else 0,
            'total_citations': sum(citations),
            'most_cited': max(citations) if citations else 0,
            'search_terms_used': len(set(r.search_term for r in results)),
            'average_text_length': sum(text_lengths) / len(text_lengths) if text_lengths else 0,
            'extraction_methods': {
                'pdf': len([r for r in results if r.extraction_method == 'pdf']),
                'web_page': len([r for r in results if r.extraction_method == 'web_page']),
                'none': len([r for r in results if not r.content_extracted])
            },
            'actors_used': {
                'scholar_search': self.scholar_actor_id,
                'content_extraction': self.extractor_actor_id
            }
        }