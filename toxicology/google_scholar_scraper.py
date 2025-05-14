"""
Enhanced Google Scholar Scraper with Cheerio/PDF Scrapers
Uses Marco Gullo for Google Scholar search + jirimoravcik PDF scraper + Cheerio for content extraction
Fixed issue with full_text being incorrectly carried over from previous results
"""

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import List, Dict, Optional, Set
import json
from datetime import datetime
import os
from apify_client import ApifyClient
import random

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
    full_text: Optional[str] = None
    content_extracted: bool = False
    extraction_method: Optional[str] = None
    extraction_scraper: Optional[str] = None
    
    def get_text_length(self) -> int:
        """Get length of full text"""
        return len(self.full_text) if self.full_text else 0

def format_text_content(text: str) -> str:
    """Format and clean text content for better readability"""
    if not text:
        return ""
    
    # Remove common UI/navigation elements
    text = re.sub(r'(Download PDF|View PDF|Subscribe|Login|Register)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Copyright.*?\d{4}', '', text, flags=re.IGNORECASE)
    text = re.sub(r'ISSN.*?\d{4}-\d{4}', '', text, flags=re.IGNORECASE)
    text = re.sub(r'DOI.*?10\.\d{4}\/[^\s]+', '', text, flags=re.IGNORECASE)
    
    # Clean up excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    # Add proper spacing after periods if missing
    text = re.sub(r'\.([A-Z])', r'. \1', text)
    
    return text.strip()

def format_abstract(abstract: str) -> str:
    """Format abstract text for better readability"""
    if not abstract:
        return ""
    
    # Clean the abstract
    abstract = format_text_content(abstract)
    
    # Remove common prefixes
    abstract = re.sub(r'^(Abstract\s*:?\s*|Summary\s*:?\s*)', '', abstract, flags=re.IGNORECASE)
    
    # Ensure minimum length for quality
    if len(abstract) < 50:
        return ""
    
    # Add structure for better readability
    structure_patterns = [
        (r'(Background[:.])', r'\n\1'),
        (r'(Objective[:.])', r'\n\1'),
        (r'(Methods?[:.])', r'\n\1'),
        (r'(Results?[:.])', r'\n\1'),
        (r'(Conclusions?[:.])', r'\n\1'),
        (r'(Purpose[:.])', r'\n\1')
    ]
    
    for pattern, replacement in structure_patterns:
        abstract = re.sub(pattern, replacement, abstract, flags=re.IGNORECASE)
    
    return abstract.strip()

class GoogleScholarScraper:
    """
    Enhanced Google Scholar scraper
    Phase 1: Marco Gullo's scraper for Google Scholar search
    Phase 2: jirimoravcik's PDF scraper for PDF content extraction (prioritized)
    Phase 3: Cheerio scraper for HTML content extraction (fallback)
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
        
        # Phase 1: Google Scholar search
        self.scholar_actor_id = "marco.gullo/google-scholar-scraper"
        
        # Phase 2 & 3: Content extraction (Puppeteer removed)
        self.pdf_actor_id = "jirimoravcik/pdf-text-extractor"
        self.cheerio_actor_id = "apify/cheerio-scraper"
        
        logger.info("GoogleScholarScraper initialized")
        logger.info(f"Scholar search: {self.scholar_actor_id}")
        logger.info(f"PDF scraper: {self.pdf_actor_id}")
        logger.info(f"Cheerio scraper: {self.cheerio_actor_id}")
    
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
        return unique_queries[:5]  # Limit to 5 queries for better performance
    
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
    
    def _is_pdf_url(self, url: str) -> bool:
        """
        Check if URL points to a PDF file
        
        Args:
            url (str): URL to check
            
        Returns:
            bool: True if URL points to PDF
        """
        if not url:
            return False
        
        url_lower = url.lower()
        return (url_lower.endswith('.pdf') or 
                'filetype=pdf' in url_lower or
                '/pdf/' in url_lower or
                'application/pdf' in url_lower)
    
    async def _extract_pdf_content(self, pdf_urls: List[str]) -> Dict[str, Dict]:
        """
        Extract content from PDF URLs using jirimoravcik's PDF scraper
        
        Args:
            pdf_urls (List[str]): URLs of PDFs to extract
            
        Returns:
            Dict[str, Dict]: Extracted content by URL
        """
        try:
            logger.info(f"Extracting PDF content for {len(pdf_urls)} URLs")
            
            # Configure PDF scraper with more comprehensive settings
            run_input = {
                "urls": pdf_urls,
                "userAgent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "headless": True,
                "screenshot": False,
                "useProxy": True,
                "timeout": 30,
                "extractText": True,
                "extractFormat": "text",
                "waitUntil": "networkidle",
                "maxRetries": 2,
                "retryDelay": 1000,
                "ignoreHTTPSErrors": True
            }
            
            # Run the PDF scraper
            logger.info(f"Starting PDF scraper with input: {run_input}")
            run = self.client.actor(self.pdf_actor_id).call(run_input=run_input)
            
            # Process results
            content_map = {}
            item_count = 0
            
            for item in self.client.dataset(run["defaultDatasetId"]).iterate_items():
                item_count += 1
                url = item.get("url")
                text = item.get("text", "")
                status = item.get("status", "")
                error = item.get("error", "")
                
                logger.info(f"Processing PDF item {item_count}: {url}")
                logger.info(f"  Status: {status}")
                logger.info(f"  Text length: {len(text)}")
                
                if error:
                    logger.warning(f"  Error: {error}")
                
                if url and text:
                    # Format the PDF text
                    formatted_text = format_text_content(text)
                    
                    # Extract abstract from PDF text (try multiple patterns)
                    abstract = ""
                    abstract_patterns = [
                        r'abstract\s*:?\s*(.*?)(?=\n\s*[1-9]\.|introduction|keywords|references)',
                        r'summary\s*:?\s*(.*?)(?=\n\s*[1-9]\.|introduction|keywords|references)',
                        r'\nabstract\n(.*?)(?=\nintroduction|\nkeywords|\nreferences)',
                        r'^abstract[:\s]+(.*?)(?=^introduction|^keywords|^references|^\d+\.)',
                    ]
                    
                    for pattern in abstract_patterns:
                        abstract_match = re.search(pattern, text, re.IGNORECASE | re.DOTALL | re.MULTILINE)
                        if abstract_match:
                            abstract = format_abstract(abstract_match.group(1).strip())
                            break
                    
                    content_map[url] = {
                        "title": item.get("title", ""),
                        "text": formatted_text,
                        "abstract": abstract,
                        "extracted": True,
                        "length": len(formatted_text),
                        "scraper": "pdf_scraper",
                        "isPdf": True,
                        "status": status
                    }
                    logger.info(f"PDF successfully extracted: {url} (length: {len(formatted_text)})")
                else:
                    if url:
                        logger.warning(f"PDF extraction failed for {url}: No text extracted")
                        # Still add to map but mark as failed
                        content_map[url] = {
                            "title": "",
                            "text": "",
                            "abstract": "",
                            "extracted": False,
                            "length": 0,
                            "scraper": "pdf_scraper",
                            "isPdf": True,
                            "status": status,
                            "error": error
                        }
            
            logger.info(f"PDF extraction completed. Processed {item_count} items, {len(content_map)} URLs mapped")
            return content_map
            
        except Exception as e:
            logger.error(f"Error with PDF scraper: {e}")
            return {}
    
    async def _extract_with_cheerio(self, urls: List[str]) -> Dict[str, Dict]:
        """
        Extract content using Cheerio scraper with improved formatting
        
        Args:
            urls (List[str]): URLs to scrape
            
        Returns:
            Dict[str, Dict]: Extracted content by URL
        """
        try:
            logger.info(f"Extracting with Cheerio scraper for {len(urls)} URLs")
            
            # Configure Cheerio scraper with enhanced text formatting
            run_input = {
                "startUrls": [{"url": url} for url in urls],
                "pageFunction": r"""
                    async function pageFunction(context) {
                        const { $, request } = context;
                        
                        // Random delay to avoid being detected as bot
                        await new Promise(resolve => setTimeout(resolve, 1000 + Math.random() * 2000));
                        
                        // Extract title
                        const title = $('title, h1, .article-title').first().text().trim();
                        
                        // Extract abstract with improved selectors
                        const abstractSelectors = [
                            '.abstract', '#abstract', '.article-abstract',
                            '.summary', '.paper-abstract', '.abstracts',
                            '.entry-summary', '.description'
                        ];
                        let abstract = '';
                        for (const selector of abstractSelectors) {
                            const element = $(selector);
                            if (element.length) {
                                abstract = element.text().trim();
                                // Clean common prefixes
                                abstract = abstract.replace(/^(Abstract\s*:?\s*|Summary\s*:?\s*)/i, '');
                                if (abstract && abstract.length > 50) break;
                            }
                        }
                        
                        // Extract main content with better selection
                        const contentSelectors = [
                            'article', '.article', '.content', 'main', '.main-content',
                            '.paper-content', '.full-text', '.article-body', '.text'
                        ];
                        let content = '';
                        let maxLength = 0;
                        
                        for (const selector of contentSelectors) {
                            const element = $(selector);
                            if (element.length) {
                                // Remove unwanted elements before getting text
                                element.find('nav, .navigation, .sidebar, .ads').remove();
                                element.find('.references, .citations').remove();
                                element.find('script, style').remove();
                                
                                const text = element.text().trim();
                                if (text && text.length > maxLength) {
                                    content = text;
                                    maxLength = text.length;
                                }
                            }
                        }
                        
                        // Clean and format content
                        if (content) {
                            // Remove common UI elements
                            content = content.replace(/Download PDF|View PDF|Subscribe|Login|Register/gi, '');
                            content = content.replace(/Copyright.*?\d{4}/gi, '');
                            content = content.replace(/ISSN.*?\d{4}-\d{4}/gi, '');
                            // Clean excessive whitespace
                            content = content.replace(/\s+/g, ' ').trim();
                        }
                        
                        return {
                            url: request.url,
                            title,
                            abstract,
                            content: content || abstract,
                            success: !!(content || abstract),
                            text_length: (content || abstract).length
                        };
                    }
                """,
                "proxyUrls": ["http://groups-RESIDENTIAL:password@proxy.apify.com:8000"],
                "timeout": 30
            }
            
            # Run the scraper
            run = self.client.actor(self.cheerio_actor_id).call(run_input=run_input)
            
            # Process results with formatting
            content_map = {}
            for item in self.client.dataset(run["defaultDatasetId"]).iterate_items():
                url = item.get("url")
                if url and item.get("success"):
                    # Format the extracted content
                    content = item.get("content", "")
                    abstract = item.get("abstract", "")
                    
                    formatted_content = format_text_content(content)
                    formatted_abstract = format_abstract(abstract)
                    
                    content_map[url] = {
                        "title": item.get("title", ""),
                        "text": formatted_content,
                        "abstract": formatted_abstract,
                        "extracted": True,
                        "length": len(formatted_content),
                        "scraper": "cheerio"
                    }
                    logger.info(f"Cheerio extracted: {url} (length: {len(formatted_content)})")
            
            return content_map
            
        except Exception as e:
            logger.error(f"Error with Cheerio scraper: {e}")
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
        logger.info(f"Starting search for compound: {compound.name} (CAS: {compound.cas_number})")
        
        # Phase 1: Search Google Scholar using Marco Gullo's scraper
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
        
        # Phase 2 & 3: Extract content from found URLs
        logger.info("Phase 2: Extracting content from found URLs...")
        search_results = await self._extract_content_from_results(search_results)
        
        logger.info(f"Pipeline complete! Found {len(search_results)} unique papers")
        return search_results
    
    def _process_results(self, raw_results: List[Dict]) -> List[SearchResult]:
        """Process and deduplicate search results with improved duplicate detection"""
        seen_titles = set()
        processed_results = []
        
        for result in raw_results:
            try:
                # Extract data with fallbacks for Marco's format
                title = result.get('title', '') or result.get('Title', '')
                title = title.strip()
                
                # Improved duplicate detection: normalize title
                normalized_title = re.sub(r'[^\w\s]', '', title.lower()).strip()
                
                # Skip if we've seen this title or if title is empty
                if not title or not normalized_title:
                    continue
                
                # Check for similar titles (not just exact matches)
                is_duplicate = False
                for seen_title in seen_titles:
                    # Simple similarity check
                    words1 = set(normalized_title.split())
                    words2 = set(seen_title.split())
                    if words1 and words2:
                        intersection = len(words1.intersection(words2))
                        union = len(words1.union(words2))
                        similarity = intersection / union
                        if similarity > 0.8:  # 80% similarity threshold
                            is_duplicate = True
                            break
                
                if is_duplicate:
                    continue
                
                seen_titles.add(normalized_title)
                
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
                
                # Create SearchResult object with formatted abstract
                abstract = result.get('abstract', '') or result.get('Abstract', '')
                formatted_abstract = format_abstract(abstract)
                
                search_result = SearchResult(
                    title=title,
                    authors=authors,
                    abstract=formatted_abstract,
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
        Extract full content from search results prioritizing PDFs
        
        Args:
            search_results (List[SearchResult]): Search results with links
            
        Returns:
            List[SearchResult]: Search results with extracted content
        """
        # Collect URLs and prioritize PDFs based on link extension
        pdf_urls = []
        html_urls = []
        url_to_results = {}
        
        for result in search_results:
            # Check if the link has a PDF extension
            if result.link:
                if self._is_pdf_url(result.link):
                    # This is a PDF link
                    if result.link not in url_to_results:
                        pdf_urls.append(result.link)
                        url_to_results[result.link] = []
                    url_to_results[result.link].append(result)
                else:
                    # This is a regular HTML link
                    if result.link not in url_to_results:
                        html_urls.append(result.link)
                        url_to_results[result.link] = []
                    url_to_results[result.link].append(result)
        
        logger.info(f"URLs to extract: {len(pdf_urls)} PDFs, {len(html_urls)} HTML pages")
        
        # Extract PDFs first
        if pdf_urls:
            pdf_batch_size = 10  # Adjust batch size for PDFs
            for i in range(0, len(pdf_urls), pdf_batch_size):
                batch = pdf_urls[i:i + pdf_batch_size]
                logger.info(f"PDF batch {i//pdf_batch_size + 1}: {len(batch)} URLs")
                
                batch_content = await self._extract_pdf_content(batch)
                
                # Update results with PDF content
                for url, content in batch_content.items():
                    if url in url_to_results:
                        for result in url_to_results[url]:
                            # FIXED: Check if content was actually extracted successfully
                            if content.get("extracted", False) and content.get("text", ""):
                                result.full_text = content.get("text", "")
                                result.content_extracted = True
                                result.extraction_scraper = content.get("scraper", "pdf_scraper")
                                result.extraction_method = "pdf_scraper"
                                
                                # Update abstract if PDF provided a better one
                                pdf_abstract = content.get("abstract", "")
                                if pdf_abstract and len(pdf_abstract) > len(result.abstract):
                                    result.abstract = pdf_abstract
                            else:
                                # FIXED: If extraction failed, ensure full_text is empty
                                result.full_text = ""
                                result.content_extracted = False
                                result.extraction_scraper = content.get("scraper", "pdf_scraper")
                                result.extraction_method = None
                
                if i + pdf_batch_size < len(pdf_urls):
                    await asyncio.sleep(3)  # Rate limiting
        
        # Extract from HTML pages only for results without content
        html_urls_to_extract = [url for url in html_urls 
                               if any(not r.content_extracted for r in url_to_results[url])]
        
        if html_urls_to_extract:
            cheerio_batch_size = 20
            for i in range(0, len(html_urls_to_extract), cheerio_batch_size):
                batch = html_urls_to_extract[i:i + cheerio_batch_size]
                logger.info(f"HTML batch {i//cheerio_batch_size + 1}: {len(batch)} URLs")
                
                batch_content = await self._extract_with_cheerio(batch)
                
                # Update results with HTML content (only if no PDF was extracted)
                for url, content in batch_content.items():
                    if url in url_to_results:
                        for result in url_to_results[url]:
                            if not result.content_extracted:  # Only update if no PDF content
                                # FIXED: Check if content was actually extracted
                                if content.get("extracted", False) and content.get("text", ""):
                                    result.full_text = content.get("text", "")
                                    result.content_extracted = True
                                    result.extraction_scraper = content.get("scraper", "cheerio")
                                    result.extraction_method = "cheerio_html"
                                    
                                    # Update abstract if HTML provided a better one
                                    html_abstract = content.get("abstract", "")
                                    if html_abstract and len(html_abstract) > len(result.abstract):
                                        result.abstract = html_abstract
                                else:
                                    # FIXED: If extraction failed, ensure full_text is empty
                                    result.full_text = ""
                                    result.content_extracted = False
                                    result.extraction_scraper = content.get("scraper", "cheerio")
                                    result.extraction_method = None
                
                if i + cheerio_batch_size < len(html_urls_to_extract):
                    await asyncio.sleep(3)  # Rate limiting
        
        # Log extraction statistics
        total_extracted = sum(1 for r in search_results if r.content_extracted)
        pdf_extracted = sum(1 for r in search_results if r.extraction_method == "pdf_scraper")
        html_extracted = sum(1 for r in search_results if r.extraction_method == "cheerio_html")
        
        logger.info(f"Extraction Results:")
        logger.info(f"  Successfully extracted: {total_extracted}/{len(search_results)} papers")
        logger.info(f"  Success rate: {(total_extracted/len(search_results)*100) if search_results else 0:.1f}%")
        logger.info(f"  PDF extractions: {pdf_extracted}")
        logger.info(f"  HTML extractions: {html_extracted}")
        
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
            'pdf_extractions': len([r for r in results if r.extraction_method == "pdf_scraper"]),
            'html_extractions': len([r for r in results if r.extraction_method == "cheerio_html"]),
            'actors_used': {
                'scholar_search': self.scholar_actor_id,
                'pdf_scraper': self.pdf_actor_id,  # Added PDF scraper
                'cheerio_scraper': self.cheerio_actor_id
                # Removed puppeteer_scraper
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
                    'extraction_scraper': r.extraction_scraper,
                    'text_length': r.get_text_length()
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

        # Extract relevant data safely
        years = [r.year for r in results if isinstance(r.year, int) and r.year > 0]
        citations = [r.citation_count for r in results if isinstance(r.citation_count, int)]
        text_lengths = [r.get_text_length() for r in results if r.content_extracted]

        # Count extraction methods and scrapers
        extraction_methods = {}
        extraction_scrapers = {}
        for r in results:
            if r.extraction_method:
                extraction_methods[r.extraction_method] = extraction_methods.get(r.extraction_method, 0) + 1
            if r.extraction_scraper:
                extraction_scrapers[r.extraction_scraper] = extraction_scrapers.get(r.extraction_scraper, 0) + 1

        return {
            'total_papers': len(results),
            'papers_with_pdfs': len([r for r in results if r.pdf_link]),
            'papers_with_full_text': len([r for r in results if r.content_extracted]),
            'extraction_success_rate': (
                len([r for r in results if r.content_extracted]) / len(results) * 100
                if results else 0
            ),
            'pdf_extractions': len([r for r in results if r.extraction_method == "pdf_scraper"]),
            'html_extractions': len([r for r in results if r.extraction_method == "cheerio_html"]),
            'average_year': sum(years) / len(years) if years else 0,
            'total_citations': sum(citations),
            'most_cited': max(citations) if citations else 0,
            'search_terms_used': len(set(r.search_term for r in results)),
            'average_text_length': sum(text_lengths) / len(text_lengths) if text_lengths else 0,
            'extraction_methods': extraction_methods,
            'extraction_scrapers': extraction_scrapers,
            'actors_used': {
                'scholar_search': self.scholar_actor_id,
                'pdf_scraper': self.pdf_actor_id,  # Added PDF scraper
                'cheerio_scraper': self.cheerio_actor_id
                # Removed puppeteer_scraper
            }
        }

# Example usage
async def main():
    """Example usage of the enhanced scraper"""
    # Initialize with your Apify token
    scraper = GoogleScholarScraper(apify_token="your_apify_token_here")
    
    # Define compound to search
    compound = CompoundInfo(
        name="Benzene",
        cas_number="71-43-2",
        synonyms=["benzol", "phenyl hydride"]
    )
    
    # Run the search and extraction pipeline
    results = await scraper.search_compound(compound, max_results_per_query=10)
    
    # Save results
    scraper.save_results(results, compound.name)
    
    # Get summary statistics
    stats = scraper.get_summary_stats(results)
    print(json.dumps(stats, indent=2))

if __name__ == "__main__":
    asyncio.run(main())
