"""
PubMed Scraper for Toxicology Research API
Uses PubMed E-utilities API for search and Cheerio for content extraction
Compatible with existing Google Scholar scraper workflow
"""

import aiohttp
import asyncio
import logging
import re
import json
import time
import urllib.parse
from typing import List, Dict, Optional, Set
from dataclasses import dataclass
from datetime import datetime

from apify_client import ApifyClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import CompoundInfo and SearchResult from shared classes (google_scholar_scraper)
from google_scholar_scraper import CompoundInfo, SearchResult, format_text_content, format_abstract

# PubMed E-utilities API endpoints
PUBMED_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
PUBMED_ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
PUBMED_ELINK_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi"

# Base URL for PubMed abstracts and articles
PUBMED_ABSTRACT_BASE = "https://pubmed.ncbi.nlm.nih.gov/"
PUBMED_ARTICLE_BASE = "https://www.ncbi.nlm.nih.gov/pmc/articles/"

class PubMedScraper:
    """
    PubMed scraper using E-utilities API and Cheerio
    Workflow:
    1. Search PubMed using E-utilities with CAS numbers and keywords
    2. Retrieve article metadata and links (both abstract and full paper)
    3. Extract content using Cheerio scraper
    """
    
    # Toxicology-specific keywords (shared with GoogleScholarScraper)
    TOXICOLOGY_KEYWORDS = [
        'toxicity',
        'inhalation',
        'gavage',
        'degradation',
        'pyrolysis',
        'vape',
        'cannabis'
    ]
    
    def __init__(self, apify_token: str, api_key: Optional[str] = None):
        """
        Initialize the PubMed scraper
        
        Args:
            apify_token (str): Apify API token for content scraping
            api_key (str, optional): NCBI API key for higher rate limits
        """
        self.client = ApifyClient(apify_token)
        self.api_key = api_key
        self.session = None
        
        # Cheerio actor for HTML extraction
        self.cheerio_actor_id = "apify/cheerio-scraper"
        
        logger.info("PubMedScraper initialized")
        logger.info(f"Cheerio scraper: {self.cheerio_actor_id}")
        
        # Rate limiting parameters (NCBI guidelines)
        self.requests_per_second = 3 if api_key else 1
        self.last_request_time = 0
    
    async def _create_session(self):
        """Create aiohttp session if not exists"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def _close_session(self):
        """Close aiohttp session"""
        if self.session and not self.session.closed:
            await self.session.close()
    
    async def _enforce_rate_limit(self):
        """Enforce NCBI API rate limits"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        # Calculate wait time to respect rate limit
        wait_time = max(0, (1 / self.requests_per_second) - time_since_last_request)
        
        if wait_time > 0:
            await asyncio.sleep(wait_time)
        
        self.last_request_time = time.time()
    
    def generate_search_queries(self, compound: CompoundInfo) -> List[str]:
        """Generate targeted search queries for PubMed"""
        queries = []
        
        # Get all identifiers (name, CAS, synonyms)
        identifiers = [compound.name]
        if compound.cas_number:
            identifiers.append(compound.cas_number)
        if compound.synonyms:
            identifiers.extend(compound.synonyms)
        
        # Generate queries: identifier + keyword combinations
        for identifier in identifiers:
            if not identifier:
                continue
                
            # Base query with the compound identifier
            queries.append(f'"{identifier}"')
            
            # Compound + each keyword
            for keyword in self.TOXICOLOGY_KEYWORDS:
                queries.append(f'"{identifier}" AND {keyword}')
        
        # Remove duplicates while preserving order
        seen = set()
        unique_queries = []
        for query in queries:
            if query not in seen:
                seen.add(query)
                unique_queries.append(query)
        
        logger.info(f"Generated {len(unique_queries)} PubMed search queries")
        return unique_queries[:5]  # Limit to 5 queries for performance (can be adjusted)
    
    async def search_pubmed(self, query: str, max_results: int = 20) -> List[str]:
        """
        Search PubMed using E-utilities
        
        Args:
            query (str): Search query
            max_results (int): Maximum results to return
            
        Returns:
            List[str]: List of PubMed IDs (PMIDs)
        """
        try:
            await self._create_session()
            await self._enforce_rate_limit()
            
            logger.info(f"Searching PubMed: {query}")
            
            # Prepare search parameters
            params = {
                "db": "pubmed",
                "term": query,
                "retmode": "json",
                "retmax": max_results,
                "sort": "relevance",
                "usehistory": "y"
            }
            
            if self.api_key:
                params["api_key"] = self.api_key
            
            # Execute search
            async with self.session.get(PUBMED_ESEARCH_URL, params=params) as response:
                if response.status != 200:
                    logger.error(f"PubMed search failed: {response.status}")
                    return []
                
                data = await response.json()
                
                # Extract PMIDs
                pmids = data.get("esearchresult", {}).get("idlist", [])
                
                if not pmids:
                    logger.warning(f"No results found for query: {query}")
                    return []
                
                logger.info(f"Found {len(pmids)} results for query: {query}")
                return pmids
                
        except Exception as e:
            logger.error(f"Error in PubMed search for query '{query}': {e}")
            return []
    
    async def get_article_details(self, pmids: List[str]) -> List[Dict]:
        """
        Retrieve article details using E-utilities ESummary
        
        Args:
            pmids (List[str]): List of PubMed IDs
            
        Returns:
            List[Dict]: List of article metadata
        """
        if not pmids:
            return []
        
        try:
            await self._create_session()
            await self._enforce_rate_limit()
            
            logger.info(f"Fetching details for {len(pmids)} articles")
            
            # Prepare request parameters
            params = {
                "db": "pubmed",
                "id": ",".join(pmids),
                "retmode": "json",
                "rettype": "abstract"
            }
            
            if self.api_key:
                params["api_key"] = self.api_key
            
            # Execute request
            async with self.session.get(PUBMED_ESUMMARY_URL, params=params) as response:
                if response.status != 200:
                    logger.error(f"PubMed ESummary failed: {response.status}")
                    return []
                
                data = await response.json()
                
                # Extract article details
                results = []
                summary_data = data.get("result", {})
                
                for pmid in pmids:
                    if pmid in summary_data:
                        article_data = summary_data[pmid]
                        
                        # Extract authors
                        authors = []
                        for author in article_data.get("authors", []):
                            if author.get("authtype") == "Author":
                                name = author.get("name", "")
                                if name:
                                    authors.append(name)
                        
                        # Create result entry
                        result = {
                            "pmid": pmid,
                            "title": article_data.get("title", "").strip(),
                            "pubdate": article_data.get("pubdate", ""),
                            "source": article_data.get("source", ""),
                            "authors": authors,
                            "doi": article_data.get("elocationid", "").replace("doi: ", ""),
                            "abstract_url": f"{PUBMED_ABSTRACT_BASE}{pmid}/",
                            "is_open_access": False,  # Will be updated later
                            "full_text_url": None,    # Will be updated later
                            "article_ids": article_data.get("articleids", [])
                        }
                        
                        # Extract year from pubdate
                        year_match = re.search(r'\d{4}', result["pubdate"])
                        if year_match:
                            result["year"] = int(year_match.group(0))
                        else:
                            result["year"] = 0
                        
                        results.append(result)
                
                return results
                
        except Exception as e:
            logger.error(f"Error fetching article details: {e}")
            return []
    
    async def check_full_text_availability(self, articles: List[Dict]) -> List[Dict]:
        """
        Check if full text is available through PubMed Central
        
        Args:
            articles (List[Dict]): List of article metadata
            
        Returns:
            List[Dict]: Updated article metadata with full text URLs
        """
        if not articles:
            return []
        
        try:
            await self._create_session()
            await self._enforce_rate_limit()
            
            # Get PMIDs
            pmids = [article["pmid"] for article in articles]
            
            # Prepare request parameters
            params = {
                "dbfrom": "pubmed",
                "db": "pmc",
                "id": ",".join(pmids),
                "retmode": "json",
                "cmd": "neighbor_links"
            }
            
            if self.api_key:
                params["api_key"] = self.api_key
            
            # Execute request
            async with self.session.get(PUBMED_ELINK_URL, params=params) as response:
                if response.status != 200:
                    logger.error(f"PubMed ELink failed: {response.status}")
                    return articles
                
                data = await response.json()
                
                # Process linksets to find PMC links
                linksets = data.get("linksets", [])
                pmid_to_pmcid = {}
                
                for linkset in linksets:
                    pmid = linkset.get("ids", [None])[0]
                    if not pmid:
                        continue
                    
                    # Look for PMC links
                    for linkgroup in linkset.get("linksetdbs", []):
                        if linkgroup.get("linkname") == "pubmed_pmc":
                            # Found a PMC link
                            if linkgroup.get("links", []):
                                pmcid = linkgroup["links"][0]
                                pmid_to_pmcid[pmid] = f"PMC{pmcid}"
                
                # Update articles with PMC information
                for article in articles:
                    pmid = article["pmid"]
                    if pmid in pmid_to_pmcid:
                        pmcid = pmid_to_pmcid[pmid]
                        article["is_open_access"] = True
                        article["pmcid"] = pmcid
                        article["full_text_url"] = f"{PUBMED_ARTICLE_BASE}{pmcid}/"
                    else:
                        # Also check if there's a PMC ID in the article_ids
                        for id_obj in article.get("article_ids", []):
                            if id_obj.get("idtype") == "pmc":
                                pmcid = id_obj.get("value", "")
                                if pmcid:
                                    article["is_open_access"] = True
                                    article["pmcid"] = pmcid
                                    article["full_text_url"] = f"{PUBMED_ARTICLE_BASE}{pmcid}/"
                                    break
                
                logger.info(f"Full text available for {sum(1 for a in articles if a.get('is_open_access'))} of {len(articles)} articles")
                return articles
                
        except Exception as e:
            logger.error(f"Error checking full text availability: {e}")
            return articles
    
    async def fetch_abstracts(self, pmids: List[str]) -> Dict[str, str]:
        """
        Fetch article abstracts using E-utilities EFetch
        
        Args:
            pmids (List[str]): List of PubMed IDs
            
        Returns:
            Dict[str, str]: Mapping of PMIDs to abstracts
        """
        if not pmids:
            return {}
        
        try:
            await self._create_session()
            await self._enforce_rate_limit()
            
            # Prepare request parameters
            params = {
                "db": "pubmed",
                "id": ",".join(pmids),
                "retmode": "xml",
                "rettype": "abstract"
            }
            
            if self.api_key:
                params["api_key"] = self.api_key
            
            # Execute request
            async with self.session.get(PUBMED_EFETCH_URL, params=params) as response:
                if response.status != 200:
                    logger.error(f"PubMed EFetch failed: {response.status}")
                    return {}
                
                xml_text = await response.text()
                
                # Use regex to extract abstracts (more reliable than XML parsing for this case)
                # Pattern to match PMID
                pmid_pattern = r'<PMID.*?>(\d+)</PMID>'
                # Pattern to match AbstractText
                abstract_pattern = r'<AbstractText.*?>(.*?)</AbstractText>'
                
                # Extract PMID and abstract pairs
                abstracts = {}
                current_pmid = None
                abstract_sections = []
                
                for line in xml_text.split('\n'):
                    # Check for PMID
                    pmid_match = re.search(pmid_pattern, line)
                    if pmid_match:
                        # If we were previously collecting an abstract, save it
                        if current_pmid and abstract_sections:
                            abstracts[current_pmid] = ' '.join(abstract_sections)
                            abstract_sections = []
                        
                        # Start tracking new PMID
                        current_pmid = pmid_match.group(1)
                    
                    # Check for AbstractText
                    abstract_match = re.search(abstract_pattern, line)
                    if abstract_match and current_pmid:
                        # Clean HTML entities and tags
                        abstract_text = re.sub(r'<.*?>', '', abstract_match.group(1))
                        abstract_text = re.sub(r'&lt;', '<', abstract_text)
                        abstract_text = re.sub(r'&gt;', '>', abstract_text)
                        abstract_text = re.sub(r'&amp;', '&', abstract_text)
                        abstract_text = re.sub(r'&quot;', '"', abstract_text)
                        abstract_text = re.sub(r'&apos;', "'", abstract_text)
                        
                        abstract_sections.append(abstract_text.strip())
                
                # Save the last abstract if there is one
                if current_pmid and abstract_sections:
                    abstracts[current_pmid] = ' '.join(abstract_sections)
                
                # Format abstracts
                for pmid, abstract in abstracts.items():
                    abstracts[pmid] = format_abstract(abstract)
                
                logger.info(f"Fetched {len(abstracts)} abstracts")
                return abstracts
                
        except Exception as e:
            logger.error(f"Error fetching abstracts: {e}")
            return {}
    
    async def _extract_content_with_cheerio(self, urls_to_extract: List[Dict]) -> Dict[str, Dict]:
        """
        Extract content from URLs using Cheerio scraper
        
        Args:
            urls_to_extract (List[Dict]): List of URLs to extract with metadata
            
        Returns:
            Dict[str, Dict]: Extracted content by URL
        """
        if not urls_to_extract:
            return {}
        
        try:
            logger.info(f"Extracting content with Cheerio for {len(urls_to_extract)} URLs")
            
            # Prepare URLs for Cheerio
            start_urls = [{"url": item["url"]} for item in urls_to_extract]
            
            # Configure page function based on URL type
            page_function = r"""
            async function pageFunction(context) {
                const { $, request } = context;
                const url = request.url;
                const isPMC = url.includes('/pmc/articles/');
                
                // Add random delay to avoid bot detection
                await new Promise(resolve => setTimeout(resolve, 1000 + Math.random() * 2000));
                
                // Extract title
                let title = $('title').text().trim();
                if (!title) {
                    title = $('.content-title, .article-title, h1.heading-title').first().text().trim();
                }
                
                // Different selectors based on URL type
                let abstract = '';
                let content = '';
                
                if (isPMC) {
                    // PMC full text article
                    abstract = $('.abstract, .sec-abstract').text().trim();
                    
                    // Extract the full content
                    $('.article-body, .jig-ncbiinpagenav').find('.ref-list, .back, .fn-group').remove();
                    content = $('.article-body, .jig-ncbiinpagenav').text().trim();
                    
                    if (!content) {
                        // Alternative selectors for PMC
                        content = $('#maincontent').find('.ref-list, .back').remove().end().text().trim();
                    }
                } else {
                    // PubMed abstract page
                    abstract = $('#abstract, .abstract-content').text().trim();
                    content = abstract; // For abstract pages, content is the same as abstract
                }
                
                // Clean up the text
                const cleanText = (text) => {
                    return text
                        .replace(/\\s+/g, ' ')
                        .replace(/This article has been cited by other articles in PMC.*/i, '')
                        .replace(/Download citation.*/i, '')
                        .replace(/Similar articles.*/i, '')
                        .trim();
                };
                
                return {
                    url: request.url,
                    title: title,
                    abstract: cleanText(abstract),
                    content: cleanText(content),
                    success: !!(abstract || content),
                    isPMC: isPMC
                };
            }
            """
            
            # Run the Cheerio scraper
            run_input = {
                "startUrls": start_urls,
                "pageFunction": page_function,
                "proxyConfiguration": {
                    "useApifyProxy": True,
                    "apifyProxyGroups": ["RESIDENTIAL"]
                },
                "maxRequestRetries": 2,
                "maxConcurrency": 5
            }
            
            run = self.client.actor(self.cheerio_actor_id).call(run_input=run_input)
            
            # Process the results
            content_map = {}
            for item in self.client.dataset(run["defaultDatasetId"]).iterate_items():
                url = item.get("url")
                
                if url and item.get("success"):
                    # Find the original URL metadata
                    original_item = next((x for x in urls_to_extract if x["url"] == url), None)
                    
                    if original_item:
                        # Format the content
                        content = format_text_content(item.get("content", ""))
                        abstract = format_abstract(item.get("abstract", ""))
                        
                        content_map[url] = {
                            "pmid": original_item.get("pmid"),
                            "title": item.get("title") or original_item.get("title", ""),
                            "text": content,
                            "abstract": abstract,
                            "extracted": True,
                            "length": len(content),
                            "scraper": "cheerio",
                            "is_pmc": item.get("isPMC", False)
                        }
                        
                        logger.info(f"Extracted content from {url} (length: {len(content)})")
            
            logger.info(f"Extracted content for {len(content_map)} of {len(urls_to_extract)} URLs")
            return content_map
            
        except Exception as e:
            logger.error(f"Error extracting content with Cheerio: {e}")
            return {}
    
    async def search_compound(self, compound: CompoundInfo, max_results_per_query: int = 15) -> List[SearchResult]:
        """
        Complete workflow: Search PubMed and extract content
        
        Args:
            compound (CompoundInfo): Compound to search for
            max_results_per_query (int): Maximum results per query
            
        Returns:
            List[SearchResult]: Search results with extracted content
        """
        try:
            logger.info(f"Starting PubMed search for compound: {compound.name} (CAS: {compound.cas_number})")
            
            # Step 1: Generate search queries
            queries = self.generate_search_queries(compound)
            
            # Step 2: Search PubMed for each query
            all_pmids = set()
            for query in queries:
                pmids = await self.search_pubmed(query, max_results_per_query)
                all_pmids.update(pmids)
                
                # Enforce rate limit between queries
                await asyncio.sleep(1)
            
            logger.info(f"Found {len(all_pmids)} unique articles across all queries")
            
            if not all_pmids:
                logger.warning("No articles found in PubMed")
                return []
            
            # Step 3: Get article details
            pmid_list = list(all_pmids)
            articles = await self.get_article_details(pmid_list)
            
            if not articles:
                logger.warning("Failed to retrieve article details")
                return []
            
            # Step 4: Check full text availability
            articles = await self.check_full_text_availability(articles)
            
            # Step 5: Fetch abstracts
            abstracts = await self.fetch_abstracts(pmid_list)
            
            # Step 6: Prepare URLs for content extraction
            urls_to_extract = []
            
            for article in articles:
                # Prioritize full text URLs when available
                if article.get("is_open_access") and article.get("full_text_url"):
                    urls_to_extract.append({
                        "url": article["full_text_url"],
                        "pmid": article["pmid"],
                        "title": article["title"],
                        "is_full_text": True
                    })
                else:
                    # Otherwise use abstract URL
                    urls_to_extract.append({
                        "url": article["abstract_url"],
                        "pmid": article["pmid"],
                        "title": article["title"],
                        "is_full_text": False
                    })
            
            # Step 7: Extract content with Cheerio
            content_map = await self._extract_content_with_cheerio(urls_to_extract)
            
            # Step 8: Build search results
            search_results = []
            
            for article in articles:
                pmid = article["pmid"]
                
                # Get URL that was actually extracted
                url_extracted = None
                for url_type in ["full_text_url", "abstract_url"]:
                    if article.get(url_type) and article[url_type] in content_map:
                        url_extracted = article[url_type]
                        break
                
                # Get content information
                content_info = content_map.get(url_extracted, {}) if url_extracted else {}
                
                # Use extracted content or fallback to API abstract
                abstract = content_info.get("abstract", "") or abstracts.get(pmid, "")
                full_text = content_info.get("text", "")
                
                # Create search result
                search_result = SearchResult(
                    title=article["title"],
                    authors=article.get("authors", []),
                    abstract=abstract,
                    link=article["abstract_url"],  # Always include abstract link
                    pdf_link=None,  # PubMed doesn't provide direct PDF links
                    citation_count=0,  # Citation count not available from PubMed API
                    year=article.get("year", 0),
                    search_term=compound.name,  # Use compound name as search term
                    full_text=full_text,
                    content_extracted=bool(full_text),
                    extraction_method="pubmed_api" if not full_text and abstract else "cheerio_html",
                    extraction_scraper="pubmed_api" if not full_text and abstract else "cheerio"
                )
                
                search_results.append(search_result)
            
            # Close the session
            await self._close_session()
            
            logger.info(f"PubMed search complete! Found {len(search_results)} papers")
            logger.info(f"  With full text: {sum(1 for r in search_results if r.full_text)}")
            logger.info(f"  With abstracts only: {sum(1 for r in search_results if not r.full_text and r.abstract)}")
            
            return search_results
            
        except Exception as e:
            logger.error(f"Error in PubMed search: {e}")
            await self._close_session()
            return []
    
    def get_summary_stats(self, results: List[SearchResult]) -> Dict:
        """Get summary statistics for PubMed search results"""
        if not results:
            return {}
        
        # Extract statistics
        years = [r.year for r in results if r.year > 0]
        text_lengths = [len(r.full_text) for r in results if r.full_text]
        abstract_lengths = [len(r.abstract) for r in results if r.abstract]
        
        # Count extraction methods
        extraction_methods = {}
        for r in results:
            if r.extraction_method:
                extraction_methods[r.extraction_method] = extraction_methods.get(r.extraction_method, 0) + 1
        
        return {
            "total_papers": len(results),
            "papers_with_full_text": sum(1 for r in results if r.full_text),
            "papers_with_abstract_only": sum(1 for r in results if not r.full_text and r.abstract),
            "extraction_success_rate": (
                sum(1 for r in results if r.content_extracted) / len(results) * 100
                if results else 0
            ),
            "average_year": sum(years) / len(years) if years else 0,
            "year_range": [min(years), max(years)] if years else [0, 0],
            "average_text_length": sum(text_lengths) / len(text_lengths) if text_lengths else 0,
            "average_abstract_length": sum(abstract_lengths) / len(abstract_lengths) if abstract_lengths else 0,
            "extraction_methods": extraction_methods,
            "source": "PubMed"
        }

# Example usage
async def main():
    """Example usage of the PubMed scraper"""
    # Initialize with your Apify token
    scraper = PubMedScraper(apify_token="your_apify_token_here")
    
    # Define compound to search
    compound = CompoundInfo(
        name="Benzene",
        cas_number="71-43-2",
        synonyms=["benzol", "phenyl hydride"]
    )
    
    # Run the search and extraction pipeline
    results = await scraper.search_compound(compound, max_results_per_query=10)
    
    # Get summary statistics
    stats = scraper.get_summary_stats(results)
    print(json.dumps(stats, indent=2))

if __name__ == "__main__":
    asyncio.run(main())
