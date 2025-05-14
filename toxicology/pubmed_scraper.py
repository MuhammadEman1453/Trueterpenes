"""
Europe PMC Scraper for Toxicology Research API
Uses Europe PMC API for search and content extraction
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
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import CompoundInfo and SearchResult from shared classes
from google_scholar_scraper import CompoundInfo, SearchResult, format_text_content, format_abstract

# Europe PMC API endpoints
EUROPE_PMC_API_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
EUROPE_PMC_ARTICLE_URL = "https://europepmc.org/article/"
EUROPE_PMC_FULLTEXT_XML_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest"

class EuropePMCScraper:
    """
    Europe PMC scraper using REST API
    Workflow:
    1. Search Europe PMC using REST API (with abstracts included)
    2. Extract DOI, PMC links, and check full text availability
    3. Get full text content via XML API when available
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
        Initialize the Europe PMC scraper
        
        Args:
            apify_token (str): Apify API token (kept for interface compatibility)
        """
        self.session = None
        
        logger.info("EuropePMCScraper initialized")
        logger.info("Using Europe PMC REST API for all content extraction")
    
    async def _create_session(self):
        """Create aiohttp session if not exists"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def _close_session(self):
        """Close aiohttp session"""
        if self.session and not self.session.closed:
            await self.session.close()
    
    def generate_search_queries(self, compound: CompoundInfo) -> List[str]:
        """Generate targeted search queries for Europe PMC"""
        queries = []
        
        # Get all identifiers
        identifiers = [compound.name]
        if compound.cas_number:
            identifiers.append(compound.cas_number)
        if compound.synonyms:
            identifiers.extend(compound.synonyms[:3])
        
        # Generate queries
        for identifier in identifiers:
            if not identifier:
                continue
                
            # Base query
            queries.append(f'"{identifier}"')
            
            # Compound + keyword combinations
            for keyword in self.TOXICOLOGY_KEYWORDS:
                queries.append(f'"{identifier}" AND {keyword}')
        
        # Remove duplicates
        seen = set()
        unique_queries = []
        for query in queries:
            if query not in seen:
                seen.add(query)
                unique_queries.append(query)
        
        logger.info(f"Generated {len(unique_queries)} Europe PMC search queries")
        return unique_queries[:8]
    
    async def search_europe_pmc(self, query: str, max_results: int = 25) -> List[Dict]:
        """
        Search Europe PMC using REST API
        
        Args:
            query (str): Search query
            max_results (int): Maximum results to return
            
        Returns:
            List[Dict]: List of article data with abstracts
        """
        try:
            await self._create_session()
            
            logger.info(f"Searching Europe PMC: {query}")
            
            # Prepare search parameters - use core for full metadata
            params = {
                "query": query,
                "resultType": "core",
                "pageSize": max_results,
                "format": "json",
                "cursorMark": "*"
            }
            
            # Execute search
            async with self.session.get(EUROPE_PMC_API_URL, params=params) as response:
                if response.status != 200:
                    logger.error(f"Europe PMC search failed: {response.status}")
                    return []
                
                data = await response.json()
                
                # Extract results
                results = data.get("resultList", {}).get("result", [])
                
                if not results:
                    logger.warning(f"No results found for query: {query}")
                    return []
                
                logger.info(f"Found {len(results)} results for query: {query}")
                
                # Process results
                processed_results = []
                for article in results:
                    # Debug: Let's see what fields are actually in the response
                    if len(processed_results) == 0:  # Log first article only
                        logger.debug(f"Sample article keys: {list(article.keys())[:20]}")
                    
                    # Extract identifiers
                    doi = article.get("doi")
                    pmid = article.get("pmid")
                    pmcid = article.get("pmcId")  # Try different field name
                    
                    # Check full text availability - let's check what fields actually exist
                    has_full_text = article.get("hasFullText") == "Y" or article.get("fullTextAvailable") == "Y"
                    has_pdf = article.get("hasPDF") == "Y"
                    is_open_access = article.get("isOpenAccess") == "Y"
                    
                    # Try to get PMC ID from different places
                    if not pmcid:
                        # Check if it's in a different field
                        pmc_id_fields = ["pmcId", "pmcid", "PMCID", "pmc"]
                        for field in pmc_id_fields:
                            if article.get(field):
                                pmcid = article.get(field)
                                break
                    
                    # Get the source (MED, PMC, etc.)
                    source = article.get("source", "")
                    
                    # If source is PMC, we might already have full text access
                    if source == "PMC" and pmid:
                        pmcid = f"PMC{pmid}" if not pmcid else pmcid
                    
                    # Debug logging
                    if len(processed_results) < 3:  # Log first few articles
                        logger.debug(f"Article {pmid}: source={source}, pmcid={pmcid}, hasFullText={has_full_text}, hasPDF={has_pdf}, isOpenAccess={is_open_access}")
                    
                    # Get URLs
                    abstract_url = f"https://europepmc.org/article/{source}/{pmid}" if pmid else None
                    
                    # Determine if we can get full text
                    can_get_full_text = (is_open_access and pmcid) or source == "PMC"
                    
                    # Get abstract directly from API
                    abstract = article.get("abstractText", "")
                    
                    processed_results.append({
                        "title": article.get("title", ""),
                        "authors": self._extract_authors(article),
                        "abstract": abstract,
                        "doi": doi,
                        "pmid": pmid,
                        "pmcid": pmcid,
                        "year": article.get("pubYear"),
                        "journal": article.get("journalTitle", ""),
                        "hasFullText": has_full_text,
                        "hasPDF": has_pdf,
                        "isOpenAccess": is_open_access,
                        "can_get_full_text": can_get_full_text,
                        "abstract_url": abstract_url,
                        "source": source
                    })
                
                # Log summary of what we found
                open_access_count = sum(1 for r in processed_results if r["isOpenAccess"])
                pmc_count = sum(1 for r in processed_results if r["source"] == "PMC")
                pmcid_count = sum(1 for r in processed_results if r["pmcid"])
                
                logger.info(f"Search summary - Open access: {open_access_count}, PMC source: {pmc_count}, Has PMC ID: {pmcid_count}")
                
                return processed_results
                
        except Exception as e:
            logger.error(f"Error in Europe PMC search for query '{query}': {e}")
            return []
    
    def _extract_authors(self, article: Dict) -> List[str]:
        """Extract author names from article data"""
        authors = []
        
        # Try authorString first
        author_string = article.get("authorString", "")
        if author_string:
            authors = [name.strip() for name in author_string.split(",")]
            return authors[:10]
        
        # Fallback to authorList
        author_list = article.get("authorList", {}).get("author", [])
        for author in author_list:
            full_name = author.get("fullName", "")
            if not full_name:
                first_name = author.get("firstName", "")
                last_name = author.get("lastName", "")
                if last_name:
                    full_name = f"{first_name} {last_name}".strip()
            
            if full_name:
                authors.append(full_name)
        
        return authors[:10]
    
    async def _get_full_text_xml(self, pmcid: str) -> Optional[str]:
        """
        Get full text content via Europe PMC XML API
        
        Args:
            pmcid (str): PMC ID (e.g., PMC1234567)
            
        Returns:
            Optional[str]: Full text content if available
        """
        try:
            await self._create_session()
            
            # Make sure PMC ID has the correct format
            if not pmcid.startswith("PMC"):
                pmcid = f"PMC{pmcid}"
            
            # Use the correct endpoint format
            xml_url = f"{EUROPE_PMC_FULLTEXT_XML_URL}/{pmcid}/fullTextXML"
            
            logger.info(f"Fetching full text XML from: {xml_url}")
            
            async with self.session.get(xml_url) as response:
                logger.debug(f"XML API response status for {pmcid}: {response.status}")
                if response.status == 200:
                    xml_content = await response.text()
                    logger.debug(f"XML content length for {pmcid}: {len(xml_content)}")
                    
                    # Extract text content from XML
                    # Remove XML tags while preserving structure
                    text_content = xml_content
                    
                    # Extract main text sections
                    main_text = ""
                    
                    # Extract title
                    title_match = re.search(r'<article-title>(.*?)</article-title>', text_content, re.DOTALL)
                    if title_match:
                        main_text += title_match.group(1) + "\n\n"
                    
                    # Extract abstract
                    abstract_match = re.search(r'<abstract>(.*?)</abstract>', text_content, re.DOTALL)
                    if abstract_match:
                        abstract_text = re.sub(r'<[^>]+>', ' ', abstract_match.group(1))
                        main_text += "Abstract:\n" + abstract_text + "\n\n"
                    
                    # Extract body sections
                    body_match = re.search(r'<body>(.*?)</body>', text_content, re.DOTALL)
                    if body_match:
                        body_text = body_match.group(1)
                        
                        # Extract sections
                        sections = re.findall(r'<sec[^>]*>(.*?)</sec>', body_text, re.DOTALL)
                        
                        for section in sections:
                            # Extract section title
                            section_title_match = re.search(r'<title>(.*?)</title>', section)
                            if section_title_match:
                                main_text += f"\n{section_title_match.group(1)}:\n"
                            
                            # Extract paragraphs
                            paragraphs = re.findall(r'<p>(.*?)</p>', section, re.DOTALL)
                            for para in paragraphs:
                                # Clean paragraph text
                                para_text = re.sub(r'<[^>]+>', ' ', para)
                                para_text = re.sub(r'\s+', ' ', para_text).strip()
                                main_text += para_text + "\n\n"
                    
                    # Clean up the final text
                    main_text = re.sub(r'&lt;', '<', main_text)
                    main_text = re.sub(r'&gt;', '>', main_text)
                    main_text = re.sub(r'&amp;', '&', main_text)
                    main_text = re.sub(r'\s+', ' ', main_text)
                    main_text = re.sub(r'\n{3,}', '\n\n', main_text)
                    
                    return format_text_content(main_text.strip())
                    
                else:
                    logger.debug(f"No XML full text available for {pmcid} (status: {response.status})")
                    return None
                    
        except Exception as e:
            logger.error(f"Error getting XML full text for {pmcid}: {e}")
            return None
    
    async def search_compound(self, compound: CompoundInfo, max_results_per_query: int = 20) -> List[SearchResult]:
        """
        Complete workflow: Search Europe PMC and extract content
        
        Args:
            compound (CompoundInfo): Compound to search for
            max_results_per_query (int): Maximum results per query
            
        Returns:
            List[SearchResult]: Search results with extracted content
        """
        try:
            logger.info(f"Starting Europe PMC search for compound: {compound.name} (CAS: {compound.cas_number})")
            
            # Step 1: Generate search queries
            queries = self.generate_search_queries(compound)
            
            # Step 2: Search Europe PMC for each query
            all_articles = []
            seen_pmids = set()
            
            for query in queries:
                articles = await self.search_europe_pmc(query, max_results_per_query)
                
                # Deduplicate by PMID
                for article in articles:
                    pmid = article.get("pmid")
                    if pmid and pmid not in seen_pmids:
                        seen_pmids.add(pmid)
                        all_articles.append(article)
                
                # Rate limiting
                await asyncio.sleep(0.5)
            
            logger.info(f"Found {len(all_articles)} unique articles across all queries")
            
            if not all_articles:
                logger.warning("No articles found in Europe PMC")
                return []
            
            # Step 3: Extract full text content for eligible articles
            articles_with_full_text = 0
            articles_eligible = sum(1 for a in all_articles if a.get("can_get_full_text"))
            logger.info(f"Articles eligible for full text extraction: {articles_eligible}/{len(all_articles)}")
            
            for article in all_articles:
                if article.get("can_get_full_text") and article.get("pmcid"):
                    logger.debug(f"Attempting to extract full text for PMC ID: {article['pmcid']}")
                    xml_content = await self._get_full_text_xml(article["pmcid"])
                    if xml_content:
                        article["extracted_full_text"] = xml_content
                        articles_with_full_text += 1
                        logger.info(f"Successfully extracted full text for {article['pmcid']}")
                    else:
                        logger.warning(f"Could not extract full text for {article['pmcid']}")
            
            logger.info(f"Extracted full text for {articles_with_full_text} articles")
            
            # Step 4: Build search results
            search_results = []
            
            for article in all_articles:
                # Use extracted full text or fallback to abstract
                full_text = article.get("extracted_full_text", "")
                abstract = article.get("abstract", "")
                
                # Determine extraction method
                if full_text:
                    extraction_method = "xml_api"
                    content_extracted = True
                else:
                    extraction_method = "europe_pmc_api"
                    content_extracted = False
                
                # Create search result
                search_result = SearchResult(
                    title=article["title"],
                    authors=article.get("authors", []),
                    abstract=abstract,
                    link=article.get("abstract_url", ""),
                    pdf_link=None,  # Europe PMC doesn't provide direct PDF links
                    citation_count=0,  # Not available from Europe PMC API
                    year=int(article.get("year", 0)) if article.get("year") else 0,
                    search_term=compound.name,
                    full_text=full_text,
                    content_extracted=content_extracted,
                    extraction_method=extraction_method,
                    extraction_scraper=extraction_method
                )
                
                # Add DOI as a custom attribute
                search_result.doi = article.get("doi", "")
                
                search_results.append(search_result)
            
            # Close the session
            await self._close_session()
            
            logger.info(f"Europe PMC search complete! Found {len(search_results)} papers")
            logger.info(f"  With full text: {sum(1 for r in search_results if r.full_text)}")
            logger.info(f"  With abstracts only: {sum(1 for r in search_results if not r.full_text and r.abstract)}")
            logger.info(f"  With DOI: {sum(1 for r in search_results if hasattr(r, 'doi') and r.doi)}")
            
            return search_results
            
        except Exception as e:
            logger.error(f"Error in Europe PMC search: {e}")
            await self._close_session()
            return []
    
    def get_summary_stats(self, results: List[SearchResult]) -> Dict:
        """Get summary statistics for Europe PMC search results"""
        if not results:
            return {}
        
        # Extract statistics
        years = [r.year for r in results if r.year > 0]
        text_lengths = [len(r.full_text) for r in results if r.full_text]
        abstract_lengths = [len(r.abstract) for r in results if r.abstract]
        dois = [r.doi for r in results if hasattr(r, 'doi') and r.doi]
        
        # Count extraction methods
        extraction_methods = {}
        for r in results:
            if r.extraction_method:
                extraction_methods[r.extraction_method] = extraction_methods.get(r.extraction_method, 0) + 1
        
        return {
            "total_papers": len(results),
            "papers_with_full_text": sum(1 for r in results if r.full_text),
            "papers_with_abstract_only": sum(1 for r in results if not r.full_text and r.abstract),
            "papers_with_doi": len(dois),
            "extraction_success_rate": (
                sum(1 for r in results if r.content_extracted) / len(results) * 100
                if results else 0
            ),
            "average_year": sum(years) / len(years) if years else 0,
            "year_range": [min(years), max(years)] if years else [0, 0],
            "average_text_length": sum(text_lengths) / len(text_lengths) if text_lengths else 0,
            "average_abstract_length": sum(abstract_lengths) / len(abstract_lengths) if abstract_lengths else 0,
            "extraction_methods": extraction_methods,
            "source": "Europe PMC"
        }

# Example usage
async def main():
    """Example usage of the Europe PMC scraper"""
    # Initialize (apify_token not used but kept for interface compatibility)
    scraper = EuropePMCScraper(apify_token="not_used")
    
    # Define compound to search
    compound = CompoundInfo(
        name="Limonene",
        cas_number="5989-27-5",
        synonyms=["(+)-limonene", "d-limonene"]
    )
    
    # Run the search and extraction pipeline
    results = await scraper.search_compound(compound, max_results_per_query=10)
    
    # Get summary statistics
    stats = scraper.get_summary_stats(results)
    print(json.dumps(stats, indent=2))

if __name__ == "__main__":
    asyncio.run(main())
