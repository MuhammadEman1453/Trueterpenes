"""
Enhanced Toxicology Research API with Europe PMC Integration and AI Filtering
API that leverages both Google Scholar and Europe PMC with OpenAI GPT-4.1-mini filtering
Supports multiple extraction methods including DOI extraction and intelligent article filtering
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal
import asyncio
import logging
from datetime import datetime
import json
import re
import os

# Import our modules
from compound_validator import CompoundInput, process_compound_input
from google_scholar_scraper import GoogleScholarScraper, CompoundInfo, SearchResult
from pubmed_scraper import EuropePMCScraper  # Import the new Europe PMC scraper
from config import get_apify_token, DEFAULT_MAX_RESULTS_PER_QUERY
from article_filter import ArticleFilter  # Import the new filtering module

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Enhanced Toxicology Research API with AI Filtering",
    description="API for searching toxicology papers by compound using Google Scholar and Europe PMC with OpenAI GPT-4.1-mini filtering",
    version="5.0.0"
)

# Extended SearchResult model to include DOI and filtering information
class SearchResultModel(BaseModel):
    """Individual search result model with DOI support and filtering info"""
    title: str
    authors: List[str]
    abstract: str
    link: str
    pdf_link: Optional[str] = None
    citation_count: int
    year: int
    search_term: str
    full_text: Optional[str] = None
    content_extracted: bool = False
    extraction_method: Optional[str] = None
    extraction_scraper: Optional[str] = None
    text_length: int = 0
    source: str = "google_scholar"
    doi: Optional[str] = None  # Added DOI field
    
    # Filtering fields
    filtering_confidence: Optional[float] = None
    study_type: Optional[str] = None
    relevant_keywords: Optional[List[str]] = None
    filtering_reasoning: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "title": "Limonene inhalation toxicity in rats",
                "authors": ["Smith J", "Doe A"],
                "abstract": "This study investigates...",
                "link": "https://example.com/paper",
                "doi": "10.1234/example.doi",
                "source": "europe_pmc",
                "filtering_confidence": 0.95,
                "study_type": "animal_studies",
                "relevant_keywords": ["inhalation", "toxicity", "gavage"]
            }
        }

# Request models
class CompoundRequest(BaseModel):
    """Request model for compound search with filtering options"""
    compound_name: Optional[str] = Field(None, description="Name of the compound")
    cas_number: Optional[str] = Field(None, description="CAS registry number")
    search_sources: Optional[List[str]] = Field(["google_scholar", "europe_pmc"], description="Data sources to search")
    max_results_per_query: Optional[int] = Field(DEFAULT_MAX_RESULTS_PER_QUERY, description="Maximum results per query")
    enable_filtering: Optional[bool] = Field(True, description="Enable OpenAI-based article filtering")
    min_confidence: Optional[float] = Field(0.7, description="Minimum confidence threshold for filtering (0.0-1.0)")
    
    class Config:
        schema_extra = {
            "example": {
                "compound_name": "limonene",
                "cas_number": "5989-27-5",
                "search_sources": ["google_scholar", "europe_pmc"],
                "max_results_per_query": 10,
                "enable_filtering": True,
                "min_confidence": 0.7
            }
        }

class ExcludedArticle(BaseModel):
    """Model for excluded articles"""
    title: str
    link: str
    exclusion_reasons: List[str]
    filtering_confidence: float
    reasoning: str

class CompoundResponse(BaseModel):
    """Response model for compound search with filtering results"""
    success: bool
    compound_info: Dict
    search_results: List[SearchResultModel]
    excluded_articles: Optional[List[ExcludedArticle]] = None
    filtering_summary: Optional[Dict] = None
    statistics: Dict
    execution_time: float
    timestamp: str
    sources_searched: List[str]

# Initialize scrapers and filter
google_scholar_scraper = None
europe_pmc_scraper = None
article_filter = None
apify_token = get_apify_token()
openai_api_key = os.getenv("OPENAI_API_KEY")

if apify_token:
    google_scholar_scraper = GoogleScholarScraper(apify_token)
    europe_pmc_scraper = EuropePMCScraper(apify_token)
    logger.info("Scrapers initialized successfully with Apify token")
else:
    logger.warning("Apify token not configured. API will return error on search requests.")

if openai_api_key:
    article_filter = ArticleFilter(openai_api_key)
    logger.info("Article filter initialized with OpenAI GPT-4.1-mini")
else:
    logger.warning("OpenAI API key not configured. Filtering will be disabled.")

@app.on_event("startup")
async def startup_event():
    """Startup event to validate configuration"""
    if not apify_token:
        logger.error("API started without valid Apify token")
    if not openai_api_key:
        logger.error("API started without OpenAI API key - filtering disabled")
    else:
        logger.info("Enhanced Toxicology API v5.0 started successfully with Google Scholar, Europe PMC, and GPT-4.1-mini filtering")

@app.post("/search", response_model=CompoundResponse)
async def search_compound(request: CompoundRequest):
    """
    Search for toxicology papers for a given compound using multiple sources with AI filtering
    
    This endpoint:
    1. Validates the input compound
    2. Retrieves synonyms from PubChem
    3. Performs targeted searches on Google Scholar and/or Europe PMC
    4. Extracts full content and DOIs from found papers
    5. Filters articles using OpenAI GPT-4.1-mini based on toxicology criteria
    6. Returns filtered results with detailed filtering information
    
    Args:
        request: CompoundRequest with compound_name and/or cas_number and filtering options
        
    Returns:
        CompoundResponse with filtered search results including DOIs
    """
    start_time = datetime.now()
    
    try:
        # Validate input
        if not request.compound_name and not request.cas_number:
            raise HTTPException(
                status_code=400, 
                detail="At least one of compound_name or cas_number must be provided"
            )
        
        # Check if scrapers are initialized
        if not google_scholar_scraper or not europe_pmc_scraper:
            raise HTTPException(
                status_code=500,
                detail="Scrapers not configured. Please set APIFY_TOKEN in config."
            )
        
        # Check filtering availability
        if request.enable_filtering and not article_filter:
            raise HTTPException(
                status_code=500,
                detail="Article filtering requested but OpenAI API key not configured. Set OPENAI_API_KEY environment variable."
            )
        
        # Validate search sources
        valid_sources = ["google_scholar", "europe_pmc"]
        search_sources = [src for src in request.search_sources if src in valid_sources]
        
        if not search_sources:
            search_sources = valid_sources  # Default to all sources
            
        logger.info(f"Processing request: {request.compound_name}, CAS: {request.cas_number}")
        logger.info(f"Searching sources: {search_sources}")
        logger.info(f"Filtering enabled: {request.enable_filtering}")
        
        # Step 1: Validate compound and get synonyms
        compound_input = CompoundInput(
            name=request.compound_name or "",
            cas_number=request.cas_number or ""
        )
        
        validated_compound = await process_compound_input(compound_input)
        
        # Step 2: Create CompoundInfo for search
        compound_info = CompoundInfo(
            name=validated_compound.name,
            cas_number=validated_compound.cas_number,
            synonyms=validated_compound.synonyms
        )
        
        # Step 3: Execute searches in parallel for selected sources
        search_tasks = []
        
        if "google_scholar" in search_sources:
            search_tasks.append(google_scholar_scraper.search_compound(
                compound_info, 
                max_results_per_query=request.max_results_per_query
            ))
        
        if "europe_pmc" in search_sources:
            search_tasks.append(europe_pmc_scraper.search_compound(
                compound_info,
                max_results_per_query=request.max_results_per_query
            ))
        
        # Wait for all search tasks to complete
        search_results_list = await asyncio.gather(*search_tasks)
        
        # Step 4: Combine and annotate results with source
        all_results = []
        google_scholar_results = []
        europe_pmc_results = []
        
        # Track which index corresponds to which source
        source_index_map = {}
        idx = 0
        if "google_scholar" in search_sources:
            source_index_map["google_scholar"] = idx
            idx += 1
        if "europe_pmc" in search_sources:
            source_index_map["europe_pmc"] = idx
        
        # Process Google Scholar results
        if "google_scholar" in search_sources:
            gs_results = search_results_list[source_index_map["google_scholar"]]
            google_scholar_results = gs_results
            for result in gs_results:
                result.source = "google_scholar"
                all_results.append(result)
                
        # Process Europe PMC results
        if "europe_pmc" in search_sources:
            epmc_results = search_results_list[source_index_map["europe_pmc"]]
            europe_pmc_results = epmc_results
            for result in epmc_results:
                result.source = "europe_pmc"
                all_results.append(result)
        
        # Step 5: Deduplicate results based on title similarity and DOI
        unique_results = deduplicate_results_with_doi(all_results)
        
        # Step 6: Apply OpenAI filtering if enabled
        filtering_summary = None
        excluded_articles = None
        
        if request.enable_filtering and article_filter:
            logger.info(f"Starting OpenAI GPT-4.1-mini filtering for {len(unique_results)} articles")
            
            filtering_result = await article_filter.filter_articles_batch(
                compound_name=validated_compound.name,
                articles=unique_results,
                batch_size=5,  # Process 5 articles at a time
                min_confidence=request.min_confidence
            )
            
            # Update results with filtering information
            unique_results = filtering_result["included_articles"]
            excluded_articles = filtering_result["excluded_articles"]
            filtering_summary = filtering_result["filtering_summary"]
            
            # Add study type distribution to filtering summary
            filtering_summary["study_types_distribution"] = filtering_result["study_types_distribution"]
            filtering_summary["exclusion_reasons_distribution"] = filtering_result["exclusion_reasons_distribution"]
            
            logger.info(f"Filtering complete: {len(unique_results)} articles passed filtering")
        
        # Step 7: Get aggregated statistics
        stats = get_combined_stats(google_scholar_results, europe_pmc_results, search_sources)
        
        # Add filtering statistics
        if filtering_summary:
            stats["filtering"] = filtering_summary
        
        # Calculate execution time
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Format response
        response_data = {
            "success": True,
            "compound_info": {
                "name": validated_compound.name,
                "cas_number": validated_compound.cas_number,
                "synonyms": validated_compound.synonyms,
                "iupac_name": validated_compound.iupac_name,
                "molecular_formula": validated_compound.molecular_formula,
                "molecular_weight": validated_compound.molecular_weight,
                "pubchem_cid": validated_compound.pubchem_cid
            },
            "search_results": [
                {
                    "title": result.title,
                    "authors": result.authors,
                    "abstract": result.abstract,
                    "link": result.link,
                    "pdf_link": result.pdf_link,
                    "citation_count": result.citation_count,
                    "year": result.year,
                    "search_term": result.search_term,
                    "full_text": result.full_text,
                    "content_extracted": result.content_extracted,
                    "extraction_method": result.extraction_method,
                    "extraction_scraper": result.extraction_scraper,
                    "text_length": len(result.full_text) if result.full_text else 0,
                    "source": result.source,
                    "doi": getattr(result, 'doi', None),  # Include DOI if available
                    # Add filtering information if available
                    "filtering_confidence": getattr(result, 'filtering_confidence', None),
                    "study_type": getattr(result, 'study_type', None),
                    "relevant_keywords": getattr(result, 'relevant_keywords', None),
                    "filtering_reasoning": getattr(result, 'filtering_result', {}).get('reasoning', None) if hasattr(result, 'filtering_result') else None
                }
                for result in unique_results
            ],
            "excluded_articles": [
                {
                    "title": article.title,
                    "link": article.link,
                    "exclusion_reasons": getattr(article, 'exclusion_reasons', []),
                    "filtering_confidence": getattr(article, 'filtering_result', {}).get('confidence', 0.0) if hasattr(article, 'filtering_result') else 0.0,
                    "reasoning": getattr(article, 'filtering_result', {}).get('reasoning', '') if hasattr(article, 'filtering_result') else ''
                }
                for article in (excluded_articles or [])
            ] if excluded_articles else None,
            "filtering_summary": filtering_summary,
            "statistics": stats,
            "execution_time": execution_time,
            "timestamp": datetime.now().isoformat(),
            "sources_searched": search_sources
        }
        
        logger.info(f"Successfully processed compound. Found {len(unique_results)} relevant papers in {execution_time:.2f}s")
        logger.info(f"Papers with extracted content: {len([r for r in unique_results if r.content_extracted])}")
        logger.info(f"Papers with DOI: {len([r for r in unique_results if hasattr(r, 'doi') and r.doi])}")
        if filtering_summary:
            logger.info(f"Filtering statistics: {filtering_summary['inclusion_rate']:.1f}% inclusion rate")
        
        return response_data
        
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing compound: {str(e)}"
        )

def deduplicate_results_with_doi(results: List[SearchResult]) -> List[SearchResult]:
    """
    Enhanced deduplication of search results based on title similarity, DOI, and other identifiers
    
    Args:
        results: List of search results from multiple sources
        
    Returns:
        List of deduplicated search results with merged information
    """
    if not results:
        return []
    
    import difflib
    from urllib.parse import urlparse
    
    # Helper function to normalize text for comparison
    def normalize_text(text: str) -> str:
        """Normalize text for comparison: lowercase, remove punctuation, extra spaces"""
        import re
        text = text.lower().strip()
        text = re.sub(r'[^\w\s]', ' ', text)  # Replace non-alphanumeric with space
        text = re.sub(r'\s+', ' ', text)      # Replace multiple spaces with single space
        return text.strip()
    
    # Helper function to calculate title similarity
    def title_similarity(title1: str, title2: str) -> float:
        """Calculate similarity between two titles using SequenceMatcher"""
        norm_title1 = normalize_text(title1)
        norm_title2 = normalize_text(title2)
        return difflib.SequenceMatcher(None, norm_title1, norm_title2).ratio()
    
    # Helper function to extract domain from URL
    def get_domain(url: str) -> str:
        """Extract domain from URL for comparison"""
        try:
            parsed = urlparse(url)
            return parsed.netloc.lower()
        except:
            return ""
    
    # Helper function to merge two results
    def merge_results(primary: SearchResult, secondary: SearchResult) -> SearchResult:
        """Merge information from secondary result into primary result"""
        # Keep the primary result but enhance with secondary data
        if not primary.full_text and secondary.full_text:
            primary.full_text = secondary.full_text
            primary.content_extracted = True
            primary.extraction_method = secondary.extraction_method
            primary.extraction_scraper = secondary.extraction_scraper
        
        # Merge DOI if missing
        if not hasattr(primary, 'doi') or not primary.doi:
            if hasattr(secondary, 'doi') and secondary.doi:
                primary.doi = secondary.doi
        
        # Use longer abstract
        if secondary.abstract and len(secondary.abstract) > len(primary.abstract):
            primary.abstract = secondary.abstract
        
        # Keep the higher citation count
        if secondary.citation_count > primary.citation_count:
            primary.citation_count = secondary.citation_count
        
        # Merge PDF links if missing
        if not primary.pdf_link and secondary.pdf_link:
            primary.pdf_link = secondary.pdf_link
        
        # Combine authors (unique)
        all_authors = primary.authors + secondary.authors
        seen_authors = set()
        unique_authors = []
        for author in all_authors:
            normalized_author = author.lower().strip()
            if normalized_author not in seen_authors:
                seen_authors.add(normalized_author)
                unique_authors.append(author)
        primary.authors = unique_authors[:15]  # Limit to 15 authors
        
        return primary
    
    # Main deduplication logic
    unique_results = []
    result_clusters = []  # Group of similar results
    
    # Sort results by content quality (prefer results with full text and DOI)
    sorted_results = sorted(results, key=lambda r: (
        r.content_extracted,  # Prefer results with extracted content
        bool(getattr(r, 'doi', None)),  # Prefer results with DOI
        len(r.abstract),  # Prefer longer abstracts
        r.citation_count  # Prefer higher citation count
    ), reverse=True)
    
    # Process each result
    for result in sorted_results:
        is_duplicate = False
        merge_with_cluster = None
        
        # Check against existing clusters
        for cluster_idx, cluster in enumerate(result_clusters):
            representative = cluster[0]  # First result in cluster is the representative
            
            # Check DOI match first (most reliable)
            if hasattr(result, 'doi') and result.doi and hasattr(representative, 'doi') and representative.doi:
                if result.doi.lower().strip() == representative.doi.lower().strip():
                    is_duplicate = True
                    merge_with_cluster = cluster_idx
                    break
            
            # Check title similarity
            title_sim = title_similarity(result.title, representative.title)
            if title_sim > 0.85:  # 85% similarity threshold
                is_duplicate = True
                merge_with_cluster = cluster_idx
                break
            
            # Check if titles are substrings of each other
            norm_title1 = normalize_text(result.title)
            norm_title2 = normalize_text(representative.title)
            if (len(norm_title1) > 10 and len(norm_title2) > 10 and 
                (norm_title1 in norm_title2 or norm_title2 in norm_title1)):
                is_duplicate = True
                merge_with_cluster = cluster_idx
                break
            
            # Check URL similarity (same paper on different sites)
            if result.link and representative.link:
                # Extract article ID from common academic sites
                url_patterns = [
                    r'/article/(.+?)(?:/|$)',  # Generic article pattern
                    r'/abs/(.+?)(?:/|$)',      # Abstract pattern
                    r'/doi/(.+?)(?:/|$)',      # DOI pattern
                    r'/pmc/articles/(.+?)(?:/|$)',  # PMC pattern
                    r'id=(.+?)(?:&|$)',        # Query parameter pattern
                ]
                
                result_id = None
                rep_id = None
                
                for pattern in url_patterns:
                    if not result_id:
                        match = re.search(pattern, result.link, re.IGNORECASE)
                        if match:
                            result_id = match.group(1)
                    
                    if not rep_id:
                        match = re.search(pattern, representative.link, re.IGNORECASE)
                        if match:
                            rep_id = match.group(1)
                
                if result_id and rep_id and result_id == rep_id:
                    is_duplicate = True
                    merge_with_cluster = cluster_idx
                    break
        
        # Handle the result
        if is_duplicate and merge_with_cluster is not None:
            # Add to existing cluster
            result_clusters[merge_with_cluster].append(result)
        else:
            # Create new cluster
            result_clusters.append([result])
    
    # Process clusters to create final results
    for cluster in result_clusters:
        if not cluster:
            continue
        
        # Start with the first (best) result
        final_result = cluster[0]
        
        # Merge information from other results in the cluster
        for other_result in cluster[1:]:
            final_result = merge_results(final_result, other_result)
        
        # Add source information
        sources = set()
        for result in cluster:
            if hasattr(result, 'source'):
                sources.add(result.source)
        
        # If result is from multiple sources, note it
        if len(sources) > 1:
            final_result.source = "multiple"  # or you could keep it as a list
            # Optionally, you could add a custom field
            final_result.sources_list = list(sources)
        
        unique_results.append(final_result)
    
    # Log deduplication statistics
    logger.info(f"Deduplication complete: {len(results)} -> {len(unique_results)} unique papers")
    logger.info(f"Found {len(results) - len(unique_results)} duplicates")
    
    # Sort final results by relevance (citation count, year, has full text)
    unique_results.sort(key=lambda r: (
        r.content_extracted,
        r.citation_count,
        r.year
    ), reverse=True)
    
    return unique_results

def get_combined_stats(google_scholar_results: List[SearchResult], 
                       europe_pmc_results: List[SearchResult],
                       sources_searched: List[str]) -> Dict:
    """Get combined statistics from multiple search sources"""
    stats = {
        "total_papers": 0,
        "papers_with_full_text": 0,
        "papers_with_abstract_only": 0,
        "papers_with_doi": 0,
        "extraction_success_rate": 0,
        "sources": {}
    }
    
    # Add Google Scholar statistics if available
    if "google_scholar" in sources_searched and google_scholar_results:
        gs_stats = google_scholar_scraper.get_summary_stats(google_scholar_results)
        doi_count = sum(1 for r in google_scholar_results if hasattr(r, 'doi') and r.doi)
        gs_stats["papers_with_doi"] = doi_count
        stats["sources"]["google_scholar"] = gs_stats
        stats["total_papers"] += gs_stats.get("total_papers", 0)
        stats["papers_with_full_text"] += gs_stats.get("papers_with_full_text", 0)
        stats["papers_with_abstract_only"] += gs_stats.get("papers_with_abstract_only", 0)
        stats["papers_with_doi"] += doi_count
    
    # Add Europe PMC statistics if available
    if "europe_pmc" in sources_searched and europe_pmc_results:
        epmc_stats = europe_pmc_scraper.get_summary_stats(europe_pmc_results)
        stats["sources"]["europe_pmc"] = epmc_stats
        stats["total_papers"] += epmc_stats.get("total_papers", 0)
        stats["papers_with_full_text"] += epmc_stats.get("papers_with_full_text", 0)
        stats["papers_with_abstract_only"] += epmc_stats.get("papers_with_abstract_only", 0)
        stats["papers_with_doi"] += epmc_stats.get("papers_with_doi", 0)
    
    # Calculate overall extraction success rate
    total_papers = stats["total_papers"]
    if total_papers > 0:
        stats["extraction_success_rate"] = (stats["papers_with_full_text"] / total_papers) * 100
    
    return stats

@app.get("/health")
async def health_check():
    """Health check endpoint with scraper and filter status"""
    apify_configured = apify_token is not None
    openai_configured = openai_api_key is not None
    
    gs_status = "initialized" if google_scholar_scraper else "not_initialized"
    epmc_status = "initialized" if europe_pmc_scraper else "not_initialized"
    filter_status = "initialized" if article_filter else "not_initialized"
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "apify_configured": apify_configured,
        "openai_configured": openai_configured,
        "apify_token_status": "configured" if apify_token else "not_configured",
        "openai_key_status": "configured" if openai_api_key else "not_configured",
        "scrapers": {
            "google_scholar": gs_status,
            "europe_pmc": epmc_status
        },
        "filtering": {
            "status": filter_status,
            "model": "gpt-4.1-mini-2025-04-14" if article_filter else None,
            "enabled": openai_configured
        },
        "actors": {
            "scholar_search": "marco.gullo/google-scholar-scraper",
            "pdf_scraper": "jirimoravcik/pdf-text-extractor",
            "cheerio_scraper": "apify/cheerio-scraper",
            "note": "Europe PMC uses REST API only, no Apify actors needed"
        }
    }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Enhanced Toxicology Research API v5.0 with GPT-4.1-mini Filtering",
        "version": "5.0.0",
        "new_features": {
            "ai_filtering": "OpenAI GPT-4.1-mini based article filtering for toxicology research",
            "filtering_confidence": "Configurable confidence thresholds for article inclusion",
            "study_type_classification": "Automatic classification of study types",
            "detailed_exclusion_reasons": "Detailed reasons for article exclusions",
            "europe_pmc_integration": "Replaced PubMed with Europe PMC for better abstracts and full text",
            "doi_extraction": "Extract and track DOIs from papers",
            "improved_deduplication": "Deduplicate by both title and DOI"
        },
        "endpoints": {
            "search": "/search (POST) - Search with OpenAI filtering",
            "health": "/health (GET) - Health check with filter status",
            "docs": "/docs - API documentation",
            "test": "/test/{compound_name} - Quick test search with filtering",
            "filter-test": "/filter-test/{compound_name} - Test filtering only",
            "source_stats": "/source-stats/{compound_name} - Source-specific statistics",
            "compare-sources": "/compare-sources/{compound_name} - Direct source comparison"
        },
        "filtering_criteria": {
            "included_studies": [
                "Inhalation toxicity studies",
                "Animal studies (any route)",
                "In vitro cell-based studies", 
                "Thermal stability & pyrolysis studies",
                "In silico/QSAR modeling"
            ],
            "excluded_studies": [
                "Reference-only mentions",
                "Chemical composition analysis only",
                "Insecticidal/repellent studies",
                "Chemical synthesis focus",
                "Degradation product mentions only"
            ]
        },
        "search_strategy": {
            "google_scholar": "PDF-first extraction with Cheerio fallback for HTML content",
            "europe_pmc": "REST API with direct abstract access and XML API for full text"
        },
        "ai_model": "GPT-4.1-mini (gpt-4.1-mini-2025-04-14)"
    }

@app.get("/test/{compound_name}")
async def test_search(
    compound_name: str, 
    cas_number: Optional[str] = None,
    sources: Optional[str] = "google_scholar,europe_pmc",
    enable_filtering: Optional[bool] = True,
    min_confidence: Optional[float] = 0.7
):
    """
    Test endpoint for quick searches with filtering
    Usage: GET /test/limonene?cas_number=5989-27-5&sources=google_scholar,europe_pmc&enable_filtering=true
    """
    search_sources = sources.split(",") if sources else ["google_scholar", "europe_pmc"]
    
    request = CompoundRequest(
        compound_name=compound_name, 
        cas_number=cas_number,
        search_sources=search_sources,
        max_results_per_query=5,  # Limit results for quick testing
        enable_filtering=enable_filtering,
        min_confidence=min_confidence
    )
    
    return await search_compound(request)

@app.get("/source-stats/{compound_name}")
async def source_statistics(
    compound_name: str,
    sources: Optional[str] = "google_scholar,europe_pmc",
    enable_filtering: Optional[bool] = True
):
    """Compare statistics across different sources for a compound with optional filtering"""
    try:
        search_sources = sources.split(",") if sources else ["google_scholar", "europe_pmc"]
        
        request = CompoundRequest(
            compound_name=compound_name,
            search_sources=search_sources,
            max_results_per_query=10,
            enable_filtering=enable_filtering
        )
        
        response = await search_compound(request)
        
        # Analyze results by source
        source_analysis = {}
        
        # Group results by source
        for result in response.search_results:
            source = result.source
            if source not in source_analysis:
                source_analysis[source] = {
                    "total_papers": 0,
                    "papers_with_full_text": 0,
                    "papers_with_abstract_only": 0,
                    "papers_with_doi": 0,
                    "extraction_methods": {},
                    "extraction_scrapers": {},
                    "study_types": {},
                    "avg_text_length": 0,
                    "total_text_length": 0,
                    "year_distribution": {},
                    "avg_filtering_confidence": 0,
                    "total_confidence": 0
                }
            
            # Update statistics
            stats = source_analysis[source]
            stats["total_papers"] += 1
            
            if result.full_text:
                stats["papers_with_full_text"] += 1
                stats["total_text_length"] += len(result.full_text)
            elif result.abstract:
                stats["papers_with_abstract_only"] += 1
            
            if result.doi:
                stats["papers_with_doi"] += 1
            
            # Track extraction methods and scrapers
            if result.extraction_method:
                stats["extraction_methods"][result.extraction_method] = stats["extraction_methods"].get(result.extraction_method, 0) + 1
            
            if result.extraction_scraper:
                stats["extraction_scrapers"][result.extraction_scraper] = stats["extraction_scrapers"].get(result.extraction_scraper, 0) + 1
            
            # Track study types (filtering result)
            if result.study_type:
                stats["study_types"][result.study_type] = stats["study_types"].get(result.study_type, 0) + 1
            
            # Track filtering confidence
            if result.filtering_confidence:
                stats["total_confidence"] += result.filtering_confidence
            
            # Track year distribution
            if result.year:
                stats["year_distribution"][result.year] = stats["year_distribution"].get(result.year, 0) + 1
        
        # Calculate averages
        for source, stats in source_analysis.items():
            if stats["papers_with_full_text"] > 0:
                stats["avg_text_length"] = stats["total_text_length"] / stats["papers_with_full_text"]
            if stats["total_papers"] > 0:
                stats["avg_filtering_confidence"] = stats["total_confidence"] / stats["total_papers"]
        
        # Compare extraction rates
        source_comparison = {
            "compound": compound_name,
            "sources_searched": search_sources,
            "filtering_enabled": enable_filtering,
            "source_analysis": source_analysis,
            "source_comparison": {
                "papers_found": {source: stats["total_papers"] for source, stats in source_analysis.items()},
                "extraction_success_rate": {
                    source: (stats["papers_with_full_text"] / stats["total_papers"] * 100) if stats["total_papers"] > 0 else 0
                    for source, stats in source_analysis.items()
                },
                "doi_coverage": {
                    source: (stats["papers_with_doi"] / stats["total_papers"] * 100) if stats["total_papers"] > 0 else 0
                    for source, stats in source_analysis.items()
                },
                "avg_text_length": {source: stats["avg_text_length"] for source, stats in source_analysis.items()},
                "avg_filtering_confidence": {source: stats["avg_filtering_confidence"] for source, stats in source_analysis.items()}
            },
            "filtering_summary": response.filtering_summary
        }
        
        return source_comparison
        
    except Exception as e:
        logger.error(f"Error generating source statistics: {e}")
        return {"error": str(e)}

@app.get("/europe-pmc-test/{compound_name}")
async def test_europe_pmc_extraction(compound_name: str):
    """Test endpoint specifically for Europe PMC extraction"""
    try:
        if not europe_pmc_scraper:
            return {"error": "Europe PMC scraper not initialized"}
        
        # Create compound info for testing
        compound_input = CompoundInput(name=compound_name, cas_number="")
        validated_compound = await process_compound_input(compound_input)
        
        compound_info = CompoundInfo(
            name=validated_compound.name,
            cas_number=validated_compound.cas_number,
            synonyms=validated_compound.synonyms
        )
        
        # Test Europe PMC search only
        results = await europe_pmc_scraper.search_compound(compound_info, max_results_per_query=5)
        
        # Get statistics
        stats = europe_pmc_scraper.get_summary_stats(results)
        
        # Format results for display
        return {
            "compound": compound_name,
            "europe_pmc_search": {
                "total_papers": len(results),
                "papers_with_full_text": sum(1 for r in results if r.full_text),
                "papers_with_abstract_only": sum(1 for r in results if not r.full_text and r.abstract),
                "papers_with_doi": sum(1 for r in results if hasattr(r, 'doi') and r.doi),
                "extraction_methods": stats.get("extraction_methods", {})
            },
            "sample_results": [
                {
                    "title": r.title,
                    "abstract_preview": r.abstract[:200] + "..." if r.abstract else "No abstract",
                    "full_text_available": bool(r.full_text),
                    "text_length": len(r.full_text) if r.full_text else 0,
                    "doi": getattr(r, 'doi', None),
                    "extraction_method": r.extraction_method,
                    "link": r.link
                }
                for r in results[:3]  # Show first 3 results
            ],
            "search_queries": europe_pmc_scraper.generate_search_queries(compound_info)[:3],
            "extraction_success_rate": stats.get("extraction_success_rate", 0)
        }
        
    except Exception as e:
        logger.error(f"Error in Europe PMC test: {e}")
        return {"error": str(e)}

@app.get("/sources-config")
async def get_sources_configuration():
    """Get detailed configuration for all data sources"""
    return {
        "api_version": "5.0.0",
        "data_sources": {
            "google_scholar": {
                "description": "Academic search engine with broad coverage",
                "search_method": "Marco Gullo's scraper with targeted queries",
                "content_extraction": {
                    "primary": "PDF extraction using jirimoravcik's PDF scraper",
                    "fallback": "HTML extraction using Cheerio"
                },
                "advantages": [
                    "High number of papers",
                    "Good PDF availability",
                    "Citation count information"
                ]
            },
            "europe_pmc": {
                "description": "European life sciences literature database",
                "search_method": "REST API with direct abstract access",
                "content_extraction": {
                    "primary": "XML API for full text extraction",
                    "fallback": "Abstracts directly from API response"
                },
                "advantages": [
                    "Direct abstract access in API",
                    "Reliable DOI information",
                    "Better full text availability for open access papers",
                    "Consistent structured data",
                    "No web scraping needed"
                ]
            }
        },
        "ai_filtering": {
            "model": "gpt-4o-mini-2024-07-18",
            "description": "OpenAI GPT-4.1-mini for intelligent article filtering",
            "capabilities": [
                "Automatic relevance assessment",
                "Study type classification",
                "Toxicology criteria matching",
                "Confidence scoring",
                "Detailed exclusion reasoning"
            ]
        },
        "integration_logic": {
            "search_execution": "Parallel searches across all enabled sources",
            "result_processing": "Source-specific search result annotation",
            "deduplication": "Cross-source deduplication based on DOI and title",
            "ai_filtering": "OpenAI analysis of each article for toxicology relevance",
            "statistics": "Aggregated statistics with source-specific and filtering breakdowns"
        },
        "usage_recommendations": {
            "toxicology_research": "Use both sources with filtering enabled for best results",
            "medical_focus": "Prioritize Europe PMC for life sciences papers",
            "broad_research": "Prioritize Google Scholar for wider coverage",
            "doi_tracking": "Europe PMC provides most reliable DOI information",
            "high_quality_results": "Enable filtering with confidence threshold ≥ 0.7"
        }
    }

@app.get("/compare-sources/{compound_name}")
async def compare_sources(compound_name: str, enable_filtering: Optional[bool] = True):
    """Direct comparison between Google Scholar and Europe PMC results with filtering analysis"""
    try:
        # Run searches for both sources
        request = CompoundRequest(
            compound_name=compound_name,
            search_sources=["google_scholar", "europe_pmc"],
            max_results_per_query=10,
            enable_filtering=enable_filtering
        )
        
        response = await search_compound(request)
        
        # Group results by source
        google_scholar_results = [r for r in response.search_results if r.source == "google_scholar"]
        europe_pmc_results = [r for r in response.search_results if r.source == "europe_pmc"]
        
        # Find overlapping papers (by DOI or title)
        gs_dois = {r.doi for r in google_scholar_results if r.doi}
        epmc_dois = {r.doi for r in europe_pmc_results if r.doi}
        doi_overlap = gs_dois.intersection(epmc_dois)
        
        gs_titles = {r.title.lower().strip() for r in google_scholar_results}
        epmc_titles = {r.title.lower().strip() for r in europe_pmc_results}
        title_overlap = gs_titles.intersection(epmc_titles)
        
        # Compare content extraction success
        gs_extraction_rate = (
            sum(1 for r in google_scholar_results if r.content_extracted) / 
            max(1, len(google_scholar_results)) * 100
        )
        
        epmc_extraction_rate = (
            sum(1 for r in europe_pmc_results if r.content_extracted) / 
            max(1, len(europe_pmc_results)) * 100
        )
        
        # Compare DOI coverage
        gs_doi_rate = (
            sum(1 for r in google_scholar_results if r.doi) / 
            max(1, len(google_scholar_results)) * 100
        )
        
        epmc_doi_rate = (
            sum(1 for r in europe_pmc_results if r.doi) / 
            max(1, len(europe_pmc_results)) * 100
        )
        
        # Compare filtering confidence
        gs_avg_confidence = (
            sum(r.filtering_confidence for r in google_scholar_results if r.filtering_confidence) / 
            max(1, len([r for r in google_scholar_results if r.filtering_confidence]))
        )
        
        epmc_avg_confidence = (
            sum(r.filtering_confidence for r in europe_pmc_results if r.filtering_confidence) / 
            max(1, len([r for r in europe_pmc_results if r.filtering_confidence]))
        )
        
        return {
            "compound": compound_name,
            "filtering_enabled": enable_filtering,
            "results_summary": {
                "google_scholar_papers": len(google_scholar_results),
                "europe_pmc_papers": len(europe_pmc_results),
                "unique_papers": len(set(gs_titles).union(epmc_titles)),
                "overlapping_papers_by_title": len(title_overlap),
                "overlapping_papers_by_doi": len(doi_overlap)
            },
            "content_extraction": {
                "google_scholar_extraction_rate": gs_extraction_rate,
                "europe_pmc_extraction_rate": epmc_extraction_rate,
                "google_scholar_with_full_text": sum(1 for r in google_scholar_results if r.full_text),
                "europe_pmc_with_full_text": sum(1 for r in europe_pmc_results if r.full_text)
            },
            "doi_coverage": {
                "google_scholar_doi_rate": gs_doi_rate,
                "europe_pmc_doi_rate": epmc_doi_rate
            },
            "filtering_analysis": {
                "google_scholar_avg_confidence": gs_avg_confidence,
                "europe_pmc_avg_confidence": epmc_avg_confidence,
                "google_scholar_study_types": get_study_type_distribution(google_scholar_results),
                "europe_pmc_study_types": get_study_type_distribution(europe_pmc_results)
            },
            "year_distribution": {
                "google_scholar": get_year_distribution(google_scholar_results),
                "europe_pmc": get_year_distribution(europe_pmc_results)
            },
            "extraction_methods": {
                "google_scholar": get_extraction_methods(google_scholar_results),
                "europe_pmc": get_extraction_methods(europe_pmc_results)
            },
            "recommendation": get_source_recommendation(google_scholar_results, europe_pmc_results),
            "filtering_summary": response.filtering_summary
        }
        
    except Exception as e:
        logger.error(f"Error comparing sources: {e}")
        return {"error": str(e)}

def get_year_distribution(results: List[SearchResult]) -> Dict[int, int]:
    """Get year distribution from results"""
    year_dist = {}
    for result in results:
        if result.year:
            year_dist[result.year] = year_dist.get(result.year, 0) + 1
    return year_dist

def get_extraction_methods(results: List[SearchResult]) -> Dict[str, int]:
    """Get extraction method distribution from results"""
    method_dist = {}
    for result in results:
        if result.extraction_method:
            method_dist[result.extraction_method] = method_dist.get(result.extraction_method, 0) + 1
    return method_dist

def get_study_type_distribution(results: List[SearchResult]) -> Dict[str, int]:
    """Get study type distribution from filtering results"""
    study_type_dist = {}
    for result in results:
        if hasattr(result, 'study_type') and result.study_type:
            study_type_dist[result.study_type] = study_type_dist.get(result.study_type, 0) + 1
    return study_type_dist

def get_source_recommendation(gs_results: List[SearchResult], epmc_results: List[SearchResult]) -> Dict:
    """Generate a recommendation based on the results from both sources"""
    gs_full_text = sum(1 for r in gs_results if r.full_text)
    epmc_full_text = sum(1 for r in epmc_results if r.full_text)
    
    gs_extraction_rate = gs_full_text / max(1, len(gs_results)) * 100
    epmc_extraction_rate = epmc_full_text / max(1, len(epmc_results)) * 100
    
    gs_doi_rate = sum(1 for r in gs_results if hasattr(r, 'doi') and r.doi) / max(1, len(gs_results)) * 100
    epmc_doi_rate = sum(1 for r in epmc_results if hasattr(r, 'doi') and r.doi) / max(1, len(epmc_results)) * 100
    
    # Calculate filtering confidence averages
    gs_confidence = sum(getattr(r, 'filtering_confidence', 0) for r in gs_results) / max(1, len(gs_results))
    epmc_confidence = sum(getattr(r, 'filtering_confidence', 0) for r in epmc_results) / max(1, len(epmc_results))
    
    recommendations = {}
    
    # Paper availability recommendation
    if len(gs_results) > len(epmc_results) * 1.5:
        recommendations["paper_availability"] = "google_scholar"
    elif len(epmc_results) > len(gs_results) * 1.5:
        recommendations["paper_availability"] = "europe_pmc"
    else:
        recommendations["paper_availability"] = "both"
    
    # Content extraction recommendation
    if gs_extraction_rate > epmc_extraction_rate * 1.5:
        recommendations["content_extraction"] = "google_scholar"
    elif epmc_extraction_rate > gs_extraction_rate * 1.5:
        recommendations["content_extraction"] = "europe_pmc"
    else:
        recommendations["content_extraction"] = "both"
    
    # DOI tracking recommendation
    if epmc_doi_rate > gs_doi_rate * 1.2:  # Europe PMC typically better for DOIs
        recommendations["doi_tracking"] = "europe_pmc"
    else:
        recommendations["doi_tracking"] = "both"
    
    # Filtering quality recommendation
    if gs_confidence > epmc_confidence * 1.1:
        recommendations["filtering_quality"] = "google_scholar"
    elif epmc_confidence > gs_confidence * 1.1:
        recommendations["filtering_quality"] = "europe_pmc"
    else:
        recommendations["filtering_quality"] = "both"
    
    # Overall recommendation
    if recommendations["paper_availability"] == recommendations["content_extraction"]:
        recommendations["overall"] = recommendations["paper_availability"]
    else:
        recommendations["overall"] = "both"
    
    recommendations["reason"] = f"Europe PMC: {epmc_doi_rate:.0f}% DOI coverage, Google Scholar: {gs_extraction_rate:.0f}% extraction rate"
    
    return recommendations

@app.get("/filter-test/{compound_name}")
async def test_filtering_only(
    compound_name: str,
    max_results: Optional[int] = 3
):
    """Test endpoint to demonstrate filtering capabilities"""
    try:
        if not article_filter:
            return {"error": "Article filter not initialized - OpenAI API key required"}
        
        # Create sample articles for testing
        sample_articles = [
            SearchResult(
                title="Inhalation toxicity of limonene in Sprague-Dawley rats",
                authors=["Smith J", "Doe A"],
                abstract="This study investigated acute and subchronic inhalation toxicity of d-limonene vapor in Sprague-Dawley rats. Animals were exposed to 0, 300, 1000, or 3000 ppm d-limonene vapor for 6 hours/day, 5 days/week for 4 weeks.",
                link="https://example.com/paper1",
                pdf_link=None,
                citation_count=25,
                year=2023,
                search_term=compound_name
            ),
            SearchResult(
                title="Chemical synthesis of limonene from citrus waste",
                authors=["Johnson B"],
                abstract="This paper describes a novel method for chemical synthesis of d-limonene from citrus peel waste using green chemistry principles.",
                link="https://example.com/paper2",
                pdf_link=None,
                citation_count=15,
                year=2022,
                search_term=compound_name
            ),
            SearchResult(
                title="In vitro cytotoxicity of limonene on human lung cells",
                authors=["Brown C", "Wilson D"],
                abstract="We evaluated the cytotoxic effects of limonene on A549 human lung adenocarcinoma cells using MTT assay and flow cytometry. Cells were exposed to concentrations ranging from 10-1000 μM.",
                link="https://example.com/paper3",
                pdf_link=None,
                citation_count=18,
                year=2023,
                search_term=compound_name
            )
        ]
        
        # Test filtering
        filtering_result = await article_filter.filter_articles_batch(
            compound_name=compound_name,
            articles=sample_articles[:max_results],
            batch_size=3,
            min_confidence=0.5
        )
        
        return {
            "compound": compound_name,
            "model_used": "gpt-4.1-mini-2025-04-14",
            "filtering_summary": filtering_result["filtering_summary"],
            "included_articles": [
                {
                    "title": article.title,
                    "confidence": article.filtering_confidence,
                    "study_type": article.study_type,
                    "relevant_keywords": article.relevant_keywords,
                    "reasoning": article.filtering_result.get("reasoning", "")
                }
                for article in filtering_result["included_articles"]
            ],
            "excluded_articles": [
                {
                    "title": article.title,
                    "exclusion_reasons": article.exclusion_reasons,
                    "reasoning": article.filtering_result.get("reasoning", "")
                }
                for article in filtering_result["excluded_articles"]
            ]
        }
        
    except Exception as e:
        logger.error(f"Error in filter test: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    
    # Check configuration
    if not get_apify_token():
        print("WARNING: Apify token not configured!")
        print("Please set your token in config.py or as environment variable:")
        print("export APIFY_TOKEN='your_actual_token'")
        print("Get your token from: https://console.apify.com/")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("WARNING: OpenAI API key not configured!")
        print("Please set OPENAI_API_KEY environment variable")
        print("Article filtering will be disabled without this key")
    
    print("Starting Enhanced Toxicology Research API v5.0...")
    print("Features: Google Scholar + Europe PMC + GPT-4.1-mini Filtering")
    print()
    print("API Documentation: http://localhost:8000/docs")
    print("Test endpoint: http://localhost:8000/test/limonene?enable_filtering=true")
    print("Filter test: http://localhost:8000/filter-test/limonene")
    print("Source statistics: http://localhost:8000/source-stats/limonene")
    print("Europe PMC test: http://localhost:8000/europe-pmc-test/limonene")
    print("Source comparison: http://localhost:8000/compare-sources/limonene")
    print("Sources config: http://localhost:8000/sources-config")
    
    # Run the API server
    uvicorn.run(app, host="0.0.0.0", port=8000)
