"""
Toxicology Research API
Enhanced API that leverages both Google Scholar and PubMed for comprehensive search results
Supports multiple extraction methods: Cheerio HTML extraction and PDF scraping
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal
import asyncio
import logging
from datetime import datetime
import json

# Import our modules
from compound_validator import CompoundInput, process_compound_input
from google_scholar_scraper import GoogleScholarScraper, CompoundInfo, SearchResult
from pubmed_scraper import PubMedScraper  # Import the new PubMed scraper
from config import get_apify_token, DEFAULT_MAX_RESULTS_PER_QUERY

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Enhanced Toxicology Research API",
    description="API for searching toxicology papers by compound using Google Scholar and PubMed with multiple extraction methods",
    version="3.0.0"
)

# Pydantic models for API
class CompoundRequest(BaseModel):
    """Request model for compound search"""
    compound_name: Optional[str] = Field(None, description="Name of the compound")
    cas_number: Optional[str] = Field(None, description="CAS registry number")
    search_sources: Optional[List[str]] = Field(["google_scholar", "pubmed"], description="Data sources to search")
    max_results_per_query: Optional[int] = Field(DEFAULT_MAX_RESULTS_PER_QUERY, description="Maximum results per query")
    
    class Config:
        schema_extra = {
            "example": {
                "compound_name": "limonene",
                "cas_number": "5989-27-5",
                "search_sources": ["google_scholar", "pubmed"],
                "max_results_per_query": 10
            }
        }

class SearchResultModel(BaseModel):
    """Individual search result model"""
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
    source: str = "google_scholar"  # Added source field to track origin

class CompoundResponse(BaseModel):
    """Response model for compound search"""
    success: bool
    compound_info: Dict
    search_results: List[SearchResultModel]
    statistics: Dict
    execution_time: float
    timestamp: str
    sources_searched: List[str]

# Initialize scrapers
google_scholar_scraper = None
pubmed_scraper = None
apify_token = get_apify_token()

if apify_token:
    google_scholar_scraper = GoogleScholarScraper(apify_token)
    pubmed_scraper = PubMedScraper(apify_token)
    logger.info("Scrapers initialized successfully with Apify token")
else:
    logger.warning("Apify token not configured. API will return error on search requests.")

@app.on_event("startup")
async def startup_event():
    """Startup event to validate configuration"""
    if not apify_token:
        logger.error("API started without valid Apify token")
    else:
        logger.info("Enhanced Toxicology API v3.0 started successfully with Google Scholar and PubMed scrapers")

@app.post("/search", response_model=CompoundResponse)
async def search_compound(request: CompoundRequest):
    """
    Search for toxicology papers for a given compound using multiple sources
    
    This endpoint:
    1. Validates the input compound
    2. Retrieves synonyms from PubChem
    3. Performs targeted searches on Google Scholar and/or PubMed
    4. Extracts full content from found papers using appropriate scrapers
    5. Returns combined and filtered results with metadata
    
    Args:
        request: CompoundRequest with compound_name and/or cas_number
        
    Returns:
        CompoundResponse with search results and metadata including full text
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
        if not google_scholar_scraper or not pubmed_scraper:
            raise HTTPException(
                status_code=500,
                detail="Scrapers not configured. Please set APIFY_TOKEN in config."
            )
        
        # Validate search sources
        valid_sources = ["google_scholar", "pubmed"]
        search_sources = [src for src in request.search_sources if src in valid_sources]
        
        if not search_sources:
            search_sources = valid_sources  # Default to all sources
            
        logger.info(f"Processing request: {request.compound_name}, CAS: {request.cas_number}")
        logger.info(f"Searching sources: {search_sources}")
        
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
        
        if "pubmed" in search_sources:
            search_tasks.append(pubmed_scraper.search_compound(
                compound_info,
                max_results_per_query=request.max_results_per_query
            ))
        
        # Wait for all search tasks to complete
        search_results_list = await asyncio.gather(*search_tasks)
        
        # Step 4: Combine and annotate results with source
        all_results = []
        google_scholar_results = []
        pubmed_results = []
        
        # Track which index corresponds to which source
        source_index_map = {}
        idx = 0
        if "google_scholar" in search_sources:
            source_index_map["google_scholar"] = idx
            idx += 1
        if "pubmed" in search_sources:
            source_index_map["pubmed"] = idx
        
        # Process Google Scholar results
        if "google_scholar" in search_sources:
            gs_results = search_results_list[source_index_map["google_scholar"]]
            google_scholar_results = gs_results
            for result in gs_results:
                result.source = "google_scholar"
                all_results.append(result)
                
        # Process PubMed results
        if "pubmed" in search_sources:
            pm_results = search_results_list[source_index_map["pubmed"]]
            pubmed_results = pm_results
            for result in pm_results:
                result.source = "pubmed"
                all_results.append(result)
        
        # Step 5: Deduplicate results based on title similarity
        unique_results = deduplicate_results(all_results)
        
        # Step 6: Get aggregated statistics
        stats = get_combined_stats(google_scholar_results, pubmed_results, search_sources)
        
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
                    "source": result.source
                }
                for result in unique_results
            ],
            "statistics": stats,
            "execution_time": execution_time,
            "timestamp": datetime.now().isoformat(),
            "sources_searched": search_sources
        }
        
        logger.info(f"Successfully processed compound. Found {len(unique_results)} papers in {execution_time:.2f}s")
        logger.info(f"Papers with extracted content: {len([r for r in unique_results if r.content_extracted])}")
        
        return response_data
        
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing compound: {str(e)}"
        )

def deduplicate_results(results: List[SearchResult]) -> List[SearchResult]:
    """
    Deduplicate search results based on title similarity
    
    Args:
        results: List of search results
        
    Returns:
        List of deduplicated search results
    """
    if not results:
        return []
    
    unique_results = []
    seen_titles = set()
    
    for result in results:
        # Normalize title for comparison
        normalized_title = result.title.lower().strip()
        
        # Simple deduplication by exact title match
        # Could be enhanced with fuzzy matching in production
        if normalized_title not in seen_titles:
            seen_titles.add(normalized_title)
            unique_results.append(result)
    
    return unique_results

def get_combined_stats(google_scholar_results: List[SearchResult], 
                       pubmed_results: List[SearchResult],
                       sources_searched: List[str]) -> Dict:
    """Get combined statistics from multiple search sources"""
    stats = {
        "total_papers": 0,
        "papers_with_full_text": 0,
        "papers_with_abstract_only": 0,
        "extraction_success_rate": 0,
        "sources": {}
    }
    
    # Add Google Scholar statistics if available
    if "google_scholar" in sources_searched and google_scholar_results:
        gs_stats = google_scholar_scraper.get_summary_stats(google_scholar_results)
        stats["sources"]["google_scholar"] = gs_stats
        stats["total_papers"] += gs_stats.get("total_papers", 0)
        stats["papers_with_full_text"] += gs_stats.get("papers_with_full_text", 0)
        stats["papers_with_abstract_only"] += gs_stats.get("papers_with_abstract_only", 0)
    
    # Add PubMed statistics if available
    if "pubmed" in sources_searched and pubmed_results:
        pm_stats = pubmed_scraper.get_summary_stats(pubmed_results)
        stats["sources"]["pubmed"] = pm_stats
        stats["total_papers"] += pm_stats.get("total_papers", 0)
        stats["papers_with_full_text"] += pm_stats.get("papers_with_full_text", 0)
        stats["papers_with_abstract_only"] += pm_stats.get("papers_with_abstract_only", 0)
    
    # Calculate overall extraction success rate
    total_papers = stats["total_papers"]
    if total_papers > 0:
        stats["extraction_success_rate"] = (stats["papers_with_full_text"] / total_papers) * 100
    
    return stats

@app.get("/health")
async def health_check():
    """Health check endpoint with scraper status"""
    apify_configured = apify_token is not None
    
    gs_status = "initialized" if google_scholar_scraper else "not_initialized"
    pm_status = "initialized" if pubmed_scraper else "not_initialized"
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "apify_configured": apify_configured,
        "apify_token_status": "configured" if apify_token else "not_configured",
        "scrapers": {
            "google_scholar": gs_status,
            "pubmed": pm_status
        },
        "actors": {
            "scholar_search": "marco.gullo/google-scholar-scraper",
            "pdf_scraper": "jirimoravcik/pdf-text-extractor",
            "cheerio_scraper": "apify/cheerio-scraper"
        }
    }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Enhanced Toxicology Research API v3.0",
        "version": "3.0.0",
        "new_features": {
            "multi_source_search": "Integrated search across Google Scholar and PubMed",
            "parallel_execution": "Searches run in parallel for faster results",
            "source_tracking": "Results include source information",
            "deduplication": "Cross-source result deduplication"
        },
        "endpoints": {
            "search": "/search (POST) - Search for toxicology papers across multiple sources",
            "health": "/health (GET) - Health check with scraper info",
            "docs": "/docs - API documentation",
            "test": "/test/{compound_name} - Quick test search",
            "source_stats": "/source-stats/{compound_name} - Source-specific statistics"
        },
        "search_strategy": {
            "google_scholar": "PDF-first extraction with Cheerio fallback for HTML content",
            "pubmed": "E-utilities API with full-text extraction via Cheerio"
        }
    }

@app.get("/test/{compound_name}")
async def test_search(
    compound_name: str, 
    cas_number: Optional[str] = None,
    sources: Optional[str] = "google_scholar,pubmed"
):
    """
    Test endpoint for quick searches
    Usage: GET /test/limonene?cas_number=5989-27-5&sources=google_scholar,pubmed
    """
    search_sources = sources.split(",") if sources else ["google_scholar", "pubmed"]
    
    request = CompoundRequest(
        compound_name=compound_name, 
        cas_number=cas_number,
        search_sources=search_sources,
        max_results_per_query=5  # Limit results for quick testing
    )
    
    return await search_compound(request)

@app.get("/source-stats/{compound_name}")
async def source_statistics(
    compound_name: str,
    sources: Optional[str] = "google_scholar,pubmed"
):
    """Compare statistics across different sources for a compound"""
    try:
        search_sources = sources.split(",") if sources else ["google_scholar", "pubmed"]
        
        request = CompoundRequest(
            compound_name=compound_name,
            search_sources=search_sources,
            max_results_per_query=10
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
                    "extraction_methods": {},
                    "extraction_scrapers": {},
                    "avg_text_length": 0,
                    "total_text_length": 0,
                    "year_distribution": {}
                }
            
            # Update statistics
            stats = source_analysis[source]
            stats["total_papers"] += 1
            
            if result.full_text:
                stats["papers_with_full_text"] += 1
                stats["total_text_length"] += len(result.full_text)
            elif result.abstract:
                stats["papers_with_abstract_only"] += 1
            
            # Track extraction methods and scrapers
            if result.extraction_method:
                stats["extraction_methods"][result.extraction_method] = stats["extraction_methods"].get(result.extraction_method, 0) + 1
            
            if result.extraction_scraper:
                stats["extraction_scrapers"][result.extraction_scraper] = stats["extraction_scrapers"].get(result.extraction_scraper, 0) + 1
            
            # Track year distribution
            if result.year:
                stats["year_distribution"][result.year] = stats["year_distribution"].get(result.year, 0) + 1
        
        # Calculate averages
        for source, stats in source_analysis.items():
            if stats["papers_with_full_text"] > 0:
                stats["avg_text_length"] = stats["total_text_length"] / stats["papers_with_full_text"]
        
        # Compare extraction rates
        source_comparison = {
            "compound": compound_name,
            "sources_searched": search_sources,
            "source_analysis": source_analysis,
            "source_comparison": {
                "papers_found": {source: stats["total_papers"] for source, stats in source_analysis.items()},
                "extraction_success_rate": {
                    source: (stats["papers_with_full_text"] / stats["total_papers"] * 100) if stats["total_papers"] > 0 else 0
                    for source, stats in source_analysis.items()
                },
                "avg_text_length": {source: stats["avg_text_length"] for source, stats in source_analysis.items()}
            }
        }
        
        return source_comparison
        
    except Exception as e:
        logger.error(f"Error generating source statistics: {e}")
        return {"error": str(e)}

@app.get("/pubmed-test/{compound_name}")
async def test_pubmed_extraction(compound_name: str):
    """Test endpoint specifically for PubMed extraction"""
    try:
        if not pubmed_scraper:
            return {"error": "PubMed scraper not initialized"}
        
        # Create compound info for testing
        compound_input = CompoundInput(name=compound_name, cas_number="")
        validated_compound = await process_compound_input(compound_input)
        
        compound_info = CompoundInfo(
            name=validated_compound.name,
            cas_number=validated_compound.cas_number,
            synonyms=validated_compound.synonyms
        )
        
        # Test PubMed search only
        results = await pubmed_scraper.search_compound(compound_info, max_results_per_query=5)
        
        # Get statistics
        stats = pubmed_scraper.get_summary_stats(results)
        
        # Format results for display
        return {
            "compound": compound_name,
            "pubmed_search": {
                "total_papers": len(results),
                "papers_with_full_text": sum(1 for r in results if r.full_text),
                "papers_with_abstract_only": sum(1 for r in results if not r.full_text and r.abstract),
                "extraction_methods": stats.get("extraction_methods", {})
            },
            "sample_results": [
                {
                    "title": r.title,
                    "abstract_preview": r.abstract[:200] + "..." if r.abstract else "No abstract",
                    "full_text_available": bool(r.full_text),
                    "text_length": len(r.full_text) if r.full_text else 0,
                    "extraction_method": r.extraction_method,
                    "link": r.link
                }
                for r in results[:3]  # Show first 3 results
            ],
            "search_queries": pubmed_scraper.generate_search_queries(compound_info)[:3],
            "extraction_success_rate": stats.get("extraction_success_rate", 0)
        }
        
    except Exception as e:
        logger.error(f"Error in PubMed test: {e}")
        return {"error": str(e)}

@app.get("/sources-config")
async def get_sources_configuration():
    """Get detailed configuration for all data sources"""
    return {
        "api_version": "3.0.0",
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
            "pubmed": {
                "description": "Specialized biomedical and life sciences database",
                "search_method": "E-utilities API with targeted queries",
                "content_extraction": {
                    "primary": "HTML extraction using Cheerio",
                    "fallback": "Abstract-only from E-utilities API"
                },
                "advantages": [
                    "Highly relevant medical/toxicology content",
                    "Reliable metadata",
                    "Official API access",
                    "PMC full text availability"
                ]
            }
        },
        "integration_logic": {
            "search_execution": "Parallel searches across all enabled sources",
            "result_processing": "Source-specific search result annotation",
            "deduplication": "Cross-source deduplication based on title similarity",
            "statistics": "Aggregated statistics with source-specific breakdowns"
        },
        "usage_recommendations": {
            "toxicology_research": "Use both sources for comprehensive coverage",
            "medical_focus": "Prioritize PubMed for medical/toxicology specific papers",
            "broad_research": "Prioritize Google Scholar for wider coverage"
        }
    }

@app.get("/compare-sources/{compound_name}")
async def compare_sources(compound_name: str):
    """Direct comparison between Google Scholar and PubMed results"""
    try:
        # Run searches for both sources
        request = CompoundRequest(
            compound_name=compound_name,
            search_sources=["google_scholar", "pubmed"],
            max_results_per_query=10
        )
        
        response = await search_compound(request)
        
        # Group results by source
        google_scholar_results = [r for r in response.search_results if r.source == "google_scholar"]
        pubmed_results = [r for r in response.search_results if r.source == "pubmed"]
        
        # Find overlapping papers (by title similarity)
        gs_titles = {r.title.lower().strip() for r in google_scholar_results}
        pm_titles = {r.title.lower().strip() for r in pubmed_results}
        
        overlap = gs_titles.intersection(pm_titles)
        
        # Compare content extraction success
        gs_extraction_rate = (
            sum(1 for r in google_scholar_results if r.content_extracted) / 
            max(1, len(google_scholar_results)) * 100
        )
        
        pm_extraction_rate = (
            sum(1 for r in pubmed_results if r.content_extracted) / 
            max(1, len(pubmed_results)) * 100
        )
        
        return {
            "compound": compound_name,
            "results_summary": {
                "google_scholar_papers": len(google_scholar_results),
                "pubmed_papers": len(pubmed_results),
                "unique_papers": len(set(gs_titles).union(pm_titles)),
                "overlapping_papers": len(overlap)
            },
            "content_extraction": {
                "google_scholar_extraction_rate": gs_extraction_rate,
                "pubmed_extraction_rate": pm_extraction_rate,
                "google_scholar_with_full_text": sum(1 for r in google_scholar_results if r.full_text),
                "pubmed_with_full_text": sum(1 for r in pubmed_results if r.full_text)
            },
            "year_distribution": {
                "google_scholar": get_year_distribution(google_scholar_results),
                "pubmed": get_year_distribution(pubmed_results)
            },
            "extraction_methods": {
                "google_scholar": get_extraction_methods(google_scholar_results),
                "pubmed": get_extraction_methods(pubmed_results)
            },
            "recommendation": get_source_recommendation(google_scholar_results, pubmed_results)
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

def get_source_recommendation(gs_results: List[SearchResult], pm_results: List[SearchResult]) -> Dict:
    """Generate a recommendation based on the results from both sources"""
    gs_full_text = sum(1 for r in gs_results if r.full_text)
    pm_full_text = sum(1 for r in pm_results if r.full_text)
    
    gs_extraction_rate = gs_full_text / max(1, len(gs_results)) * 100
    pm_extraction_rate = pm_full_text / max(1, len(pm_results)) * 100
    
    recommendations = {}
    
    # Paper availability recommendation
    if len(gs_results) > len(pm_results) * 1.5:
        recommendations["paper_availability"] = "google_scholar"
    elif len(pm_results) > len(gs_results) * 1.5:
        recommendations["paper_availability"] = "pubmed"
    else:
        recommendations["paper_availability"] = "both"
    
    # Content extraction recommendation
    if gs_extraction_rate > pm_extraction_rate * 1.5:
        recommendations["content_extraction"] = "google_scholar"
    elif pm_extraction_rate > gs_extraction_rate * 1.5:
        recommendations["content_extraction"] = "pubmed"
    else:
        recommendations["content_extraction"] = "both"
    
    # Overall recommendation
    if recommendations["paper_availability"] == recommendations["content_extraction"]:
        recommendations["overall"] = recommendations["paper_availability"]
    else:
        recommendations["overall"] = "both"
    
    return recommendations

if __name__ == "__main__":
    import uvicorn
    
    # Check configuration
    if not get_apify_token():
        print("WARNING: Apify token not configured!")
        print("Please set your token in config.py or as environment variable:")
        print("export APIFY_TOKEN='your_actual_token'")
        print("Get your token from: https://console.apify.com/")
    
    print("Starting Enhanced Toxicology Research API v3.0...")
    print("New features: Multi-source search (Google Scholar + PubMed)")
    print()
    print("API Documentation: http://localhost:8000/docs")
    print("Test endpoint: http://localhost:8000/test/limonene")
    print("Source statistics: http://localhost:8000/source-stats/limonene")
    print("PubMed specific test: http://localhost:8000/pubmed-test/limonene")
    print("Source comparison: http://localhost:8000/compare-sources/limonene")
    print("Sources config: http://localhost:8000/sources-config")
    
    # Run the API server
    uvicorn.run(app, host="0.0.0.0", port=8000)
