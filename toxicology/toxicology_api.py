"""
Toxicology Research API
Simple API that accepts compound name and CAS number, returns search results
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import asyncio
import logging
from datetime import datetime

# Import our modules
from compound_validator import CompoundInput, process_compound_input
from google_scholar_scraper import GoogleScholarScraper, CompoundInfo
from config import get_apify_token, DEFAULT_MAX_RESULTS_PER_QUERY

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Toxicology Research API",
    description="API for searching toxicology papers by compound",
    version="1.0.0"
)

# Pydantic models for API
class CompoundRequest(BaseModel):
    """Request model for compound search"""
    compound_name: Optional[str] = Field(None, description="Name of the compound")
    cas_number: Optional[str] = Field(None, description="CAS registry number")
    
    class Config:
        schema_extra = {
            "example": {
                "compound_name": "limonene",
                "cas_number": "5989-27-5"
            }
        }

class CompoundResponse(BaseModel):
    """Response model for compound search"""
    success: bool
    compound_info: Dict
    search_results: List[Dict]
    statistics: Dict
    execution_time: float
    timestamp: str

# Initialize scraper
scraper = None
apify_token = get_apify_token()
if apify_token:
    scraper = GoogleScholarScraper(apify_token)
    logger.info("GoogleScholarScraper initialized successfully")
else:
    logger.warning("Apify token not configured. API will return error on search requests.")

@app.on_event("startup")
async def startup_event():
    """Startup event to validate configuration"""
    if not scraper:
        logger.error("API started without valid Apify token")
    else:
        logger.info("Toxicology API started successfully")

@app.post("/search", response_model=CompoundResponse)
async def search_compound(request: CompoundRequest):
    """
    Search for toxicology papers for a given compound
    
    This endpoint:
    1. Validates the input compound
    2. Retrieves synonyms from PubChem
    3. Performs targeted searches on Google Scholar using Apify
    4. Returns filtered results with metadata
    
    Args:
        request: CompoundRequest with compound_name and/or cas_number
        
    Returns:
        CompoundResponse with search results and metadata
    """
    start_time = datetime.now()
    
    try:
        # Validate input
        if not request.compound_name and not request.cas_number:
            raise HTTPException(
                status_code=400, 
                detail="At least one of compound_name or cas_number must be provided"
            )
        
        # Check if scraper is initialized
        if not scraper:
            raise HTTPException(
                status_code=500,
                detail="Apify token not configured. Please set APIFY_TOKEN in config."
            )
        
        logger.info(f"Processing request: {request.compound_name}, CAS: {request.cas_number}")
        
        # Step 1: Validate compound and get synonyms
        compound_input = CompoundInput(
            name=request.compound_name or "",
            cas_number=request.cas_number or ""
        )
        
        validated_compound = await process_compound_input(compound_input)
        
        # Step 2: Search Google Scholar
        compound_info = CompoundInfo(
            name=validated_compound.name,
            cas_number=validated_compound.cas_number,
            synonyms=validated_compound.synonyms
        )
        
        # Perform search
        search_results = await scraper.search_compound(
            compound_info, 
            max_results_per_query=DEFAULT_MAX_RESULTS_PER_QUERY
        )
        
        # Get statistics
        stats = scraper.get_summary_stats(search_results)
        
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
                    "search_term": result.search_term
                }
                for result in search_results
            ],
            "statistics": stats,
            "execution_time": execution_time,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Successfully processed compound. Found {len(search_results)} papers in {execution_time:.2f}s")
        return response_data
        
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing compound: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    apify_configured = scraper is not None and get_apify_token() is not None
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "apify_configured": apify_configured,
        "apify_token_status": "configured" if get_apify_token() else "not_configured"
    }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Toxicology Research API",
        "version": "1.0.0",
        "endpoints": {
            "search": "/search (POST) - Search for toxicology papers",
            "health": "/health (GET) - Health check",
            "docs": "/docs - API documentation"
        },
        "search_strategy": "Targeted searches with keywords (not broad searches)"
    }

# Example usage for testing
@app.get("/test/{compound_name}")
async def test_search(compound_name: str, cas_number: Optional[str] = None):
    """
    Test endpoint for quick searches
    Usage: GET /test/limonene?cas_number=5989-27-5
    """
    request = CompoundRequest(compound_name=compound_name, cas_number=cas_number)
    return await search_compound(request)

if __name__ == "__main__":
    import uvicorn
    
    # Check configuration
    if not get_apify_token():
        print("WARNING: Apify token not configured!")
        print("Please set your token in config.py or as environment variable:")
        print("export APIFY_TOKEN='your_actual_token'")
        print("Get your token from: https://console.apify.com/")
    
    print("Starting Toxicology Research API...")
    print("API Documentation: http://localhost:8000/docs")
    print("Test endpoint: http://localhost:8000/test/limonene")
    
    # Run the API server
    uvicorn.run(app, host="0.0.0.0", port=8000)