"""
Configuration file for Toxicology Research API
"""

import os
from typing import List
from dotenv import load_dotenv
# API Configuration
API_HOST = "0.0.0.0"
API_PORT = 8000
API_WORKERS = 1
load_dotenv()
# Apify Configuration
APIFY_TOKEN = os.getenv("APIFY_TOKEN", "apify_api_7o1OPiuoAcBTSuk4d1sFalIubXBexM2Bnlnt")

# Google Scholar Scraper Settings
GOOGLE_SCHOLAR_ACTOR_ID = "petr_cermak/google-scholar-scraper"
DEFAULT_MAX_RESULTS_PER_QUERY = 20
BATCH_SIZE = 5  # Number of queries to run in parallel
RATE_LIMIT_DELAY = 2  # Seconds to wait between batches

# PubChem API Settings
PUBCHEM_BASE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
PUBCHEM_RETRY_DELAY = 1  # Seconds to wait between retries
PUBCHEM_MAX_RETRIES = 3

# Toxicology Keywords for Targeted Searches
TOXICOLOGY_KEYWORDS = [
    'toxicity',
    'inhalation',
    'gavage',
    'degradation',
    'pyrolysis',
    'vape',
    'cannabis'
]

STUDY_TYPE_KEYWORDS = [
    'in vitro',
    'in vivo',
    'in silico',
    'QSAR',
    'thermal stability'
]

# File Paths
OUTPUT_DIRECTORY = "output"
LOG_DIRECTORY = "logs"
TEMP_DIRECTORY = "temp"

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# API Response Limits
MAX_SYNONYMS_RETURNED = 50
MAX_SEARCH_RESULTS_RETURNED = 100

# Validation settings
CAS_NUMBER_PATTERN = r'^\d{2,7}-\d{2}-\d$'
MIN_COMPOUND_NAME_LENGTH = 2
MAX_COMPOUND_NAME_LENGTH = 100

# Error messages
ERROR_MESSAGES = {
    'missing_input': 'At least one of compound_name or cas_number must be provided',
    'invalid_cas': 'Invalid CAS number format',
    'invalid_name': 'Invalid compound name',
    'apify_not_configured': 'Apify token not configured',
    'search_failed': 'Failed to search for compound',
    'validation_failed': 'Failed to validate compound'
}

def get_apify_token() -> str:
    """
    Get Apify token from environment or config
    
    Returns:
        str: Apify token or None if not configured
    """
    token = os.getenv("APIFY_TOKEN", APIFY_TOKEN)
    if token == "your_apify_token_here":
        return None
    return token

def validate_configuration() -> List[str]:
    """
    Validate configuration settings
    
    Returns:
        List[str]: List of validation errors
    """
    errors = []
    
    if not get_apify_token():
        errors.append("APIFY_TOKEN is not configured")
    
    if BATCH_SIZE < 1 or BATCH_SIZE > 10:
        errors.append("BATCH_SIZE should be between 1 and 10")
    
    if DEFAULT_MAX_RESULTS_PER_QUERY < 1 or DEFAULT_MAX_RESULTS_PER_QUERY > 100:
        errors.append("DEFAULT_MAX_RESULTS_PER_QUERY should be between 1 and 100")
    
    return errors