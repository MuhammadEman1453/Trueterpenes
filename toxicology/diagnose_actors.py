"""
Apify Actor Diagnostic Tool (Fixed for new token)
Helps identify the correct actor ID and test configuration
"""

import asyncio
from apify_client import ApifyClient
import logging
from dotenv import load_dotenv
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

async def diagnose_apify_actors(apify_token: str):
    """
    Diagnose which Google Scholar actors are available
    """
    # Fixed: Only check if token is empty or placeholder
    if not apify_token or apify_token == "your_apify_token_here":
        print("ERROR: Please set a valid APIFY_TOKEN")
        print(f"Current token: {apify_token}")
        print("\nTo fix this:")
        print("1. Create a .env file with your token:")
        print("   echo 'APIFY_TOKEN=your_actual_token' > .env")
        print("2. Or set environment variable:")
        print("   export APIFY_TOKEN='your_actual_token'")
        return
    
    # Validate token format (basic check)
    if len(apify_token) < 20:
        print(f"WARNING: Token seems too short ({len(apify_token)} characters)")
        print("Apify tokens are usually much longer")
    
    client = ApifyClient(apify_token)
    
    print("=== Apify Actor Diagnostic Tool ===")
    print(f"Token: {apify_token[:8]}...{apify_token[-5:]}")
    print(f"Token length: {len(apify_token)} characters")
    
    # First, let's test if the token is valid by trying to get user info
    try:
        print("\n--- Testing Token Validity ---")
        # This is a simple way to test if the token works
        actors = client.actors().list()
        print("✓ Token is valid and working!")
    except Exception as e:
        print(f"✗ Token validation failed: {e}")
        print("This indicates your token might be incorrect or expired")
        return
    
    # Test different Google Scholar actors
    actors_to_test = [
        {
            "id": "apify/google-scholar-scraper",
            "name": "Official Apify Google Scholar Scraper",
            "test_input": {
                "searchQuery": "limonene",
                "maxResults": 5
            }
        },
        {
            "id": "davidl93/serpapi-scraper", 
            "name": "SerpAPI Scraper",
            "test_input": {
                "queries": ["limonene"],
                "searchType": "scholar",
                "numResults": 5
            }
        },
        {
            "id": "petr_cermak/google-scholar-scraper",
            "name": "Petr Cermak's Scholar Scraper",
            "test_input": {
                "queries": ["limonene"],
                "includeAbstracts": True,
                "maxCitations": 5
            }
        },
        {
            "id": "marco.gullo/google-scholar-scraper",
            "name": "Marco gullo Scholar Scraper",
            "test_input": {
                "queries": ["limonene"],
                "includeAbstracts": True,
                "maxCitations": 5
            }
        },

    ]
    
    working_actors = []
    
    for actor in actors_to_test:
        print(f"\n--- Testing: {actor['name']} ---")
        print(f"Actor ID: {actor['id']}")
        
        try:
            # Check if actor exists
            actor_info = client.actor(actor['id']).get()
            
            if actor_info:
                print("✓ Actor found")
                print(f"  Title: {actor_info.get('title', 'N/A')}")
                print(f"  Author: {actor_info.get('username', 'N/A')}")
                print(f"  Last Modified: {actor_info.get('modifiedAt', 'N/A')}")
                
                # Try a test run
                print("  Testing with small query...")
                try:
                    run = client.actor(actor['id']).call(run_input=actor['test_input'])
                    print("  ✓ Test run successful")
                    
                    # Check results
                    dataset = client.dataset(run["defaultDatasetId"])
                    items = list(dataset.iterate_items())
                    print(f"  ✓ Retrieved {len(items)} results")
                    
                    working_actors.append(actor)
                    
                except Exception as e:
                    print(f"  ✗ Test run failed: {e}")
                    
            else:
                print("✗ Actor not found")
                
        except Exception as e:
            print(f"✗ Error accessing actor: {e}")
    
    print(f"\n=== Summary ===")
    if working_actors:
        print("Working actors found:")
        for actor in working_actors:
            print(f"  - {actor['id']} ({actor['name']})")
        
        # Recommend the best one
        print(f"\nRecommendation: Use {working_actors[0]['id']}")
        
        # Generate code snippet
        print(f"\nUpdate your google_scholar_scraper.py:")
        print(f"  self.actor_id = \"{working_actors[0]['id']}\"")
        
    else:
        print("No working actors found. Please check:")
        print("1. Your Apify token is valid")
        print("2. Your account has access to these actors")
        print("3. Try searching Apify Store for 'google scholar' actors")

if __name__ == "__main__":
    # Test with your token
    import os
    
    # Try multiple ways to get the token
    token = None
    
    # Method 1: From environment variable
    token = os.getenv("APIFY_TOKEN")
    
    # Method 2: From config.py if available
    if not token:
        try:
            from config import get_apify_token
            token = get_apify_token()
        except ImportError:
            pass
    
    # Method 3: Direct from environment (alternative)
    if not token:
        token = os.environ.get("APIFY_TOKEN")
    
    print("=== Token Detection ===")
    print(f"Token found: {'Yes' if token else 'No'}")
    if token:
        print(f"Token source: Environment variable")
        print(f"Token preview: {token[:8]}...{token[-5:] if len(token) > 5 else '*'}")
    else:
        print("No token found in:")
        print("  - APIFY_TOKEN environment variable")
        print("  - .env file (if you have python-dotenv)")
        print("  - config.py file")
        print("\nTo set your token:")
        print("  export APIFY_TOKEN='fGI4fXFV0ezlArof3d8RbBQCRgcIGh3fwYpn'")
        print("  # OR create .env file:")
        print("  echo 'APIFY_TOKEN=fGI4fXFV0ezlArof3d8RbBQCRgcIGh3fwYpn' > .env")
    
    if token:
        asyncio.run(diagnose_apify_actors(token))
    else:
        print("\nPlease set your APIFY_TOKEN and try again.")