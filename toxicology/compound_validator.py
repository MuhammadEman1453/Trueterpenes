"""
Compound Input Validation & Synonym Retrieval
Validates compound input and fetches all synonyms using PubChem API
Purpose: Validates and enriches compound input using PubChem API
Flow:

Validation: Checks compound name and CAS number format
PubChem Lookup: Searches by name and/or CAS number
Synonym Retrieval: Fetches all known synonyms
Enrichment: Adds molecular formula, weight, IUPAC name
Output: Returns validated compound with complete metadata

Features:

✅ Smart CAS number formatting
✅ Multiple PubChem search strategies
✅ Comprehensive error handling
✅ Unicode and special character handling
"""

import re
import logging
import asyncio
import aiohttp
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import json
from urllib.parse import quote

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class CompoundInput:
    """Input from the 2-field form"""
    name: str
    cas_number: str

@dataclass
class ValidatedCompound:
    """Validated compound with synonyms"""
    name: str
    cas_number: str
    synonyms: List[str]
    iupac_name: Optional[str] = None
    molecular_formula: Optional[str] = None
    molecular_weight: Optional[float] = None
    pubchem_cid: Optional[int] = None

class CompoundValidator:
    """
    Validates compound input and retrieves synonyms from PubChem
    """
    
    def __init__(self):
        """Initialize the validator"""
        self.pubchem_base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
        self.session = None
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    def validate_cas_number(self, cas_number: str) -> Tuple[bool, str]:
        """
        Validate CAS number format
        
        Args:
            cas_number (str): CAS number to validate
            
        Returns:
            Tuple[bool, str]: (is_valid, cleaned_cas_number)
        """
        # Remove spaces and clean
        cleaned = cas_number.strip().replace(' ', '').replace('-', '')
        
        # Re-add dashes in correct format (XXXX-XX-X)
        if len(cleaned) >= 5:
            # Handle common patterns
            if len(cleaned) == 8:  # XXXXXXXX format
                formatted = f"{cleaned[:4]}-{cleaned[4:6]}-{cleaned[6:]}"
            elif len(cleaned) == 9:  # XXXXXXXXX format
                formatted = f"{cleaned[:5]}-{cleaned[5:7]}-{cleaned[7:]}"
            elif len(cleaned) == 10:  # XXXXXXXXXX format
                formatted = f"{cleaned[:6]}-{cleaned[6:8]}-{cleaned[8:]}"
            else:
                # If already has dashes, keep as is
                formatted = cas_number.strip()
        else:
            formatted = cas_number.strip()
        
        # Basic format validation (digits and dashes)
        pattern = r'^\d{2,7}-\d{2}-\d$'
        is_valid = bool(re.match(pattern, formatted))
        
        if not is_valid:
            logger.warning(f"Invalid CAS format: {cas_number}")
        
        return is_valid, formatted
    
    def validate_compound_name(self, name: str) -> Tuple[bool, str]:
        """
        Validate compound name
        
        Args:
            name (str): Compound name to validate
            
        Returns:
            Tuple[bool, str]: (is_valid, cleaned_name)
        """
        cleaned = name.strip()
        
        # Check if not empty and has valid characters
        if not cleaned:
            return False, ""
        
        # Remove excessive whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Must contain at least one letter
        if not re.search(r'[a-zA-Z]', cleaned):
            return False, cleaned
        
        logger.info(f"Validated compound name: {cleaned}")
        return True, cleaned
    
    async def get_synonyms_by_name(self, compound_name: str) -> Optional[Dict]:
        """
        Get compound information by name from PubChem
        
        Args:
            compound_name (str): Name of the compound
            
        Returns:
            Optional[Dict]: Compound information or None if not found
        """
        try:
            # URL encode the compound name
            encoded_name = quote(compound_name)
            url = f"{self.pubchem_base_url}/compound/name/{encoded_name}/property/MolecularFormula,MolecularWeight,IUPACName/JSON"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    compound_info = data['PropertyTable']['Properties'][0]
                    
                    # Get CID for synonym lookup
                    cid = compound_info.get('CID')
                    synonyms = await self.get_synonyms_by_cid(cid) if cid else []
                    
                    return {
                        'cid': cid,
                        'synonyms': synonyms,
                        'molecular_formula': compound_info.get('MolecularFormula'),
                        'molecular_weight': compound_info.get('MolecularWeight'),
                        'iupac_name': compound_info.get('IUPACName')
                    }
                else:
                    logger.warning(f"PubChem name search failed: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error searching by name '{compound_name}': {e}")
            return None
    
    async def get_synonyms_by_cas(self, cas_number: str) -> Optional[Dict]:
        """
        Get compound information by CAS number from PubChem
        
        Args:
            cas_number (str): CAS number
            
        Returns:
            Optional[Dict]: Compound information or None if not found
        """
        try:
            url = f"{self.pubchem_base_url}/compound/name/{cas_number}/property/MolecularFormula,MolecularWeight,IUPACName/JSON"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    compound_info = data['PropertyTable']['Properties'][0]
                    
                    # Get CID for synonym lookup
                    cid = compound_info.get('CID')
                    synonyms = await self.get_synonyms_by_cid(cid) if cid else []
                    
                    return {
                        'cid': cid,
                        'synonyms': synonyms,
                        'molecular_formula': compound_info.get('MolecularFormula'),
                        'molecular_weight': compound_info.get('MolecularWeight'),
                        'iupac_name': compound_info.get('IUPACName')
                    }
                else:
                    logger.warning(f"PubChem CAS search failed: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error searching by CAS '{cas_number}': {e}")
            return None
    
    async def get_synonyms_by_cid(self, cid: int) -> List[str]:
        """
        Get all synonyms for a compound by CID
        
        Args:
            cid (int): PubChem CID
            
        Returns:
            List[str]: List of synonyms
        """
        try:
            url = f"{self.pubchem_base_url}/compound/cid/{cid}/synonyms/JSON"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    synonyms = data['InformationList']['Information'][0]['Synonym']
                    
                    # Filter and clean synonyms
                    filtered_synonyms = []
                    for synonym in synonyms:
                        # Skip CAS numbers and very long names
                        if not re.match(r'^\d+-\d+-\d+$', synonym) and len(synonym) < 100:
                            filtered_synonyms.append(synonym)
                    
                    logger.info(f"Found {len(filtered_synonyms)} synonyms for CID {cid}")
                    return filtered_synonyms[:50]  # Limit to 50 synonyms
                else:
                    logger.warning(f"Synonym lookup failed for CID {cid}: {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error getting synonyms for CID {cid}: {e}")
            return []
    
    async def validate_and_enrich_compound(self, compound_input: CompoundInput) -> ValidatedCompound:
        """
        Validate input and enrich with synonyms
        
        Args:
            compound_input (CompoundInput): Input from form
            
        Returns:
            ValidatedCompound: Validated and enriched compound data
        """
        logger.info(f"Processing compound: {compound_input.name}, CAS: {compound_input.cas_number}")
        
        # Validate inputs
        name_valid, clean_name = self.validate_compound_name(compound_input.name)
        cas_valid, clean_cas = self.validate_cas_number(compound_input.cas_number)
        
        if not name_valid:
            raise ValueError(f"Invalid compound name: {compound_input.name}")
        
        # Try to get information from both name and CAS
        compound_info = None
        
        # First try by name
        if clean_name:
            compound_info = await self.get_synonyms_by_name(clean_name)
        
        # If name search failed or no name, try CAS
        if not compound_info and cas_valid:
            compound_info = await self.get_synonyms_by_cas(clean_cas)
        
        # Create result object
        validated = ValidatedCompound(
            name=clean_name,
            cas_number=clean_cas if cas_valid else "",
            synonyms=compound_info.get('synonyms', []) if compound_info else [],
            iupac_name=compound_info.get('iupac_name') if compound_info else None,
            molecular_formula=compound_info.get('molecular_formula') if compound_info else None,
            molecular_weight=compound_info.get('molecular_weight') if compound_info else None,
            pubchem_cid=compound_info.get('cid') if compound_info else None
        )
        
        # Add input name and CAS to synonyms if not already included
        if clean_name and clean_name not in validated.synonyms:
            validated.synonyms.insert(0, clean_name)
        if cas_valid and clean_cas not in validated.synonyms:
            validated.synonyms.insert(0, clean_cas)
        
        logger.info(f"Successfully validated compound with {len(validated.synonyms)} synonyms")
        return validated

# Main processing function
async def process_compound_input(compound_input: CompoundInput) -> ValidatedCompound:
    """
    Main function to process compound input and get synonyms
    
    Args:
        compound_input (CompoundInput): Input from form
        
    Returns:
        ValidatedCompound: Validated compound with synonyms
    """
    async with CompoundValidator() as validator:
        try:
            validated_compound = await validator.validate_and_enrich_compound(compound_input)
            return validated_compound
        except Exception as e:
            logger.error(f"Error processing compound: {e}")
            raise
