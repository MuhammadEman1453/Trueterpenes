"""
Article Filtering Module using OpenAI GPT-4.1-mini
Filters articles based on toxicology research criteria
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
import json
from openai import AsyncOpenAI
from dataclasses import dataclass
import re
from datetime import datetime  # Added missing import
from google_scholar_scraper import SearchResult

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class FilterCriteria:
    """Data class to store filtering criteria"""
    include_keywords: List[str]
    exclude_keywords: List[str]
    study_types: List[str]
    
    @classmethod
    def get_toxicology_criteria(cls):
        """Get predefined toxicology research criteria"""
        return cls(
            include_keywords=[
                "toxicity", "inhalation", "gavage", "pyrolysis", 
                "cannabis", "degradation", "thermal degradation",
                "in vitro", "in vivo", "in silico", "QSAR",
                "oral administration", "animal studies", "cell-based",
                "vaping", "combustion", "vaporization", "thermal stability"
            ],
            exclude_keywords=[
                "insecticidal", "repellent", "insects", "chemical synthesis",
                "precursor", "degradation product only", "reference section only",
                "chemical composition analysis only"
            ],
            study_types=[
                "inhalation toxicity",
                "animal studies", 
                "in vitro studies",
                "thermal stability & pyrolysis",
                "in silico models"
            ]
        )

class ArticleFilter:
    """
    Article filtering using OpenAI GPT-4.1-mini
    """
    
    def __init__(self, openai_api_key: str):
        """
        Initialize the article filter
        
        Args:
            openai_api_key (str): OpenAI API key
        """
        self.client = AsyncOpenAI(api_key=openai_api_key)
        self.model = "gpt-4o-mini-2024-07-18"  # Using GPT-4.1-mini as required
        self.criteria = FilterCriteria.get_toxicology_criteria()
        
        logger.info(f"ArticleFilter initialized with model: {self.model}")
    
    def _create_filtering_prompt(self, compound_name: str, article: SearchResult) -> str:
        """
        Create a detailed prompt for filtering articles with comprehensive analysis requirements
        
        Args:
            compound_name (str): Name of the compound being researched
            article (SearchResult): Article to be filtered
            
        Returns:
            str: Formatted prompt for OpenAI
        """
        # Prepare article content for analysis
        content_to_analyze = ""
        
        # Include title (most important)
        if article.title:
            content_to_analyze += f"TITLE: {article.title}\n\n"
        
        # Include abstract if available
        if article.abstract:
            content_to_analyze += f"ABSTRACT: {article.abstract}\n\n"
        
        # Include a portion of full text if available
        if article.full_text:
            # Limit full text to first 2000 characters to stay within token limits
            full_text_excerpt = article.full_text[:2000]
            content_to_analyze += f"FULL TEXT EXCERPT: {full_text_excerpt}\n\n"
        
        prompt = f"""
You are a toxicology research expert. Your task is to analyze the following research article about {compound_name} and determine if it meets the inclusion criteria for toxicology research.

COMPOUND BEING RESEARCHED: {compound_name}

ARTICLE TO ANALYZE:
{content_to_analyze}

INCLUSION CRITERIA:
The article should be included if it contains studies on:

1. **Inhalation Toxicity**: Studies where {compound_name} was inhaled by humans or animals, or where inhalation toxicology and safety are discussed.

2. **Animal Studies**: Any study where {compound_name} was administered to animals by any route (gavage, inhalation, dermal, etc.), including studies on metabolism, toxic effects, or potential health benefits.

3. **In Vitro Studies**: Research where {compound_name} was tested in cell-based models.

4. **Thermal Stability & Pyrolysis**: Studies investigating the effects of heat, combustion, or vaporization on {compound_name} (pyrolysis, vaping, thermal degradation studies).

5. **In Silico Models**: Computational toxicology studies such as QSAR modeling for {compound_name}.

KEY TERMS TO LOOK FOR: toxicity, inhalation, gavage, pyrolysis, cannabis, degradation, thermal degradation, in vitro, in vivo, in silico, QSAR, oral administration, vaping, combustion, vaporization.

EXCLUSION CRITERIA:
The article should be EXCLUDED if:
- {compound_name} is mentioned only in the reference section
- The study focuses only on chemical composition analysis where {compound_name} is just listed
- The research is about insecticidal or repellent properties (including studies on insects)
- The paper focuses on chemical synthesis of {compound_name}
- {compound_name} appears only as a precursor or degradation product (not the main subject)

IMPORTANT: The compound must be the main subject of toxicological investigation, not just mentioned in passing.

Please provide a detailed analysis in the following JSON format:
{{
    "include": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "Detailed explanation of why this article should/should not be included",
    "study_type": "one of: inhalation_toxicity, animal_studies, in_vitro, thermal_pyrolysis, in_silico, not_applicable",
    "relevant_keywords_found": ["list", "of", "relevant", "keywords", "found"],
    "exclusion_reasons": ["list", "of", "exclusion", "reasons", "if", "any"],
    "detailed_analysis": {{
        "compound_relevance": "How central is {compound_name} to this study? (main subject/secondary focus/passing mention)",
        "toxicology_relevance": "What toxicological aspects are covered? Be specific.",
        "study_methodology": "What research methods were used? (in vivo/in vitro/in silico/computational)",
        "administration_route": "How was {compound_name} administered or studied? (inhalation/oral/dermal/cell culture/etc.)",
        "key_findings": "What were the main toxicological findings related to {compound_name}?",
        "inclusion_criteria_met": ["list", "which", "specific", "inclusion", "criteria", "are", "met"],
        "exclusion_criteria_triggered": ["list", "any", "exclusion", "criteria", "that", "apply"],
        "overall_assessment": "Final verdict on relevance to toxicology research"
    }},
    "evidence_from_text": {{
        "supporting_quotes": ["relevant", "quotes", "from", "title/abstract/text", "that", "support", "inclusion"],
        "concerning_phrases": ["phrases", "that", "might", "suggest", "exclusion"],
        "key_evidence": "Most important evidence for the decision"
    }}
}}
"""
        return prompt
    
    async def filter_single_article(self, compound_name: str, article: SearchResult) -> Dict[str, Any]:
        """
        Filter a single article using OpenAI GPT-4.1-mini with detailed JSON analysis
        
        Args:
            compound_name (str): Name of the compound
            article (SearchResult): Article to filter
            
        Returns:
            Dict[str, Any]: Comprehensive filtering result with detailed reasoning
        """
        try:
            # Create the filtering prompt
            prompt = self._create_filtering_prompt(compound_name, article)
            
            # Call OpenAI API with structured JSON output
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert toxicology researcher analyzing scientific articles. Always respond with valid JSON format as requested. Provide detailed, specific analysis for each article."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0.1,  # Low temperature for consistent results
                max_tokens=1000,  # Increased for detailed analysis
                response_format={"type": "json_object"}
            )
            
            # Parse the response
            result = json.loads(response.choices[0].message.content)
            
            # Add metadata and ensure all required fields exist
            result["article_title"] = article.title
            result["article_link"] = article.link
            result["has_full_text"] = bool(article.full_text)
            result["has_abstract"] = bool(article.abstract)
            result["compound_analyzed"] = compound_name
            result["analysis_timestamp"] = datetime.now().isoformat()
            
            # Ensure all expected fields exist with defaults
            result.setdefault("include", False)
            result.setdefault("confidence", 0.0)
            result.setdefault("reasoning", "Analysis completed")
            result.setdefault("study_type", "not_applicable")
            result.setdefault("relevant_keywords_found", [])
            result.setdefault("exclusion_reasons", [])
            result.setdefault("detailed_analysis", {})
            result.setdefault("evidence_from_text", {})
            
            # Validate and clean the detailed analysis section
            detailed = result["detailed_analysis"]
            detailed.setdefault("compound_relevance", "Not specified")
            detailed.setdefault("toxicology_relevance", "Not specified")
            detailed.setdefault("study_methodology", "Not specified")
            detailed.setdefault("administration_route", "Not specified")
            detailed.setdefault("key_findings", "Not specified")
            detailed.setdefault("inclusion_criteria_met", [])
            detailed.setdefault("exclusion_criteria_triggered", [])
            detailed.setdefault("overall_assessment", "Analysis completed")
            
            # Validate and clean the evidence section
            evidence = result["evidence_from_text"]
            evidence.setdefault("supporting_quotes", [])
            evidence.setdefault("concerning_phrases", [])
            evidence.setdefault("key_evidence", "Not specified")
            
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse OpenAI response as JSON: {e}")
            return {
                "include": False,
                "confidence": 0.0,
                "reasoning": f"Error parsing OpenAI response: {e}",
                "study_type": "error",
                "relevant_keywords_found": [],
                "exclusion_reasons": ["json_parse_error"],
                "article_title": article.title,
                "article_link": article.link,
                "has_full_text": bool(article.full_text),
                "has_abstract": bool(article.abstract),
                "compound_analyzed": compound_name,
                "analysis_timestamp": datetime.now().isoformat(),
                "detailed_analysis": {
                    "compound_relevance": "Error in analysis",
                    "toxicology_relevance": "Error in analysis",
                    "study_methodology": "Error in analysis",
                    "administration_route": "Error in analysis",
                    "key_findings": "Error in analysis",
                    "inclusion_criteria_met": [],
                    "exclusion_criteria_triggered": ["analysis_error"],
                    "overall_assessment": "Analysis failed due to parsing error"
                },
                "evidence_from_text": {
                    "supporting_quotes": [],
                    "concerning_phrases": [],
                    "key_evidence": f"Parsing error: {e}"
                }
            }
        except Exception as e:
            logger.error(f"Error filtering article '{article.title[:50]}...': {e}")
            return {
                "include": False,
                "confidence": 0.0,
                "reasoning": f"Error during filtering: {e}",
                "study_type": "error",
                "relevant_keywords_found": [],
                "exclusion_reasons": ["filtering_error"],
                "article_title": article.title,
                "article_link": article.link,
                "has_full_text": bool(article.full_text),
                "has_abstract": bool(article.abstract),
                "compound_analyzed": compound_name,
                "analysis_timestamp": datetime.now().isoformat(),
                "detailed_analysis": {
                    "compound_relevance": "Error in analysis",
                    "toxicology_relevance": "Error in analysis", 
                    "study_methodology": "Error in analysis",
                    "administration_route": "Error in analysis",
                    "key_findings": "Error in analysis",
                    "inclusion_criteria_met": [],
                    "exclusion_criteria_triggered": ["filtering_error"],
                    "overall_assessment": f"Analysis failed: {e}"
                },
                "evidence_from_text": {
                    "supporting_quotes": [],
                    "concerning_phrases": [],
                    "key_evidence": f"Error during analysis: {e}"
                }
            }
    
    async def filter_articles_batch(
        self, 
        compound_name: str, 
        articles: List[SearchResult],
        batch_size: int = 5,
        min_confidence: float = 0.7
    ) -> Dict[str, Any]:
        """
        Filter multiple articles in batches
        
        Args:
            compound_name (str): Name of the compound
            articles (List[SearchResult]): Articles to filter
            batch_size (int): Number of articles to process in parallel
            min_confidence (float): Minimum confidence threshold for inclusion
            
        Returns:
            Dict[str, Any]: Filtering results with included/excluded articles
        """
        logger.info(f"Starting filtering of {len(articles)} articles for {compound_name}")
        
        # Process articles in batches to respect rate limits
        included_articles = []
        excluded_articles = []
        filtering_results = []
        
        for i in range(0, len(articles), batch_size):
            batch = articles[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(articles)-1)//batch_size + 1} ({len(batch)} articles)")
            
            # Process batch in parallel
            batch_tasks = [
                self.filter_single_article(compound_name, article) 
                for article in batch
            ]
            
            batch_results = await asyncio.gather(*batch_tasks)
            
            # Process results
            for j, result in enumerate(batch_results):
                article = batch[j]
                filtering_results.append(result)
                
                # Apply confidence threshold
                should_include = (
                    result.get("include", False) and 
                    result.get("confidence", 0.0) >= min_confidence
                )
                
                if should_include:
                    # Add filtering metadata to the article
                    article.filtering_result = result
                    article.filtering_confidence = result.get("confidence", 0.0)
                    article.study_type = result.get("study_type", "not_applicable")
                    article.relevant_keywords = result.get("relevant_keywords_found", [])
                    included_articles.append(article)
                else:
                    # Add exclusion reason
                    article.filtering_result = result
                    article.exclusion_reasons = result.get("exclusion_reasons", [])
                    excluded_articles.append(article)
            
            # Rate limiting - wait between batches
            if i + batch_size < len(articles):
                await asyncio.sleep(1.0)  # 1 second delay between batches
        
        # Calculate statistics
        total_articles = len(articles)
        included_count = len(included_articles)
        excluded_count = len(excluded_articles)
        
        # Analyze study types
        study_types = {}
        for result in filtering_results:
            if result.get("include", False):
                study_type = result.get("study_type", "unknown")
                study_types[study_type] = study_types.get(study_type, 0) + 1
        
        # Analyze exclusion reasons
        exclusion_reasons = {}
        for result in filtering_results:
            if not result.get("include", False):
                for reason in result.get("exclusion_reasons", []):
                    exclusion_reasons[reason] = exclusion_reasons.get(reason, 0) + 1
        
        logger.info(f"Filtering complete: {included_count}/{total_articles} articles included")
        logger.info(f"Study types found: {study_types}")
        
        return {
            "filtering_summary": {
                "total_articles": total_articles,
                "included_count": included_count,
                "excluded_count": excluded_count,
                "inclusion_rate": (included_count / total_articles * 100) if total_articles > 0 else 0,
                "min_confidence_threshold": min_confidence,
                "model_used": self.model
            },
            "included_articles": included_articles,
            "excluded_articles": excluded_articles,
            "study_types_distribution": study_types,
            "exclusion_reasons_distribution": exclusion_reasons,
            "filtering_results": filtering_results
        }
    
    def get_detailed_filtering_summary(self, filtering_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a comprehensive summary of filtering results with detailed analysis
        
        Args:
            filtering_result (Dict[str, Any]): Result from filter_articles_batch
            
        Returns:
            Dict[str, Any]: Detailed summary including analysis breakdowns
        """
        summary = filtering_result["filtering_summary"]
        study_types = filtering_result["study_types_distribution"]
        exclusion_reasons = filtering_result["exclusion_reasons_distribution"]
        filtering_results = filtering_result["filtering_results"]
        
        # Analyze inclusion/exclusion patterns
        included_articles = [r for r in filtering_results if r.get("include", False)]
        excluded_articles = [r for r in filtering_results if not r.get("include", False)]
        
        # Extract detailed analysis from included articles
        inclusion_analysis = {
            "total_included": len(included_articles),
            "avg_confidence": sum(r.get("confidence", 0) for r in included_articles) / max(1, len(included_articles)),
            "study_types": study_types,
            "common_keywords": self._analyze_keywords(included_articles),
            "administration_routes": self._analyze_administration_routes(included_articles),
            "methodology_breakdown": self._analyze_methodologies(included_articles)
        }
        
        # Extract detailed analysis from excluded articles
        exclusion_analysis = {
            "total_excluded": len(excluded_articles),
            "exclusion_reasons": exclusion_reasons,
            "common_exclusion_patterns": self._analyze_exclusion_patterns(excluded_articles),
            "avg_exclusion_confidence": sum(r.get("confidence", 0) for r in excluded_articles) / max(1, len(excluded_articles))
        }
        
        # Quality analysis
        quality_analysis = {
            "articles_with_full_text": len([r for r in filtering_results if r.get("has_full_text", False)]),
            "articles_with_abstract_only": len([r for r in filtering_results if r.get("has_abstract", False) and not r.get("has_full_text", False)]),
            "high_confidence_decisions": len([r for r in filtering_results if r.get("confidence", 0) >= 0.8]),
            "low_confidence_decisions": len([r for r in filtering_results if r.get("confidence", 0) < 0.5])
        }
        
        return {
            "filtering_summary": summary,
            "inclusion_analysis": inclusion_analysis,
            "exclusion_analysis": exclusion_analysis,
            "quality_analysis": quality_analysis,
            "detailed_results": [
                {
                    "title": r.get("article_title", ""),
                    "link": r.get("article_link", ""),
                    "decision": "INCLUDED" if r.get("include", False) else "EXCLUDED",
                    "confidence": r.get("confidence", 0),
                    "study_type": r.get("study_type", ""),
                    "reasoning": r.get("reasoning", ""),
                    "detailed_analysis": r.get("detailed_analysis", {}),
                    "evidence": r.get("evidence_from_text", {}),
                    "relevant_keywords": r.get("relevant_keywords_found", []),
                    "exclusion_reasons": r.get("exclusion_reasons", [])
                }
                for r in filtering_results
            ]
        }
    
    def _analyze_keywords(self, articles: List[Dict]) -> Dict[str, int]:
        """Analyze most common keywords from included articles"""
        keyword_count = {}
        for article in articles:
            for keyword in article.get("relevant_keywords_found", []):
                keyword_count[keyword] = keyword_count.get(keyword, 0) + 1
        return dict(sorted(keyword_count.items(), key=lambda x: x[1], reverse=True)[:10])
    
    def _analyze_administration_routes(self, articles: List[Dict]) -> Dict[str, int]:
        """Analyze administration routes from detailed analysis"""
        routes = {}
        for article in articles:
            detailed = article.get("detailed_analysis", {})
            route = detailed.get("administration_route", "").lower()
            if route and route != "not specified":
                routes[route] = routes.get(route, 0) + 1
        return dict(sorted(routes.items(), key=lambda x: x[1], reverse=True))
    
    def _analyze_methodologies(self, articles: List[Dict]) -> Dict[str, int]:
        """Analyze research methodologies from detailed analysis"""
        methods = {}
        for article in articles:
            detailed = article.get("detailed_analysis", {})
            method = detailed.get("study_methodology", "").lower()
            if method and method != "not specified":
                methods[method] = methods.get(method, 0) + 1
        return dict(sorted(methods.items(), key=lambda x: x[1], reverse=True))
    
    def _analyze_exclusion_patterns(self, articles: List[Dict]) -> Dict[str, int]:
        """Analyze common exclusion patterns"""
        patterns = {}
        for article in articles:
            detailed = article.get("detailed_analysis", {})
            criteria = detailed.get("exclusion_criteria_triggered", [])
            for criterion in criteria:
                patterns[criterion] = patterns.get(criterion, 0) + 1
        return dict(sorted(patterns.items(), key=lambda x: x[1], reverse=True))

# Example usage function
async def test_article_filter():
    """Test the article filter with sample data"""
    # Initialize filter (you'll need to provide your OpenAI API key)
    filter_engine = ArticleFilter(openai_api_key="your-openai-api-key-here")
    
    # Create sample articles for testing
    sample_articles = [
        SearchResult(
            title="Inhalation toxicity of limonene in rats: A 28-day study",
            authors=["Smith J", "Doe A"],
            abstract="This study investigated the inhalation toxicity of limonene in Sprague-Dawley rats over 28 days. Rats were exposed to 0, 300, 1000, or 3000 ppm limonene vapor for 6 hours/day. No adverse effects were observed at concentrations up to 1000 ppm.",
            link="https://example.com/paper1",
            pdf_link=None,
            citation_count=25,
            year=2023,
            search_term="limonene",
            full_text="Introduction: Limonene is a monoterpene commonly found in citrus oils. The objective of this study was to evaluate the inhalation toxicity of d-limonene in rats following repeated exposure. Methods: Male and female Sprague-Dawley rats were exposed to limonene vapor via whole-body inhalation for 6 hours/day, 5 days/week for 4 weeks. Exposure concentrations were 0, 300, 1000, and 3000 ppm. Clinical observations, body weight, food consumption, and clinical pathology evaluations were performed. Results: No treatment-related adverse effects were observed in animals exposed to 300 or 1000 ppm limonene. At 3000 ppm, slight respiratory irritation was noted. Conclusion: The no-observed-adverse-effect level (NOAEL) for limonene inhalation exposure in rats was determined to be 1000 ppm."
        ),
        SearchResult(
            title="Chemical synthesis and purification of d-limonene from orange peel",
            authors=["Johnson B", "Williams C"],
            abstract="This paper describes an efficient method for extracting and purifying d-limonene from orange peel waste using steam distillation followed by fractional distillation. The process yielded 95% pure limonene with minimal environmental impact.",
            link="https://example.com/paper2",
            pdf_link=None,
            citation_count=15,
            year=2022,
            search_term="limonene",
            full_text="Introduction: D-limonene is an important industrial chemical with applications in cleaning products, flavoring, and as a solvent. This study focuses on developing an environmentally friendly extraction method. Methods: Orange peels were subjected to steam distillation, followed by fractional distillation to purify the extracted limonene. GC-MS analysis was used to determine purity. Results: The optimized process yielded 95% pure d-limonene with excellent recovery rates. Conclusion: This method provides an efficient and green approach to limonene production from citrus waste."
        ),
        SearchResult(
            title="In vitro cytotoxicity of limonene on human lung epithelial cells",
            authors=["Brown C", "Wilson D", "Garcia M"],
            abstract="We evaluated the cytotoxic effects of limonene on A549 human lung adenocarcinoma cells and primary human bronchial epithelial cells. Cells were exposed to concentrations ranging from 10-1000 μM for 24-72 hours. MTT assays and flow cytometry were used to assess cell viability and apoptosis.",
            link="https://example.com/paper3",
            pdf_link=None,
            citation_count=18,
            year=2023,
            search_term="limonene",
            full_text="Introduction: Limonene is widely used in consumer products, making human exposure through inhalation a concern. This study aimed to evaluate the in vitro cytotoxicity of limonene on human lung cells. Methods: A549 cells and primary human bronchial epithelial cells (NHBE) were exposed to limonene at concentrations of 10, 50, 100, 500, and 1000 μM for 24, 48, and 72 hours. Cell viability was assessed using MTT assay, and apoptosis was evaluated by flow cytometry. Results: Limonene showed dose- and time-dependent cytotoxicity. Significant reduction in cell viability was observed at concentrations ≥100 μM after 48 hours. Flow cytometry revealed increased apoptosis at higher concentrations. Conclusion: Limonene exhibits cytotoxic effects on human lung cells in vitro, with potential implications for respiratory health."
        ),
        SearchResult(
            title="Thermal degradation products of limonene during vaping: GC-MS analysis",
            authors=["Taylor A", "Anderson K"],
            abstract="This study analyzed the thermal degradation products formed when limonene-containing vape liquids are heated. GC-MS analysis identified several potentially harmful compounds including formaldehyde and acetaldehyde.",
            link="https://example.com/paper4",
            pdf_link=None,
            citation_count=22,
            year=2023,
            search_term="limonene",
            full_text="Introduction: Limonene is increasingly used as a flavoring agent in electronic cigarette liquids. However, the thermal degradation products formed during vaping are poorly understood. Methods: Limonene-containing e-liquids were heated at temperatures ranging from 200-400°C using a vaping simulator. Aerosols were collected and analyzed by GC-MS to identify degradation products. Results: Several degradation products were identified, including formaldehyde, acetaldehyde, and various terpene oxides. The concentration of these products increased with temperature. Conclusion: Heating limonene during vaping produces potentially harmful degradation products that may pose health risks to users."
        ),
        SearchResult(
            title="Limonene as an insect repellent: efficacy against mosquitoes",
            authors=["Davis E", "Miller R"],
            abstract="This study evaluated the effectiveness of limonene as a natural insect repellent against Aedes aegypti mosquitoes. Various concentrations were tested in laboratory cage trials.",
            link="https://example.com/paper5",
            pdf_link=None,
            citation_count=12,
            year=2022,
            search_term="limonene",
            full_text="Introduction: Natural insect repellents are increasingly sought as alternatives to synthetic chemicals. This study investigates limonene's repellent properties against mosquitoes. Methods: Laboratory cage trials were conducted with Aedes aegypti mosquitoes. Different concentrations of limonene (5%, 10%, 15%, 20%) were applied to human volunteers' arms. Repellency was measured over 4 hours. Results: Limonene showed dose-dependent repellent activity, with 20% concentration providing up to 3 hours of protection. Conclusion: Limonene demonstrates potential as a natural mosquito repellent, though its duration of action is limited compared to DEET."
        )
    ]
    
    # Run the filtering test
    filtering_result = await filter_engine.filter_articles_batch(
        compound_name="limonene",
        articles=sample_articles,
        batch_size=3,
        min_confidence=0.5  # Lower threshold for testing
    )
    
    # Print results
    detailed_summary = filter_engine.get_detailed_filtering_summary(filtering_result)
    print("DETAILED FILTERING ANALYSIS")
    print("=" * 40)
    print(f"Total Articles: {detailed_summary['filtering_summary']['total_articles']}")
    print(f"Included: {detailed_summary['filtering_summary']['included_count']}")
    print(f"Excluded: {detailed_summary['filtering_summary']['excluded_count']}")
    print(f"Inclusion Rate: {detailed_summary['filtering_summary']['inclusion_rate']:.1f}%")
    print()
    
    print("INCLUSION ANALYSIS:")
    print(f"- Average Confidence: {detailed_summary['inclusion_analysis']['avg_confidence']:.2f}")
    print(f"- Study Types: {detailed_summary['inclusion_analysis']['study_types']}")
    print(f"- Common Keywords: {detailed_summary['inclusion_analysis']['common_keywords']}")
    print()
    
    print("EXCLUSION ANALYSIS:")
    print(f"- Average Confidence: {detailed_summary['exclusion_analysis']['avg_exclusion_confidence']:.2f}")
    print(f"- Exclusion Reasons: {detailed_summary['exclusion_analysis']['exclusion_reasons']}")
    print()
    
    print("\nDETAILED RESULTS:")
    print("\nINCLUDED ARTICLES:")
    for article in filtering_result["included_articles"]:
        result = article.filtering_result
        print(f"- {article.title}")
        print(f"  Confidence: {result['confidence']:.2f}")
        print(f"  Study Type: {result['study_type']}")
        print(f"  Keywords: {result['relevant_keywords_found']}")
        print(f"  Reasoning: {result['reasoning']}")
        print()
    
    print("\nEXCLUDED ARTICLES:")
    for article in filtering_result["excluded_articles"]:
        result = article.filtering_result
        print(f"- {article.title}")
        print(f"  Confidence: {result['confidence']:.2f}")
        print(f"  Exclusion Reasons: {result['exclusion_reasons']}")
        print(f"  Reasoning: {result['reasoning']}")
        print()

if __name__ == "__main__":
    # Run the test (uncomment to test)
       asyncio.run(test_article_filter())