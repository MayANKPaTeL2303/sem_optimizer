import os
import yaml
import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
import random
import re
import google.generativeai as genai
import logging
from typing import List, Dict, Any, Optional, Tuple, TypedDict
from dataclasses import dataclass
from google.ads.googleads.client import GoogleAdsClient
from google.ads.googleads.errors import GoogleAdsException
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebsiteContent(TypedDict):
    url: str
    title: str
    meta_description: str
    headings: List[str]
    content: str
    word_count: int
    error: Optional[str]

@dataclass
class KeywordData:
    keyword: str
    avg_monthly_searches: int
    competition: str
    top_page_bid_low: float
    top_page_bid_high: float
    competition_score: float = 0.0

@dataclass
class RealKeywordData:
    keyword: str
    avg_monthly_searches: int
    competition: str
    competition_index: float
    low_top_of_page_bid_micros: int
    high_top_of_page_bid_micros: int
    
    @property
    def top_page_bid_low(self) -> float:
        """Convert micros to dollars"""
        return self.low_top_of_page_bid_micros / 1_000_000
    
    @property 
    def top_page_bid_high(self) -> float:
        """Convert micros to dollars"""
        return self.high_top_of_page_bid_micros / 1_000_000

@dataclass
class AdGroup:
    name: str
    keywords: List[Dict[str, Any]]
    suggested_cpc_range: Tuple[float, float]
    match_types: List[str]

class GoogleKeywordPlannerAPI:
    """Real Google Keyword Planner API integration using Google Ads API"""
    
    def __init__(self, config_path: str = None, service_locations: List[str] = None, language_id: str = "1000"):
        """
        Initialize Google Ads API client
        
        Args:
            config_path: Path to google-ads.yaml config file
            service_locations: List of service locations for geo targeting
            language_id: Google Ads language ID (default: 1000 = English)
        """
        self.client = None
        self.customer_id = None
        self.service_locations = service_locations or []
        self.language_id = language_id
        
        try:
            # Initialize Google Ads client
            if config_path:
                self.client = GoogleAdsClient.load_from_storage(path=config_path)
            else:
                self.client = GoogleAdsClient.load_from_storage()
            
            self.customer_id = os.getenv('GOOGLE_ADS_CUSTOMER_ID')
            if not self.customer_id:
                raise ValueError("GOOGLE_ADS_CUSTOMER_ID environment variable not set")
                
            logger.info("Google Ads API client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Google Ads API: {e}")
            logger.error("Falling back to simulation mode")
            self.client = None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def get_geo_target_ids(self) -> List[str]:
        """Map service locations to geo target constant IDs."""
        if not self.client or not self.service_locations:
            return ["2840"]  # Default to US if no locations or client
        
        try:
            geo_service = self.client.get_service("GeoTargetConstantService")
            request = self.client.get_type("SuggestGeoTargetConstantsRequest")
            request.locale = "en"
            request.country_code = None  # Dynamic based on locations
            request.location_names.names.extend(self.service_locations)
            
            response = geo_service.suggest_geo_target_constants(request)
            location_ids = [gtc.geo_target_constant.split('/')[-1] for gtc in response.geo_target_constant_suggestions]
            if not location_ids:
                logger.warning("No geo target IDs found, defaulting to US")
                return ["2840"]
            logger.info(f"Mapped {len(location_ids)} geo target IDs")
            return location_ids
        except Exception as e:
            logger.error(f"Failed to get geo target IDs: {e}")
            return ["2840"]

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def get_keyword_ideas(self, 
                         seed_keywords: List[str] = None, 
                         urls: List[str] = None,
                         include_adult_keywords: bool = False) -> List[RealKeywordData]:
        """
        Get keyword ideas from Google Keyword Planner using keywords or URLs
        
        Args:
            seed_keywords: List of seed keywords
            urls: List of URLs for UrlSeed
            include_adult_keywords: Whether to include adult keywords
            
        Returns:
            List of RealKeywordData objects
        """
        if not self.client:
            logger.error("Google Ads API client not initialized")
            return []
        
        try:
            keyword_plan_idea_service = self.client.get_service("KeywordPlanIdeaService")
            request = self.client.get_type("GenerateKeywordIdeasRequest")
            request.customer_id = self.customer_id
            
            # Set language
            language = self.client.get_type("LanguageInfo")
            language.language_constant = f"languageConstants/{self.language_id}"
            request.language = language
            
            # Set locations
            location_ids = self.get_geo_target_ids()
            locations = []
            for location_id in location_ids:
                location = self.client.get_type("LocationInfo")
                location.geo_target_constant = f"geoTargetConstants/{location_id}"
                locations.append(location)
            request.geo_target_constants.extend(locations)
            
            # Set seed (keywords or URLs)
            if urls:
                url_seed = self.client.get_type("UrlSeed")
                url_seed.urls.extend(urls)
                request.url_seed = url_seed
            elif seed_keywords:
                keyword_seed = self.client.get_type("KeywordSeed")
                keyword_seed.keywords.extend(seed_keywords)
                request.keyword_seed = keyword_seed
            else:
                logger.error("No seed keywords or URLs provided")
                return []
            
            request.include_adult_keywords = include_adult_keywords
            request.keyword_plan_network = self.client.enums.KeywordPlanNetworkEnum.GOOGLE_SEARCH
            
            logger.info(f"Requesting keyword ideas with {'URLs' if urls else 'keywords'}...")
            response = keyword_plan_idea_service.generate_keyword_ideas(request=request)
            
            keyword_data = []
            for idea in response.results:
                metrics = idea.keyword_idea_metrics
                competition_level = metrics.competition  # Fixed: Correct access
                if competition_level == self.client.enums.KeywordPlanCompetitionLevelEnum.HIGH:
                    competition_str = "High"
                    competition_index = 0.8
                elif competition_level == self.client.enums.KeywordPlanCompetitionLevelEnum.MEDIUM:
                    competition_str = "Medium"
                    competition_index = 0.5
                else:
                    competition_str = "Low"
                    competition_index = 0.2
                
                # Handle search volume (use midpoint if ranged)
                avg_searches = metrics.avg_monthly_searches or 0
                if not avg_searches and hasattr(metrics, 'monthly_search_volumes'):
                    monthly_volumes = [mv.avg_monthly_searches for mv in metrics.monthly_search_volumes if mv.avg_monthly_searches]
                    avg_searches = sum(monthly_volumes) // len(monthly_volumes) if monthly_volumes else 0
                
                keyword_data.append(RealKeywordData(
                    keyword=idea.text,
                    avg_monthly_searches=avg_searches,
                    competition=competition_str,
                    competition_index=competition_index,
                    low_top_of_page_bid_micros=metrics.low_top_of_page_bid_micros or 0,
                    high_top_of_page_bid_micros=metrics.high_top_of_page_bid_micros or 0
                ))
            
            if not keyword_data:
                logger.warning("No keyword ideas returned from API")
                return []
            
            logger.info(f"Retrieved {len(keyword_data)} keyword ideas from Google Ads API")
            return keyword_data
            
        except GoogleAdsException as ex:
            logger.error(f"Google Ads API error: {ex}")
            for error in ex.failure.errors:
                logger.error(f"Error: {error.message}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error getting keyword ideas: {e}")
            return []

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def get_historical_metrics(self, keywords: List[str]) -> List[RealKeywordData]:
        """
        Get historical metrics for specific keywords
        
        Args:
            keywords: List of keywords to get metrics for
            
        Returns:
            List of RealKeywordData objects with historical metrics
        """
        if not self.client:
            logger.error("Google Ads API client not initialized")
            return []
        
        try:
            keyword_plan_idea_service = self.client.get_service("KeywordPlanIdeaService")
            request = self.client.get_type("GenerateKeywordHistoricalMetricsRequest")
            request.customer_id = self.customer_id
            request.keywords.extend(keywords)
            
            language = self.client.get_type("LanguageInfo")
            language.language_constant = f"languageConstants/{self.language_id}"
            request.language = language
            
            location_ids = self.get_geo_target_ids()
            for location_id in location_ids:
                location = self.client.get_type("LocationInfo")
                location.geo_target_constant = f"geoTargetConstants/{location_id}"
                request.geo_target_constants.append(location)
            
            request.keyword_plan_network = self.client.enums.KeywordPlanNetworkEnum.GOOGLE_SEARCH
            
            logger.info(f"Getting historical metrics for {len(keywords)} keywords...")
            response = keyword_plan_idea_service.generate_keyword_historical_metrics(request=request)
            
            keyword_data = []
            for result in response.results:
                metrics = result.keyword_metrics
                competition_level = metrics.competition
                if competition_level == self.client.enums.KeywordPlanCompetitionLevelEnum.HIGH:
                    competition_str = "High"
                    competition_index = 0.8
                elif competition_level == self.client.enums.KeywordPlanCompetitionLevelEnum.MEDIUM:
                    competition_str = "Medium"
                    competition_index = 0.5
                else:
                    competition_str = "Low"
                    competition_index = 0.2
                
                avg_searches = metrics.avg_monthly_searches or 0
                if not avg_searches and hasattr(metrics, 'monthly_search_volumes'):
                    monthly_volumes = [mv.avg_monthly_searches for mv in metrics.monthly_search_volumes if mv.avg_monthly_searches]
                    avg_searches = sum(monthly_volumes) // len(monthly_volumes) if monthly_volumes else 0
                
                keyword_data.append(RealKeywordData(
                    keyword=result.text,
                    avg_monthly_searches=avg_searches,
                    competition=competition_str,
                    competition_index=competition_index,
                    low_top_of_page_bid_micros=metrics.low_top_of_page_bid_micros or 0,
                    high_top_of_page_bid_micros=metrics.high_top_of_page_bid_micros or 0
                ))
            
            if not keyword_data:
                logger.warning("No historical metrics returned from API")
                return []
            
            logger.info(f"Retrieved historical metrics for {len(keyword_data)} keywords")
            return keyword_data
            
        except GoogleAdsException as ex:
            logger.error(f"Google Ads API error: {ex}")
            return []
        except Exception as e:
            logger.error(f"Error getting historical metrics: {e}")
            return []

class SEMCampaignOptimizer:
    def __init__(self, config_path: str, google_ads_config_path: str = None, use_real_data: bool = True):
        """Initialize SEM Campaign Optimizer with config from YAML file."""
        try:
            with open(config_path, 'r') as file:
                self.config = yaml.safe_load(file) or {}
        except Exception as e:
            logger.error(f"Failed to load config.yaml: {e}")
            self.config = {}
        
        # Required inputs with defaults
        self.brand_url = self.config.get('inputs', {}).get('brand_website', '')
        self.competitor_url = self.config.get('inputs', {}).get('competitor_website', '')
        self.service_locations = self.config.get('inputs', {}).get('service_locations', [])
        self.budgets = self.config.get('inputs', {}).get('budgets', {
            'shopping_ads': 0,
            'search_ads': 0,
            'pmax_ads': 0
        })
        
        if not (self.brand_url and self.competitor_url and self.service_locations and any(self.budgets.values())):
            logger.warning("Some required inputs missing, using defaults")
        
        # Optional inputs
        self.conversion_rate = self.config.get('conversion_rate', 0.02)
        self.min_search_volume = self.config.get('advanced', {}).get('min_search_volume', 500)
        self.competitor_terms = self.config.get('advanced', {}).get('competitor_terms', ['competitor', 'vs', 'versus', 'alternative'])
        self.brand_terms = self.config.get('advanced', {}).get('brand_terms', ['brand'])
        self.language_id = self.config.get('advanced', {}).get('language_id', '1000')
        
        # Configure Google Gemini API
        self.gemini_key = self.config.get('gemini_api_key')
        if self.gemini_key:
            try:
                genai.configure(api_key=self.gemini_key)
                logger.info("Google Gemini API configured successfully")
            except Exception as e:
                logger.warning(f"Failed to configure Gemini API: {e}. Using fallback method.")
                self.gemini_key = None
        
        # Initialize Google Keyword Planner API
        self.use_real_data = use_real_data
        self.keyword_planner_api = None
        if use_real_data:
            try:
                self.keyword_planner_api = GoogleKeywordPlannerAPI(
                    config_path=google_ads_config_path,
                    service_locations=self.service_locations,
                    language_id=self.language_id
                )
                if self.keyword_planner_api.client:
                    logger.info("Google Ads API integration enabled")
                else:
                    logger.warning("Google Ads API not available, will use simulation")
                    self.use_real_data = False
            except Exception as e:
                logger.error(f"Failed to initialize Google Ads API: {e}")
                self.use_real_data = False
        
        self.master_keywords = []
        self.filtered_keywords = []

    def extract_content_from_website(self, url: str, max_pages: int = 5) -> WebsiteContent:
        """Extract content from website for keyword analysis with retry logic."""
        logger.info(f"Extracting content from: {url}")
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        for attempt in range(3):
            try:
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                text_content = soup.get_text(separator=' ', strip=True)
                title = soup.find('title')
                title_text = title.get_text().strip() if title else ""
                meta_desc = soup.find('meta', attrs={'name': 'description'})
                meta_description = meta_desc.get('content', '') if meta_desc else ""
                headings = [h.get_text().strip() for h in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])]
                
                return {
                    'url': url,
                    'title': title_text,
                    'meta_description': meta_description,
                    'headings': headings,
                    'content': text_content[:5000],
                    'word_count': len(text_content.split()),
                    'error': None
                }
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
                if attempt == 2:
                    logger.error(f"Failed to extract content from {url}: {e}")
                    return {'url': url, 'title': '', 'meta_description': '', 'headings': [], 'content': '', 'word_count': 0, 'error': str(e)}
    
    def generate_seed_keywords_with_llm(self, brand_content: WebsiteContent, 
                                      competitor_content: WebsiteContent) -> List[str]:
        """Generate seed keywords using LLM analysis of website content."""
        logger.info("Generating seed keywords using LLM analysis...")
        
        if not self.gemini_key:
            logger.warning("Google Gemini API key not configured. Using fallback method.")
            return self.generate_seed_keywords_fallback(brand_content, competitor_content)
        
        try:
            prompt = f"""
            Analyze the following website content and generate 10 high-value seed keywords for SEM campaigns.
            
            Brand Website Content:
            Title: {brand_content.get('title', '')}
            Meta Description: {brand_content.get('meta_description', '')}
            Key Headings: {', '.join(brand_content.get('headings', [])[:10])}
            Content Preview: {brand_content.get('content', '')[:1000]}
            
            Competitor Website Content:
            Title: {competitor_content.get('title', '')}
            Meta Description: {competitor_content.get('meta_description', '')}
            Key Headings: {', '.join(competitor_content.get('headings', [])[:10])}
            
            Generate seed keywords focusing on:
            1. Core product/service terms
            2. Brand-related terms (e.g., {', '.join(self.brand_terms)})
            3. Category terms
            4. Problem-solution keywords
            5. Competitor comparison terms (e.g., {', '.join(self.competitor_terms)})
            
            Return only the keywords, one per line, without numbering or explanation.
            """
            
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(prompt)
            keywords = response.text.strip().split('\n')
            keywords = [kw.strip() for kw in keywords if kw.strip()]
            
            logger.info(f"Generated {len(keywords)} seed keywords using Gemini")
            return keywords[:20]
            
        except Exception as e:
            logger.error(f"Error generating keywords with LLM: {e}")
            return self.generate_seed_keywords_fallback(brand_content, competitor_content)
    
    def generate_seed_keywords_fallback(self, brand_content: WebsiteContent, 
                                      competitor_content: WebsiteContent) -> List[str]:
        """Fallback method to generate seed keywords without LLM."""
        logger.info("Using fallback method for seed keyword generation...")
        
        keywords = set()
        for content in [brand_content, competitor_content]:
            if 'title' in content:
                keywords.update(self.extract_keywords_from_text(content['title']))
            if 'headings' in content:
                for heading in content['headings'][:5]:
                    keywords.update(self.extract_keywords_from_text(heading))
        
        # Add generic seeds
        keywords.update(['product', 'service', 'buy', 'online'])
        return list(keywords)[:20]
    
    def extract_keywords_from_text(self, text: str) -> List[str]:
        """Extract potential keywords from text, excluding stop words."""
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'use', 'man', 'new', 'now', 'way', 'may', 'say'}
        return [word for word in words if word not in stop_words and len(word) > 3][:10]
    
    def get_real_keyword_data(self, keywords: List[str], brand_content: WebsiteContent) -> List[KeywordData]:
        """
        Get real keyword data from Google Ads API, using URLs if content is sufficient
        """
        logger.info(f"Getting real keyword data for {len(keywords)} keywords...")
        
        if self.keyword_planner_api and self.keyword_planner_api.client:
            try:
                # Use UrlSeed if brand content is sufficient (>500 words)
                if brand_content.get('word_count', 0) > 500:
                    logger.info("Using UrlSeed for keyword ideas due to sufficient brand content")
                    real_data = self.keyword_planner_api.get_keyword_ideas(
                        urls=[self.brand_url, self.competitor_url],
                        include_adult_keywords=False
                    )
                else:
                    logger.info("Using KeywordSeed for keyword ideas")
                    real_data = self.keyword_planner_api.get_keyword_ideas(
                        seed_keywords=keywords[:20],  # Increased limit
                        include_adult_keywords=False
                    )
                
                if real_data:
                    converted_data = [
                        KeywordData(
                            keyword=kw.keyword,
                            avg_monthly_searches=kw.avg_monthly_searches,
                            competition=kw.competition,
                            top_page_bid_low=kw.top_page_bid_low,
                            top_page_bid_high=kw.top_page_bid_high,
                            competition_score=kw.competition_index
                        ) for kw in real_data
                    ]
                    logger.info(f"Successfully retrieved real data for {len(converted_data)} keywords")
                    return converted_data
                
            except Exception as e:
                logger.error(f"Failed to get real keyword data: {e}")
        
        logger.warning("Falling back to simulated keyword data")
        return self.simulate_keyword_planner_data(keywords)
    
    def simulate_keyword_planner_data(self, keywords: List[str]) -> List[KeywordData]:
        """Simulate Google Keyword Planner data for keywords."""
        logger.info(f"Simulating keyword planner data for {len(keywords)} keywords...")
        
        random.seed(42)
        keyword_data = []
        
        for keyword in keywords:
            word_count = len(keyword.split())
            if word_count == 1:
                base_volume = random.randint(5000, 50000)
            elif word_count == 2:
                base_volume = random.randint(1000, 10000)
            else:
                base_volume = random.randint(100, 2000)
            
            search_volume = max(100, base_volume + random.randint(-base_volume//3, base_volume//3))
            
            if any(brand_word in keyword.lower() for brand_word in self.brand_terms):
                competition = 'High'
                comp_score = random.uniform(0.7, 1.0)
            elif word_count >= 3:
                competition = 'Low'
                comp_score = random.uniform(0.1, 0.4)
            else:
                competition = 'Medium'
                comp_score = random.uniform(0.4, 0.7)
            
            base_cpc = random.uniform(0.5, 15.0)
            if competition == 'High':
                low_bid = base_cpc * 1.5
                high_bid = base_cpc * 3.0
            elif competition == 'Medium':
                low_bid = base_cpc * 1.0
                high_bid = base_cpc * 2.0
            else:
                low_bid = base_cpc * 0.5
                high_bid = base_cpc * 1.5
            
            keyword_data.append(KeywordData(
                keyword=keyword,
                avg_monthly_searches=search_volume,
                competition=competition,
                top_page_bid_low=round(low_bid, 2),
                top_page_bid_high=round(high_bid, 2),
                competition_score=round(comp_score, 2)
            ))
        
        return keyword_data
    
    def filter_keywords(self, keyword_data: List[KeywordData]) -> List[KeywordData]:
        """Filter keywords based on search volume and negative keywords."""
        logger.info("Filtering keywords...")
        
        negative_keywords = self.config.get('advanced', {}).get('negative_keywords', [])
        filtered = []
        for kw in keyword_data:
            if (kw.avg_monthly_searches >= self.min_search_volume and 
                not any(neg in kw.keyword.lower() for neg in negative_keywords)):
                filtered.append(kw)
        
        filtered.sort(key=lambda x: x.avg_monthly_searches * (1.1 - x.competition_score), reverse=True)
        logger.info(f"Filtered to {len(filtered)} keywords from {len(keyword_data)}")
        return filtered
    
    def categorize_keywords_with_llm(self, keywords: List[KeywordData]) -> Dict[str, List[KeywordData]]:
        """Categorize keywords into ad groups using LLM."""
        logger.info("Categorizing keywords into ad groups...")
        
        if not self.gemini_key:
            logger.warning("Google Gemini API key not configured. Using fallback method.")
            return self.categorize_keywords_fallback(keywords)
        
        try:
            keyword_list = [kw.keyword for kw in keywords[:100]]  # Increased limit
            
            prompt = f"""
            Categorize the following keywords into these ad group types:
            1. Brand Terms - keywords containing brand names (e.g., {', '.join(self.brand_terms)})
            2. Category Terms - broad product/service categories
            3. Competitor Terms - competitor-related keywords (e.g., {', '.join(self.competitor_terms)})
            4. Location Terms - location-specific keywords
            5. Long-Tail Terms - specific, longer queries (3+ words)
            
            Keywords to categorize:
{chr(10).join(keyword_list)}
            
            IMPORTANT: Return ONLY a valid JSON object with no additional text, explanations, or formatting.
            The JSON should have ad group names as keys and arrays of keywords as values.
            
            Example format:
            {{"Brand Terms": ["brand whey"], "Category Terms": ["sports nutrition"], "Competitor Terms": ["competitor whey"], "Location Terms": ["product new york"], "Long-Tail Terms": ["best product for use"]}}
            """
            
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            for attempt in range(3):
                try:
                    response = model.generate_content(prompt)
                    if not response or not hasattr(response, 'text') or not response.text:
                        logger.warning(f"Empty response from Gemini API on attempt {attempt + 1}")
                        continue
                    
                    raw_response = response.text.strip()
                    json_start = raw_response.find('{')
                    json_end = raw_response.rfind('}') + 1
                    
                    if json_start == -1 or json_end == 0:
                        logger.warning(f"No JSON object found in response on attempt {attempt + 1}")
                        continue
                    
                    json_content = raw_response[json_start:json_end]
                    categorization = json.loads(json_content)
                    
                    if not isinstance(categorization, dict) or not all(isinstance(v, list) for v in categorization.values()):
                        logger.warning(f"Invalid categorization structure on attempt {attempt + 1}")
                        continue
                    
                    categorization = {k: v for k, v in categorization.items() if v}
                    if not categorization:
                        logger.warning(f"All categories are empty on attempt {attempt + 1}")
                        continue
                    
                    result = {}
                    for name, kws in categorization.items():
                        matched_keywords = [kw for kw in keywords if kw.keyword in kws]
                        if matched_keywords:
                            result[name] = matched_keywords
                    
                    if not result:
                        logger.warning(f"No keywords matched categories on attempt {attempt + 1}")
                        continue
                    
                    logger.info(f"Successfully categorized keywords into {len(result)} groups")
                    return result
                
                except Exception as e:
                    logger.warning(f"API call failed on attempt {attempt + 1}: {e}")
            
            logger.error("Failed to categorize with LLM after 3 attempts")
            return self.categorize_keywords_fallback(keywords)
        
        except Exception as e:
            logger.error(f"Unexpected error in LLM categorization: {e}")
            return self.categorize_keywords_fallback(keywords)
    
    def categorize_keywords_fallback(self, keywords: List[KeywordData]) -> Dict[str, List[KeywordData]]:
        """Fallback method to categorize keywords without LLM."""
        logger.info("Using fallback method for keyword categorization...")
        
        categories = {
            'Brand Terms': [],
            'Category Terms': [],
            'Competitor Terms': [],
            'Location Terms': [],
            'Long-Tail Terms': [],
            'General Terms': []
        }
        
        for kw in keywords:
            keyword_lower = kw.keyword.lower()
            word_count = len(kw.keyword.split())
            
            if any(loc.lower() in keyword_lower for loc in self.service_locations):
                categories['Location Terms'].append(kw)
            elif any(term in keyword_lower for term in self.competitor_terms):
                categories['Competitor Terms'].append(kw)
            elif word_count >= 3:
                categories['Long-Tail Terms'].append(kw)
            elif word_count == 2 and kw.avg_monthly_searches > 2000:
                categories['Category Terms'].append(kw)
            elif any(brand_word in keyword_lower for brand_word in self.brand_terms):
                categories['Brand Terms'].append(kw)
            else:
                categories['General Terms'].append(kw)
        
        logger.info(f"Categorized into {len([k for k, v in categories.items() if v])} non-empty groups")
        return {k: v for k, v in categories.items() if v}
    
    def calculate_suggested_cpc(self, keywords: List[KeywordData], budget: float) -> Tuple[float, float]:
        """Calculate suggested CPC range for a group of keywords."""
        if not keywords:
            logger.warning("No keywords provided for CPC calculation, using default range")
            return (1.0, 3.0)
        
        avg_low = sum(kw.top_page_bid_low for kw in keywords) / len(keywords)
        avg_high = sum(kw.top_page_bid_high for kw in keywords) / len(keywords)
        
        daily_budget = budget / 30
        estimated_clicks = daily_budget / max(avg_high, 0.01)
        
        if estimated_clicks < 10:
            return (avg_low * 0.8, avg_high * 0.9)
        return (avg_low, avg_high)
    
    def create_ad_groups(self, categorized_keywords: Dict[str, List[KeywordData]]) -> List[AdGroup]:
        """Create structured ad groups with match type recommendations."""
        logger.info("Creating ad groups...")
        
        ad_groups = []
        search_budget = self.budgets.get('search_ads', 0)
        num_categories = len(categorized_keywords) or 1
        
        for group_name, keywords in categorized_keywords.items():
            if not keywords:
                continue
            
            if 'Brand' in group_name:
                match_types = ['Exact', 'Phrase']
            elif 'Long-Tail' in group_name:
                match_types = ['Phrase', 'Broad Match']
            else:
                match_types = ['Phrase', 'Broad Match', 'Exact']
            
            group_budget = search_budget / num_categories
            cpc_range = self.calculate_suggested_cpc(keywords, group_budget)
            
            keyword_output = [{
                'keyword': kw.keyword,
                'avg_monthly_searches': kw.avg_monthly_searches,
                'competition': kw.competition,
                'top_page_bid_low': kw.top_page_bid_low,
                'top_page_bid_high': kw.top_page_bid_high,
                'competition_score': kw.competition_score
            } for kw in keywords]
            
            ad_groups.append(AdGroup(
                name=group_name,
                keywords=keyword_output,
                suggested_cpc_range=cpc_range,
                match_types=match_types
            ))
        
        logger.info(f"Created {len(ad_groups)} ad groups")
        return ad_groups
    
    def generate_pmax_themes(self, categorized_keywords: Dict[str, List[KeywordData]]) -> List[Dict[str, Any]]:
        """Generate Performance Max campaign themes."""
        logger.info("Generating Performance Max themes...")
        
        themes = []
        if 'Category Terms' in categorized_keywords:
            category_keywords = categorized_keywords['Category Terms'][:5]
            themes.append({
                'theme_name': 'Product Category Focus',
                'theme_type': 'Product Category',
                'keywords': [kw.keyword for kw in category_keywords],
                'target_audience': 'Users searching for product categories',
                'asset_group_focus': 'Product features and benefits'
            })
        
        if 'Location Terms' in categorized_keywords:
            location_keywords = categorized_keywords['Location Terms'][:5]
            themes.append({
                'theme_name': 'Local Services Focus',
                'theme_type': 'Geographic',
                'keywords': [kw.keyword for kw in location_keywords],
                'target_audience': 'Local customers',
                'asset_group_focus': 'Local presence and service areas'
            })
        
        if 'Long-Tail Terms' in categorized_keywords:
            longtail_keywords = categorized_keywords['Long-Tail Terms'][:5]
            themes.append({
                'theme_name': 'High-Intent Queries',
                'theme_type': 'Use-case Based',
                'keywords': [kw.keyword for kw in longtail_keywords],
                'target_audience': 'Users with specific needs',
                'asset_group_focus': 'Solution-oriented messaging'
            })
        
        logger.info(f"Generated {len(themes)} PMax themes")
        return themes
    
    def calculate_shopping_cpc_suggestions(self, keywords: List[KeywordData]) -> Dict[str, Any]:
        """Calculate suggested CPC bids for Shopping campaigns."""
        shopping_budget = self.budgets.get('shopping_ads', 0)
        
        commercial_keywords = [
            kw for kw in keywords 
            if any(term in kw.keyword.lower() for term in ['buy', 'price', 'cheap', 'best', 'review', 'compare'])
        ]
        
        if not commercial_keywords:
            commercial_keywords = sorted(keywords, key=lambda x: x.avg_monthly_searches, reverse=True)[:10]
            logger.warning("No commercial keywords found, using top 10 high-volume keywords")
        
        # Estimate expected conversions (sum of search volumes * conversion rate)
        total_search_volume = sum(kw.avg_monthly_searches for kw in commercial_keywords)
        expected_conversions = total_search_volume * self.conversion_rate
        
        # Calculate Target CPA (budget per conversion)
        target_cpa = shopping_budget / max(expected_conversions, 1) if shopping_budget > 0 else 50.0
        
        # Target CPC = Target CPA * Conversion Rate
        target_cpc = target_cpa * self.conversion_rate
        
        # Adjust CPC based on competition
        avg_high_bid = sum(kw.top_page_bid_high for kw in commercial_keywords) / max(len(commercial_keywords), 1)
        suggested_cpc = min(target_cpc, avg_high_bid * 0.8)  # Cap at 80% of high bid for profitability
        
        return {
            'suggested_max_cpc': round(suggested_cpc, 2),
            'target_cpa': round(target_cpa, 2),
            'daily_budget_recommendation': round(shopping_budget / 30, 2),
            'high_priority_keywords': [kw.keyword for kw in commercial_keywords[:5]],
            'budget_allocation': {
                'high_volume_products': 0.6,
                'medium_volume_products': 0.3,
                'long_tail_products': 0.1
            }
        }
    
    def run_campaign_optimization(self) -> Dict[str, Any]:
        """Run the complete SEM campaign optimization process."""
        logger.info("Starting SEM Campaign Optimization...")
        
        used_fallback_categorization = False
        used_real_keyword_data = self.use_real_data and self.keyword_planner_api and self.keyword_planner_api.client
        
        brand_content = self.extract_content_from_website(self.brand_url)
        competitor_content = self.extract_content_from_website(self.competitor_url)
        
        seed_keywords = self.generate_seed_keywords_with_llm(brand_content, competitor_content)
        
        expanded_keywords = set(seed_keywords)
        for location in self.service_locations:
            for seed in seed_keywords[:5]:
                expanded_keywords.add(f"{seed} {location}")
                expanded_keywords.add(f"{seed} near {location}")
        expanded_keywords = list(expanded_keywords)
        
        keyword_data = self.get_real_keyword_data(expanded_keywords, brand_content)
        self.filtered_keywords = self.filter_keywords(keyword_data)
        
        categorized_keywords = self.categorize_keywords_with_llm(self.filtered_keywords)
        if not self.gemini_key or not any(categorized_keywords.values()):
            used_fallback_categorization = True
        
        ad_groups = self.create_ad_groups(categorized_keywords)
        pmax_themes = self.generate_pmax_themes(categorized_keywords)
        shopping_suggestions = self.calculate_shopping_cpc_suggestions(self.filtered_keywords)
        
        results = {
            'campaign_summary': {
                'total_keywords_discovered': len(expanded_keywords),
                'total_keywords_filtered': len(self.filtered_keywords),
                'total_ad_groups': len(ad_groups),
                'budgets': self.budgets,
                'conversion_rate': self.conversion_rate,
                'used_fallback_categorization': used_fallback_categorization,
                'used_real_keyword_data': used_real_keyword_data,
                'data_source': 'Google Ads API' if used_real_keyword_data else 'Simulated Data'
            },
            'search_campaign': {
                'ad_groups': [
                    {
                        'name': ag.name,
                        'keywords': ag.keywords,
                        'suggested_cpc_range': {
                            'low': ag.suggested_cpc_range[0],
                            'high': ag.suggested_cpc_range[1]
                        },
                        'recommended_match_types': ag.match_types,
                        'keyword_count': len(ag.keywords)
                    }
                    for ag in ad_groups
                ]
            },
            'performance_max_campaign': {
                'themes': pmax_themes
            },
            'shopping_campaign': shopping_suggestions,
            'website_analysis': {
                'brand_website': {
                    'url': brand_content.get('url'),
                    'title': brand_content.get('title'),
                    'word_count': brand_content.get('word_count', 0)
                },
                'competitor_website': {
                    'url': competitor_content.get('url'),
                    'title': competitor_content.get('title'),
                    'word_count': competitor_content.get('word_count', 0)
                }
            }
        }
        
        logger.info("SEM Campaign Optimization completed")
        return results
    
    def save_results(self, results: Dict[str, Any], output_file: str = 'sem_campaign_results.json'):
        """Save results to a JSON file and export keywords to CSV in an output folder."""
    
        output_dir = 'output'
        os.makedirs(output_dir, exist_ok=True)
    
        json_path = os.path.join(output_dir, output_file)
    
    # Save JSON file
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {json_path}")
    
    # Export keywords to CSV in output folder
        self.export_keywords_to_csv(results, output_dir)
    
    def export_keywords_to_csv(self, results: Dict[str, Any], output_dir: str = '.'):
        """Export keywords to CSV for analysis."""
        all_keywords = []
        
        for ad_group in results['search_campaign']['ad_groups']:
            for keyword in ad_group['keywords']:
                keyword_row = keyword.copy()
                keyword_row['ad_group'] = ad_group['name']
                keyword_row['suggested_cpc_low'] = ad_group['suggested_cpc_range']['low']
                keyword_row['suggested_cpc_high'] = ad_group['suggested_cpc_range']['high']
                keyword_row['match_types'] = ', '.join(ad_group['recommended_match_types'])
                all_keywords.append(keyword_row)
        
        df = pd.DataFrame(all_keywords)
        csv_path = os.path.join(output_dir, 'keyword_analysis.csv')
        df.to_csv(csv_path, index=False)
        logger.info(f"Keywords exported to {csv_path}")


def main():
    """Main function to run the SEM Campaign Optimizer."""
    try:
        optimizer = SEMCampaignOptimizer(
            config_path='config.yaml',
            google_ads_config_path= os.loadenv('GOOGLE_ADS_CONFIG_PATH', 'google-ads.yaml'),
            use_real_data=True
        )
        
        results = optimizer.run_campaign_optimization()
        optimizer.save_results(results)
        
        print("\n" + "="*60)
        print("SEM CAMPAIGN OPTIMIZATION COMPLETE")
        print("="*60)
        print(f"Data Source: {results['campaign_summary']['data_source']}")
        print(f"Total Keywords Discovered: {results['campaign_summary']['total_keywords_discovered']}")
        print(f"Keywords After Filtering: {results['campaign_summary']['total_keywords_filtered']}")
        print(f"Ad Groups Created: {results['campaign_summary']['total_ad_groups']}")
        print(f"Performance Max Themes: {len(results['performance_max_campaign']['themes'])}")
        
        if results['campaign_summary']['used_real_keyword_data']:
            print("✓ Used real Google Ads API data")
        else:
            print("⚠ Used simulated keyword data")
            
        if results['campaign_summary']['used_fallback_categorization']:
            print("⚠ Used fallback categorization method")
        else:
            print("✓ Used AI-powered categorization")
        
        print("\nAd Group Summary:")
        print("-" * 40)
        
        for ad_group in results['search_campaign']['ad_groups']:
            print(f"• {ad_group['name']}: {ad_group['keyword_count']} keywords")
            print(f"  CPC Range: ${ad_group['suggested_cpc_range']['low']:.2f} - ${ad_group['suggested_cpc_range']['high']:.2f}")
        
        print(f"\nResults saved to: sem_campaign_results.json")
        print(f"Keywords exported to: keyword_analysis.csv")
        
    except Exception as e:
        logger.error(f"Error running optimization: {e}")
        raise

if __name__ == "__main__":
    main()