"""
Advanced Campaign Builder
Provides sophisticated campaign structuring with ROAS optimization
"""

import json
import pandas as pd
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
import math
import logging
import google.generativeai as genai

logger = logging.getLogger(__name__)

@dataclass
class CampaignStructure:
    campaign_name: str
    campaign_type: str  
    daily_budget: float
    target_cpa: float
    target_roas: float
    ad_groups: List[Dict[str, Any]]

@dataclass
class BidStrategy:
    strategy_type: str  # Manual, Enhanced CPC, Target CPA, Target ROAS
    target_value: float
    bid_adjustments: Dict[str, float]

class AdvancedCampaignBuilder:
    """
    Advanced campaign builder with ROAS optimization and sophisticated structuring
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.conversion_rate = config.get('conversion_rate', 0.02)
        self.target_roas = config.get('target_roas', 4.0)  # 400% ROAS target
        
        # Configure Google Gemini 1.5 Flash
        api_key = config.get('gemini_api_key')
        if api_key:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            logger.info("Google Gemini 1.5 Flash model initialized successfully.")
        else:
            self.model = None
            logger.warning("Gemini API key not provided. Falling back to static methods.")
        
    def build_search_campaigns(self, keyword_data: List[Dict], budgets: Dict) -> List[CampaignStructure]:
        """
        Build optimized search campaigns with advanced structuring
        """
        logger.info("Building advanced search campaign structure...")
        
        campaigns = []
        
        # Separate campaigns by performance tier
        high_performance_keywords = self._filter_high_performance_keywords(keyword_data)
        brand_keywords = self._filter_brand_keywords(keyword_data)
        competitor_keywords = self._filter_competitor_keywords(keyword_data)
        generic_keywords = self._filter_generic_keywords(keyword_data)
        
        if high_performance_keywords:
            high_perf_campaign = self._create_high_performance_campaign(
                high_performance_keywords, budgets['search_ads'] * 0.4
            )
            campaigns.append(high_perf_campaign)
        
        if brand_keywords:
            brand_campaign = self._create_brand_campaign(
                brand_keywords, budgets['search_ads'] * 0.2
            )
            campaigns.append(brand_campaign)
        
        if competitor_keywords:
            competitor_campaign = self._create_competitor_campaign(
                competitor_keywords, budgets['search_ads'] * 0.2
            )
            campaigns.append(competitor_campaign)
        
        if generic_keywords:
            generic_campaign = self._create_generic_campaign(
                generic_keywords, budgets['search_ads'] * 0.2
            )
            campaigns.append(generic_campaign)
        
        return campaigns
    def _extract_product_category(self, keyword: str) -> str:
        """Extract product category from keyword for cubehq.ai products"""
        keyword_lower = keyword.lower()
    
        category_mapping = {
        'analytics': ['analytics', 'analysis', 'insights', 'metrics', 'kpi'],
        'dashboard': ['dashboard', 'visualization', 'report', 'chart', 'graph'],
        'integration': ['integration', 'api', 'connect', 'sync', 'import'],
        'automation': ['automate', 'auto', 'scheduled', 'workflow', 'pipeline'],
        'ai': ['ai', 'machine learning', 'ml', 'predictive', 'forecast']
        }
    
    # Check keyword against each category
        for category, terms in category_mapping.items():
            if any(term in keyword_lower for term in terms):
                return category.capitalize()
    
    # Default category if no match found
        return 'General'
    
    def _estimate_shopping_performance(self, campaigns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Estimate performance metrics for Shopping campaigns"""
        total_metrics = {
        'estimated_daily_clicks': 0,
        'estimated_daily_conversions': 0,
        'estimated_daily_cost': 0,
        'estimated_daily_value': 0,
        'estimated_roas': 0
        }
    
        for campaign in campaigns:
            campaign_metrics = {
            'estimated_daily_clicks': 0,
            'estimated_daily_conversions': 0,
            'estimated_daily_cost': 0,
            'estimated_daily_value': 0
            }
        
            for product_group in campaign.get('product_groups', []):
                suggested_cpc = product_group.get('suggested_cpc', 0)
                metrics = product_group.get('performance_metrics', {})
            
                estimated_ctr = 0.02 
                monthly_searches = metrics.get('avg_monthly_searches', 0)
                daily_clicks = int((monthly_searches / 30) * estimated_ctr)
            
                conversions = daily_clicks * self.conversion_rate
            
                cost = daily_clicks * suggested_cpc
                value = conversions * (suggested_cpc * self.target_roas) / self.conversion_rate
            
                campaign_metrics['estimated_daily_clicks'] += daily_clicks
                campaign_metrics['estimated_daily_conversions'] += conversions
                campaign_metrics['estimated_daily_cost'] += cost
                campaign_metrics['estimated_daily_value'] += value
            
                total_metrics['estimated_daily_clicks'] += daily_clicks
                total_metrics['estimated_daily_conversions'] += conversions
                total_metrics['estimated_daily_cost'] += cost
                total_metrics['estimated_daily_value'] += value
        
            campaign['estimated_performance'] = campaign_metrics
            if campaign_metrics['estimated_daily_cost'] > 0:
                campaign['estimated_performance']['estimated_roas'] = (
                    campaign_metrics['estimated_daily_value'] / campaign_metrics['estimated_daily_cost']
                )
    
        # Calculate overall ROAS
        if total_metrics['estimated_daily_cost'] > 0:
            total_metrics['estimated_roas'] = (
                total_metrics['estimated_daily_value'] / total_metrics['estimated_daily_cost']
            )
    
        return total_metrics
    
    def _generate_shopping_optimizations(self, campaigns: List[Dict[str, Any]]) -> List[str]:
        """Generate optimization recommendations for Shopping campaigns"""
        optimizations = []
    
        # Check if we have high priority campaigns
        high_priority_campaigns = [c for c in campaigns if c.get('priority') == 'High']
        if high_priority_campaigns:
            optimizations.append(
                "Prioritize high-performing product groups with increased bids during peak conversion times"
            )
    
        # Check budget allocation
        total_budget = sum(c.get('daily_budget', 0) for c in campaigns)
        if total_budget > 0:
            high_priority_ratio = sum(
                c.get('daily_budget', 0) for c in campaigns if c.get('priority') == 'High'
            ) / total_budget
        
            if high_priority_ratio < 0.5:
                optimizations.append(
                    "Consider reallocating more budget to high priority campaigns (currently {:.0f}%)".format(
                        high_priority_ratio * 100
                    )
                )
    
        # Check for high CPC products
        for campaign in campaigns:
            for product_group in campaign.get('product_groups', []):
                cpc = product_group.get('suggested_cpc', 0)
                competition = product_group.get('performance_metrics', {}).get('competition_level', 0)
            
                if cpc > 5.0 and competition > 0.7:
                    optimizations.append(
                        f"Review bids for {product_group['name']} (high CPC ${cpc:.2f} and high competition)"
                    )
    
        # Default optimizations
        optimizations.extend([
            "Review search term reports weekly to identify new negative keywords",
            "Optimize product feed with high-performing keywords in titles and descriptions",
            "Test different product image styles (lifestyle vs white background)",
            "Implement remarketing audiences for Shopping campaign viewers"
        ])
    
        return optimizations[:5] 
    
    def _filter_high_performance_keywords(self, keywords: List[Dict]) -> List[Dict]:
        """Filter keywords with high commercial intent and good volume/competition ratio"""
        high_performance = []
        
        for kw in keywords:
            volume = kw.get('avg_monthly_searches', 0)
            comp_score = kw.get('competition_score', 1.0)
            cpc_low = kw.get('top_page_bid_low', 0)
            
            if volume > 2000 and comp_score < 0.6 and cpc_low < 5.0:
                performance_score = (volume / 1000) * (1 - comp_score) / (cpc_low + 0.1)
                if performance_score > 100:
                    kw['performance_score'] = performance_score
                    high_performance.append(kw)
        
        return sorted(high_performance, key=lambda x: x['performance_score'], reverse=True)[:20]
    
    def _filter_brand_keywords(self, keywords: List[Dict]) -> List[Dict]:
        """Filter brand-related keywords"""
        brand_terms = ['brand', 'company', 'official', 'website', 'reviews']
        return [kw for kw in keywords if any(term in kw['keyword'].lower() for term in brand_terms)]
    
    def _filter_competitor_keywords(self, keywords: List[Dict]) -> List[Dict]:
        """Filter competitor-related keywords"""
        competitor_terms = ['vs', 'versus', 'alternative', 'compare', 'competitor']
        return [kw for kw in keywords if any(term in kw['keyword'].lower() for term in competitor_terms)]
    
    def _filter_generic_keywords(self, keywords: List[Dict]) -> List[Dict]:
        """Filter generic/category keywords"""
        filtered_sets = [
            self._filter_high_performance_keywords(keywords),
            self._filter_brand_keywords(keywords),
            self._filter_competitor_keywords(keywords)
        ]
        
        used_keywords = set()
        for keyword_set in filtered_sets:
            used_keywords.update(kw['keyword'] for kw in keyword_set)
        
        return [kw for kw in keywords if kw['keyword'] not in used_keywords]
    
    def _create_high_performance_campaign(self, keywords: List[Dict], budget: float) -> CampaignStructure:
        """Create high-performance campaign with aggressive bidding"""
        ad_groups = self._create_performance_ad_groups(keywords)
        
        # Calculate target CPA based on ROAS
        avg_cpc = sum(kw.get('top_page_bid_high', 1.0) for kw in keywords) / len(keywords)
        target_cpa = avg_cpc / self.conversion_rate
        
        return CampaignStructure(
            campaign_name="High Performance Search",
            campaign_type="Search",
            daily_budget=budget / 30,
            target_cpa=target_cpa,
            target_roas=self.target_roas * 1.2,  
            ad_groups=ad_groups
        )
    
    def _create_brand_campaign(self, keywords: List[Dict], budget: float) -> CampaignStructure:
        """Create brand protection campaign"""
        ad_groups = [{
            'name': 'Brand Terms',
            'keywords': keywords,
            'match_types': ['Exact', 'Phrase'],
            'max_cpc': min(kw.get('top_page_bid_high', 1.0) for kw in keywords) * 0.8,
            'bid_strategy': 'Enhanced CPC'
        }]
        
        avg_cpc = sum(kw.get('top_page_bid_low', 1.0) for kw in keywords) / len(keywords)
        target_cpa = avg_cpc / (self.conversion_rate * 2)  
        
        return CampaignStructure(
            campaign_name="Brand Protection",
            campaign_type="Search",
            daily_budget=budget / 30,
            target_cpa=target_cpa,
            target_roas=self.target_roas * 1.5,  
            ad_groups=ad_groups
        )
    
    def _create_competitor_campaign(self, keywords: List[Dict], budget: float) -> CampaignStructure:
        """Create competitor targeting campaign"""
        ad_groups = [{
            'name': 'Competitor Terms',
            'keywords': keywords,
            'match_types': ['Phrase', 'Broad Match'],
            'max_cpc': sum(kw.get('top_page_bid_high', 1.0) for kw in keywords) / len(keywords) * 1.1,
            'bid_strategy': 'Target CPA',
            'landing_page_strategy': 'Comparison focused'
        }]
        
        avg_cpc = sum(kw.get('top_page_bid_high', 1.0) for kw in keywords) / len(keywords)
        target_cpa = avg_cpc / (self.conversion_rate * 0.8) 
        
        return CampaignStructure(
            campaign_name="Competitor Targeting",
            campaign_type="Search",
            daily_budget=budget / 30,
            target_cpa=target_cpa,
            target_roas=self.target_roas * 0.8, 
            ad_groups=ad_groups
        )
    def _calculate_target_cpc(self, keyword_data: Dict[str, Any]) -> float:
        """Calculate target CPC based on conversion rate and target ROAS"""
        avg_cpc = keyword_data.get('top_page_bid_high', 1.0)
        target_cpa = (avg_cpc * self.target_roas) / self.conversion_rate
        target_cpc = target_cpa * self.conversion_rate
        return round(target_cpc, 2)
    
    def _create_generic_campaign(self, keywords: List[Dict], budget: float) -> CampaignStructure:
        """Create generic terms campaign"""
        ad_groups = self._create_themed_ad_groups(keywords)
        
        avg_cpc = sum(kw.get('top_page_bid_high', 1.0) for kw in keywords) / len(keywords)
        target_cpa = avg_cpc / self.conversion_rate
        
        return CampaignStructure(
            campaign_name="Generic Search",
            campaign_type="Search",
            daily_budget=budget / 30,
            target_cpa=target_cpa,
            target_roas=self.target_roas,
            ad_groups=ad_groups
        )
    
    def _create_performance_ad_groups(self, keywords: List[Dict]) -> List[Dict[str, Any]]:
        """Create ad groups optimized for performance"""
        ad_groups = []
        
        high_cpc_keywords = [kw for kw in keywords if kw.get('top_page_bid_high', 0) > 3.0]
        medium_cpc_keywords = [kw for kw in keywords if 1.0 <= kw.get('top_page_bid_high', 0) <= 3.0]
        low_cpc_keywords = [kw for kw in keywords if kw.get('top_page_bid_high', 0) < 1.0]
        
        if high_cpc_keywords:
            ad_groups.append({
                'name': 'High Value Terms',
                'keywords': high_cpc_keywords[:10],  # Limit to best performers
                'match_types': ['Exact', 'Phrase'],
                'max_cpc': sum(kw.get('top_page_bid_high', 1.0) for kw in high_cpc_keywords) / len(high_cpc_keywords) * 0.9,
                'bid_strategy': 'Target ROAS',
                'quality_score_target': 8
            })
        
        if medium_cpc_keywords:
            ad_groups.append({
                'name': 'Medium Value Terms',
                'keywords': medium_cpc_keywords,
                'match_types': ['Phrase', 'Broad Match'],
                'max_cpc': sum(kw.get('top_page_bid_high', 1.0) for kw in medium_cpc_keywords) / len(medium_cpc_keywords),
                'bid_strategy': 'Enhanced CPC',
                'quality_score_target': 7
            })
        
        if low_cpc_keywords:
            ad_groups.append({
                'name': 'Volume Terms',
                'keywords': low_cpc_keywords,
                'match_types': ['Broad Match'],
                'max_cpc': sum(kw.get('top_page_bid_high', 1.0) for kw in low_cpc_keywords) / len(low_cpc_keywords) * 1.2,
                'bid_strategy': 'Maximize Clicks',
                'quality_score_target': 6
            })
        
        return ad_groups
    
    def _create_themed_ad_groups(self, keywords: List[Dict]) -> List[Dict[str, Any]]:
        """Create thematically grouped ad groups"""
        themes = {
            'Product Terms': [],
            'Service Terms': [],
            'Local Terms': [],
            'Long-Tail Terms': [],
            'Question Terms': []
        }
        
        for kw in keywords:
            keyword_lower = kw['keyword'].lower()
            word_count = len(kw['keyword'].split())
            
            if any(term in keyword_lower for term in ['product', 'item', 'buy', 'purchase']):
                themes['Product Terms'].append(kw)
            elif any(term in keyword_lower for term in ['service', 'help', 'support', 'consultation']):
                themes['Service Terms'].append(kw)
            elif any(term in keyword_lower for term in ['near', 'local', 'around', 'nearby']):
                themes['Local Terms'].append(kw)
            elif word_count >= 4:
                themes['Long-Tail Terms'].append(kw)
            elif any(term in keyword_lower for term in ['what', 'how', 'why', 'when', 'where']):
                themes['Question Terms'].append(kw)
            else:
                themes['Product Terms'].append(kw)  
        
        ad_groups = []
        for theme_name, theme_keywords in themes.items():
            if theme_keywords:
                avg_cpc = sum(kw.get('top_page_bid_high', 1.0) for kw in theme_keywords) / len(theme_keywords)
                
                ad_groups.append({
                    'name': theme_name,
                    'keywords': theme_keywords,
                    'match_types': self._determine_match_types(theme_name),
                    'max_cpc': avg_cpc,
                    'bid_strategy': self._determine_bid_strategy(theme_name),
                    'estimated_clicks_per_day': self._estimate_daily_clicks(theme_keywords, avg_cpc)
                })
        
        return ad_groups
    
    def _determine_match_types(self, theme_name: str) -> List[str]:
        """Determine optimal match types based on theme"""
        match_type_mapping = {
            'Product Terms': ['Exact', 'Phrase'],
            'Service Terms': ['Phrase', 'Broad Match'],
            'Local Terms': ['Phrase', 'Broad Match'],
            'Long-Tail Terms': ['Phrase'],
            'Question Terms': ['Broad Match']
        }
        return match_type_mapping.get(theme_name, ['Phrase', 'Broad Match'])
    
    def _determine_bid_strategy(self, theme_name: str) -> str:
        """Determine optimal bid strategy based on theme"""
        strategy_mapping = {
            'Product Terms': 'Target ROAS',
            'Service Terms': 'Target CPA',
            'Local Terms': 'Enhanced CPC',
            'Long-Tail Terms': 'Manual CPC',
            'Question Terms': 'Maximize Clicks'
        }
        return strategy_mapping.get(theme_name, 'Enhanced CPC')
    
    def _estimate_daily_clicks(self, keywords: List[Dict], avg_cpc: float) -> int:
        """Estimate daily clicks based on search volume and CPC"""
        total_monthly_searches = sum(kw.get('avg_monthly_searches', 0) for kw in keywords)
        daily_searches = total_monthly_searches / 30
        
        # Estimate CTR based on avg position 
        estimated_ctr = 0.02  
        if avg_cpc > 3.0:
            estimated_ctr *= 1.5  
        elif avg_cpc < 1.0:
            estimated_ctr *= 0.7  
        
        estimated_daily_clicks = int(daily_searches * estimated_ctr)
        return max(1, estimated_daily_clicks)
    
    def create_performance_max_strategy(self, keyword_data: List[Dict], budget: float) -> Dict[str, Any]:
        """Create Performance Max campaign strategy"""
        logger.info("Creating Performance Max campaign strategy...")
        
        # Analyze keywords to create asset group themes
        themes = self._analyze_keyword_themes(keyword_data)
        
        asset_groups = []
        theme_budget = budget / max(1, len(themes))
        
        for theme_name, theme_keywords in themes.items():
            if not theme_keywords:
                continue
                
            # Calculate performance metrics for theme
            avg_volume = sum(kw.get('avg_monthly_searches', 0) for kw in theme_keywords) / len(theme_keywords)
            avg_cpc = sum(kw.get('top_page_bid_high', 1.0) for kw in theme_keywords) / len(theme_keywords)
            
            asset_group = {
                'name': f"{theme_name} Asset Group",
                'theme': theme_name,
                'target_keywords': [kw['keyword'] for kw in theme_keywords[:10]], 
                'audience_signals': self._generate_audience_signals(theme_keywords),
                'budget_allocation': theme_budget / 30, 
                'target_cpa': avg_cpc / self.conversion_rate,
                'target_roas': self.target_roas,
                'creative_themes': self._generate_creative_themes(theme_name, theme_keywords),
                'performance_estimate': {
                    'estimated_daily_impressions': avg_volume * len(theme_keywords) / 30,
                    'estimated_daily_clicks': self._estimate_daily_clicks(theme_keywords, avg_cpc),
                    'estimated_daily_conversions': self._estimate_daily_clicks(theme_keywords, avg_cpc) * self.conversion_rate
                }
            }
            
            asset_groups.append(asset_group)
        
        return {
            'campaign_name': 'Performance Max - Multi-Theme',
            'campaign_type': 'Performance Max',
            'daily_budget': budget / 30,
            'bidding_strategy': 'Target ROAS',
            'target_roas': self.target_roas,
            'asset_groups': asset_groups,
            'conversion_goals': self._define_conversion_goals(),
            'audience_expansion': True,
            'creative_rotation': 'Optimize'
        }
    
    def _analyze_keyword_themes(self, keywords: List[Dict]) -> Dict[str, List[Dict]]:
        """Analyze keywords to identify natural themes for Performance Max"""
        themes = {
        'Product Category Themes': [],      
        'Use-case Based Themes': [],         
        'Demographic Themes': [],             
        'Seasonal/Event-Based Themes': [],    
        'Solution-Focused Themes': []        
        }

        product_terms = ['analytics', 'dashboard', 'reporting', 'insights', 'metrics', 'kpi', 'visualization']
        use_case_terms = ['customer', 'behavior', 'predictive', 'trend', 'forecast', 'tracking', 'monitoring']
        demographic_terms = ['saas', 'enterprise', 'startup', 'ecommerce', 'marketing', 'sales', 'team']
        seasonal_terms = ['q1', 'q2', 'q3', 'q4', 'quarterly', 'annual', 'year end', 'monthly']
        solution_terms = ['integration', 'automated', 'real-time', 'scalable', 'custom', 'embedded']
        
        for kw in keywords:
            keyword_lower = kw['keyword'].lower()
        
            if any(term in keyword_lower for term in product_terms):
                themes['Product Category Themes'].append(kw)
            elif any(term in keyword_lower for term in use_case_terms):
                themes['Use-case Based Themes'].append(kw)
            elif any(term in keyword_lower for term in demographic_terms):
                themes['Demographic Themes'].append(kw)
            elif any(term in keyword_lower for term in seasonal_terms):
                themes['Seasonal/Event-Based Themes'].append(kw)
            elif any(term in keyword_lower for term in solution_terms):
                themes['Solution-Focused Themes'].append(kw)
            else:
                themes['Product Category Themes'].append(kw)
        
        # Remove empty themes
        return {k: v for k, v in themes.items() if v}
    
    def _generate_audience_signals(self, keywords: List[Dict]) -> List[str]:
        """Generate audience signals based on keyword analysis using Gemini if available"""
        keyword_text = ' '.join(kw['keyword'].lower() for kw in keywords)
    
        if self.model:
            prompt = f"""Based on these keywords related to AI analytics (cubehq.ai): {keyword_text}
    Generate a list of 5 highly relevant audience signals for Google Ads campaigns targeting business analytics professionals.
    Focus on signals like job roles, industries, and business needs.
    Output as a bullet list."""
            try:
                response = self.model.generate_content(prompt)
                signals = [line.strip('- ').strip() for line in response.text.split('\n') if line.strip()]
                return signals[:5]
            except Exception as e:
                logger.error(f"Error generating audience signals with Gemini: {e}")

        # Fallback logic specifically for cubehq.ai (AI analytics platform)
        signals = []

        # Industry signals
        if 'saas' in keyword_text or 'software' in keyword_text:
            signals.extend(['SaaS companies', 'Software development teams'])
        if 'ecommerce' in keyword_text or 'retail' in keyword_text:
            signals.extend(['Ecommerce managers', 'Retail analytics teams'])
        if 'marketing' in keyword_text or 'growth' in keyword_text:
            signals.extend(['Marketing analysts', 'Growth teams'])

        # Job role signals
        if 'analyst' in keyword_text or 'analytics' in keyword_text:
            signals.extend(['Data analysts', 'Business intelligence professionals'])
        if 'cto' in keyword_text or 'technical' in keyword_text:
            signals.extend(['CTOs', 'Technical decision makers'])

        # Business size signals
        if 'enterprise' in keyword_text or 'large' in keyword_text:
            signals.append('Enterprise businesses')
        if 'startup' in keyword_text or 'small business' in keyword_text:
            signals.append('Startup founders')

        # Add high-value signals based on CPC
        avg_cpc = sum(kw.get('top_page_bid_high', 1.0) for kw in keywords) / len(keywords)
        if avg_cpc > 5.0:
            signals.append('High-budget analytics teams')

        unique_signals = list(dict.fromkeys(signals)) 
        return unique_signals[:5] 
    
    def _generate_creative_themes(self, theme_name: str, keywords: List[Dict]) -> List[str]:
        """Generate creative themes for asset groups"""
        theme_mapping = {
        'Product Category Themes': [
            'AI-powered analytics dashboard features',
            'Real-time data visualization capabilities',
            'Custom reporting tool demonstrations',
            'Key metric tracking showcase'
        ],
        'Use-case Based Themes': [
            'Customer behavior analysis workflows',
            'Predictive modeling use cases',
            'Trend identification processes',
            'Performance monitoring scenarios'
        ],
        'Demographic Themes': [
            'SaaS company analytics solutions',
            'Enterprise-grade AI reporting',
            'Startup-friendly analytics pricing',
            'Marketing team dashboard examples'
        ],
        'Seasonal/Event-Based Themes': [
            'Quarterly reporting preparation',
            'Year-end analytics packages',
            'Monthly performance review templates',
            'Seasonal trend analysis tools'
        ],
        'Solution-Focused Themes': [
            'Data integration challenges solved',
            'Automated reporting time savings',
            'Custom analytics implementation',
            'Scalable dashboard solutions'
        ]
        }
    
        return theme_mapping.get(theme_name, [
            'AI-powered business intelligence',
            'Advanced analytics solutions',
            'Data-driven decision making',
            'Interactive reporting tools'
        ])
    
    def _define_conversion_goals(self) -> List[Dict[str, Any]]:
        """Define conversion goals for Performance Max"""
        return [
            {
                'goal_type': 'Purchase',
                'value': 'Dynamic',
                'attribution_model': 'Data-driven',
                'priority': 'Primary'
            },
            {
                'goal_type': 'Lead Generation',
                'value': 50.0, 
                'attribution_model': 'First-click',
                'priority': 'Secondary'
            }
        ]
    
    def create_shopping_campaign_strategy(self, keyword_data: List[Dict], budget: float) -> Dict[str, Any]:
        """Create Shopping campaign strategy with product prioritization"""
        logger.info("Creating Shopping campaign strategy...")
        
        # Filter and prioritize products
        product_keywords = [kw for kw in keyword_data if self._is_product_keyword(kw['keyword'])]
        high_priority_products = self._identify_high_priority_products(product_keywords)
        medium_priority_products = self._identify_medium_priority_products(product_keywords)
    
        # Create product groups with calculated bids
        high_priority_groups = self._create_product_groups(high_priority_products, 'high')
        medium_priority_groups = self._create_product_groups(medium_priority_products, 'medium')
        
        campaigns = []
        
        # High Priority Campaign
        if high_priority_groups:
            high_priority_campaign = {
            'campaign_name': 'Shopping - High Priority',
            'campaign_type': 'Shopping',
            'priority': 'High',
            'daily_budget': (budget * 0.6) / 30,
            'bidding_strategy': 'Target CPA',
            'target_cpa': round(
                sum(pg['target_cpc'] for pg in high_priority_groups) / 
                len(high_priority_groups) / self.conversion_rate, 2),
            'product_groups': high_priority_groups,
            'bid_adjustments': {
                'mobile': 0.9,
                'tablet': 1.0,
                'desktop': 1.1
            }
        }
            campaigns.append(high_priority_campaign)
        
        # Medium Priority Campaign
        if medium_priority_groups:
            medium_priority_campaign = {
            'campaign_name': 'Shopping - Medium Priority',
            'campaign_type': 'Shopping',
            'priority': 'Medium',
            'daily_budget': (budget * 0.4) / 30,
            'bidding_strategy': 'Manual CPC',
            'product_groups': medium_priority_groups,
            'bid_adjustments': {
                'mobile': 1.0,
                'tablet': 1.0,
                'desktop': 1.0
            }
            }
            campaigns.append(medium_priority_campaign)
        
        return {
        'campaigns': campaigns,
        'budget_allocation': {
            'high_priority': round(budget * 0.6, 2),
            'medium_priority': round(budget * 0.4, 2)
        },
        'target_settings': {
            'conversion_rate': self.conversion_rate,
            'target_roas': self.target_roas
        }
    }
    
    def _is_product_keyword(self, keyword: str) -> bool:
        """Check if keyword indicates product intent"""
        product_indicators = ['buy', 'purchase', 'price', 'cost', 'cheap', 'best', 'review', 'compare']
        return any(indicator in keyword.lower() for indicator in product_indicators)
    
    def _identify_high_priority_products(self, keywords: List[Dict]) -> List[Dict]:
        """Identify high-priority products based on keyword metrics"""
        high_priority = []
        
        for kw in keywords:
            volume = kw.get('avg_monthly_searches', 0)
            cpc_high = kw.get('top_page_bid_high', 0)
            comp_score = kw.get('competition_score', 1.0)
            
            priority_score = volume / max(1, cpc_high) * (1 - comp_score)
            
            if priority_score > 500 and volume > 1000:
                kw['priority_score'] = priority_score
                high_priority.append(kw)
        
        return sorted(high_priority, key=lambda x: x['priority_score'], reverse=True)[:15]
    
    def _identify_medium_priority_products(self, keywords: List[Dict]) -> List[Dict]:
        """Identify medium-priority products"""
        high_priority_keywords = set(kw['keyword'] for kw in self._identify_high_priority_products(keywords))
        
        medium_priority = []
        for kw in keywords:
            if kw['keyword'] not in high_priority_keywords and kw.get('avg_monthly_searches', 0) > 500:
                medium_priority.append(kw)
        
        return medium_priority[:25]
    
    def _create_product_groups(self, keywords: List[Dict], priority_level: str) -> List[Dict[str, Any]]:
        """Create product groups based on keywords"""
        product_groups = []
        
        # Group by keyword themes
        grouped_keywords = {}
        for kw in keywords:
            category = self._extract_product_category(kw['keyword'])
            if category not in grouped_keywords:
                grouped_keywords[category] = []
            grouped_keywords[category].append(kw)
    
        for category, group_keywords in grouped_keywords.items():
            if len(group_keywords) < 1: 
                continue
             # Calculate performance metrics
            avg_monthly_searches = sum(kw.get('avg_monthly_searches', 0) for kw in group_keywords) / len(group_keywords)
            avg_top_page_bid_low = sum(kw.get('top_page_bid_low', 0) for kw in group_keywords) / len(group_keywords)
            avg_top_page_bid_high = sum(kw.get('top_page_bid_high', 0) for kw in group_keywords) / len(group_keywords)
            avg_competition = sum(kw.get('competition_score', 0) for kw in group_keywords) / len(group_keywords)

            target_cpc = self._calculate_target_cpc({
            'top_page_bid_high': avg_top_page_bid_high,
            'competition_score': avg_competition
            })
        
            if priority_level == 'high':
                target_cpc *= 1.1  # Increase bids for high priority
            elif priority_level == 'medium':
                target_cpc *= 0.9  # Decrease bids for medium priority
            
            if avg_competition > 0.7:  # High competition
                target_cpc *= 1.05  # Slightly increase bids
            elif avg_competition < 0.3:  # Low competition
                target_cpc *= 0.95  # Slightly decrease bids
        


            product_group = {
            'name': f"{category} Products",
            'category': category,
            'target_cpc': target_cpc,
            'max_cpc': min(target_cpc * 1.2, avg_top_page_bid_high * 1.5),  
            'performance_metrics': {
                'avg_monthly_searches': round(avg_monthly_searches),
                'top_page_bid_low': round(avg_top_page_bid_low, 2),
                'top_page_bid_high': round(avg_top_page_bid_high, 2),
                'competition_level': round(avg_competition, 2),
                'estimated_conversion_rate': self.conversion_rate,
                'estimated_cpa': round(target_cpc / self.conversion_rate, 2)
            },
            'keywords': [kw['keyword'] for kw in group_keywords]
        }
        
            product_groups.append(product_group)

        return sorted(product_groups, key=lambda x: x['performance_metrics']['avg_monthly_searches'], reverse=True)
    
    def _generate_shopping_negative_keywords(self, keyword_data: List[Dict]) -> List[str]:
        """Generate negative keywords for Shopping campaigns using Gemini if available"""
        keyword_text = ', '.join(kw['keyword'] for kw in keyword_data)
        
        if self.model:
            prompt = f"""Based on these keywords: {keyword_text}
Generate 15 negative keywords for Google Shopping campaigns to avoid low-intent or irrelevant traffic.
Output as a bullet list."""
            try:
                response = self.model.generate_content(prompt)
                negatives = [line.strip('- ').strip() for line in response.text.split('\n') if line.strip()]
                return negatives[:15]
            except Exception as e:
                logger.error(f"Error generating negative keywords with Gemini: {e}")
        
        return [
            'free',
            'cheap',
            'wholesale',
            'job',
            'jobs',
            'career',
            'diy',
            'how to',
            'tutorial',
            'review',
            'used',
            'second hand'
        ]
    
    def _create_merchant_center_recommendations(self, keywords: List[Dict]) -> Dict[str, Any]:
        """Create Merchant Center optimization recommendations"""
        return {
            'title_optimization': {
                'include_keywords': [kw['keyword'] for kw in keywords[:10]],
                'title_length': '150 characters max',
                'include_brand': True,
                'include_key_attributes': ['size', 'color', 'material']
            },
            'description_optimization': {
                'keyword_density': '2-3%',
                'include_benefits': True,
                'use_bullet_points': True,
                'include_specifications': True
            },
            'image_optimization': {
                'image_quality': 'High resolution minimum 800x800px',
                'multiple_angles': True,
                'lifestyle_images': True,
                'consistent_background': True
            }
        }
    
    def _create_feed_optimization_recommendations(self, keywords: List[Dict]) -> Dict[str, Any]:
        """Create product feed optimization recommendations"""
        return {
            'data_quality': {
                'completeness_target': '95%+',
                'required_fields': ['title', 'description', 'price', 'availability', 'condition'],
                'recommended_fields': ['brand', 'gtin', 'mpn', 'custom_labels']
            },
            'categorization': {
                'use_google_taxonomy': True,
                'specific_categories': True,
                'custom_product_types': True
            },
            'pricing_strategy': {
                'competitive_pricing': 'Monitor competitor prices',
                'promotional_pricing': 'Use sale_price attribute',
                'dynamic_pricing': 'Update prices regularly'
            }
        }
    
    def generate_campaign_report(self, campaigns: List[CampaignStructure], 
                               pmax_strategy: Dict[str, Any], 
                               shopping_strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive campaign report"""
        
        total_daily_budget = sum(c.daily_budget for c in campaigns) + pmax_strategy['daily_budget']
        if shopping_strategy.get('campaigns'):
            total_daily_budget += sum(c['daily_budget'] for c in shopping_strategy['campaigns'])
        
        report = {
            'executive_summary': {
                'total_campaigns': len(campaigns) + 1 + len(shopping_strategy.get('campaigns', [])),
                'total_daily_budget': round(total_daily_budget, 2),
                'total_monthly_budget': round(total_daily_budget * 30, 2),
                'average_target_roas': round(sum(c.target_roas for c in campaigns) / len(campaigns), 2),
                'campaign_distribution': {
                    'search_campaigns': len(campaigns),
                    'performance_max_campaigns': 1,
                    'shopping_campaigns': len(shopping_strategy.get('campaigns', []))
                }
            },
            'search_campaigns': [asdict(c) for c in campaigns],
            'performance_max_strategy': pmax_strategy,
            'shopping_strategy': shopping_strategy,
            'recommendations': {
                'launch_sequence': [
                    '1. Start with Brand Protection campaign',
                    '2. Launch High Performance Search campaign',
                    '3. Activate Shopping campaigns',
                    '4. Enable Performance Max after 2 weeks of data',
                    '5. Scale Generic and Competitor campaigns based on performance'
                ],
                'optimization_schedule': {
                    'daily': 'Monitor bid adjustments and budget pacing',
                    'weekly': 'Review search term reports and add negative keywords',
                    'bi_weekly': 'Optimize ad copy and landing pages',
                    'monthly': 'Analyze campaign performance and reallocate budgets'
                },
                'success_metrics': {
                    'primary': 'ROAS >= target ROAS',
                    'secondary': ['CPA within target', 'Quality Score >= 7', 'Impression Share > 80%']
                }
            }
        }
        
        return report

def save_campaign_structure(report: Dict[str, Any], filename: str = "campaign_structure.json"):
    """Save campaign structure to file"""
    with open(filename, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info(f"Campaign structure saved to {filename}")

def export_to_google_ads_editor(campaigns: List[CampaignStructure], filename: str = "google_ads_import.csv"):
    """Export campaigns to Google Ads Editor format"""
    
    rows = []
    
    for campaign in campaigns:
        rows.append({
            'Campaign': campaign.campaign_name,
            'Campaign Type': campaign.campaign_type,
            'Campaign Status': 'Enabled',
            'Budget': campaign.daily_budget * 30, 
            'Bid Strategy': 'Target ROAS' if campaign.target_roas > 0 else 'Target CPA'
        })
        
        for ad_group in campaign.ad_groups:
            for keyword in ad_group.get('keywords', []):
                rows.append({
                    'Campaign': campaign.campaign_name,
                    'Ad Group': ad_group['name'],
                    'Keyword': keyword.get('keyword', ''),
                    'Match Type': 'Phrase',  # Default
                    'Max CPC': ad_group.get('max_cpc', 1.0),
                    'Status': 'Enabled'
                })
    
    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)
    logger.info(f"Google Ads Editor import file saved to {filename}")