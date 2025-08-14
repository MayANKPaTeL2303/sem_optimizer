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

logger = logging.getLogger(__name__)

@dataclass
class CampaignStructure:
    campaign_name: str
    campaign_type: str  # Search, Shopping, Performance Max
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
        
        # Campaign 1: High-Performance Keywords
        if high_performance_keywords:
            high_perf_campaign = self._create_high_performance_campaign(
                high_performance_keywords, budgets['search_ads'] * 0.4
            )
            campaigns.append(high_perf_campaign)
        
        # Campaign 2: Brand Protection
        if brand_keywords:
            brand_campaign = self._create_brand_campaign(
                brand_keywords, budgets['search_ads'] * 0.2
            )
            campaigns.append(brand_campaign)
        
        # Campaign 3: Competitor Targeting
        if competitor_keywords:
            competitor_campaign = self._create_competitor_campaign(
                competitor_keywords, budgets['search_ads'] * 0.2
            )
            campaigns.append(competitor_campaign)
        
        # Campaign 4: Generic Terms
        if generic_keywords:
            generic_campaign = self._create_generic_campaign(
                generic_keywords, budgets['search_ads'] * 0.2
            )
            campaigns.append(generic_campaign)
        
        return campaigns
    
    def _filter_high_performance_keywords(self, keywords: List[Dict]) -> List[Dict]:
        """Filter keywords with high commercial intent and good volume/competition ratio"""
        high_performance = []
        
        for kw in keywords:
            # Calculate performance score
            volume = kw.get('avg_monthly_searches', 0)
            comp_score = kw.get('competition_score', 1.0)
            cpc_low = kw.get('top_page_bid_low', 0)
            
            # Performance score: high volume, low competition, reasonable CPC
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
            target_roas=self.target_roas * 1.2,  # Higher ROAS expectation
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
        target_cpa = avg_cpc / (self.conversion_rate * 2)  # Higher conversion rate expected
        
        return CampaignStructure(
            campaign_name="Brand Protection",
            campaign_type="Search",
            daily_budget=budget / 30,
            target_cpa=target_cpa,
            target_roas=self.target_roas * 1.5,  # Brand terms should perform better
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
        target_cpa = avg_cpc / (self.conversion_rate * 0.8)  # Lower conversion rate expected
        
        return CampaignStructure(
            campaign_name="Competitor Targeting",
            campaign_type="Search",
            daily_budget=budget / 30,
            target_cpa=target_cpa,
            target_roas=self.target_roas * 0.8,  # Lower ROAS acceptable
            ad_groups=ad_groups
        )
    
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
        
        # Group by CPC ranges for better bid management
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
                themes['Product Terms'].append(kw)  # Default
        
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
        
        # Estimate CTR based on avg position (simplified)
        estimated_ctr = 0.02  # 2% baseline CTR
        if avg_cpc > 3.0:
            estimated_ctr *= 1.5  # Higher CPC usually means better position
        elif avg_cpc < 1.0:
            estimated_ctr *= 0.7  # Lower CPC might mean lower position
        
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
                'target_keywords': [kw['keyword'] for kw in theme_keywords[:10]],  # Top 10 per theme
                'audience_signals': self._generate_audience_signals(theme_keywords),
                'budget_allocation': theme_budget / 30,  # Daily budget
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
            'Product Focus': [],
            'Service Focus': [],
            'Brand Focus': [],
            'Local Focus': [],
            'Problem Solving': []
        }
        
        for kw in keywords:
            keyword_lower = kw['keyword'].lower()
            
            # Product-focused themes
            if any(term in keyword_lower for term in ['product', 'buy', 'purchase', 'order', 'shop']):
                themes['Product Focus'].append(kw)
            # Service-focused themes  
            elif any(term in keyword_lower for term in ['service', 'help', 'support', 'consultation', 'hire']):
                themes['Service Focus'].append(kw)
            # Brand-focused themes
            elif any(term in keyword_lower for term in ['brand', 'company', 'official', 'reviews']):
                themes['Brand Focus'].append(kw)
            # Location-focused themes
            elif any(term in keyword_lower for term in ['near', 'local', 'around', 'nearby', 'in']):
                themes['Local Focus'].append(kw)
            # Problem-solving themes
            elif any(term in keyword_lower for term in ['how to', 'solution', 'fix', 'solve', 'help']):
                themes['Problem Solving'].append(kw)
            else:
                # Default to product focus
                themes['Product Focus'].append(kw)
        
        # Remove empty themes
        return {k: v for k, v in themes.items() if v}
    
    def _generate_audience_signals(self, keywords: List[Dict]) -> List[str]:
        """Generate audience signals based on keyword analysis"""
        signals = []
        
        # Analyze keywords to infer audience interests
        keyword_text = ' '.join(kw['keyword'].lower() for kw in keywords)
        
        if 'fitness' in keyword_text or 'health' in keyword_text:
            signals.extend(['Fitness enthusiasts', 'Health conscious consumers'])
        if 'business' in keyword_text or 'professional' in keyword_text:
            signals.extend(['Business professionals', 'Entrepreneurs'])
        if 'home' in keyword_text or 'house' in keyword_text:
            signals.extend(['Homeowners', 'Home improvement enthusiasts'])
        if 'tech' in keyword_text or 'software' in keyword_text:
            signals.extend(['Technology enthusiasts', 'Software users'])
        
        # Add demographic signals based on keyword characteristics
        avg_cpc = sum(kw.get('top_page_bid_high', 1.0) for kw in keywords) / len(keywords)
        if avg_cpc > 5.0:
            signals.append('High-income demographics')
        
        return signals[:5]  # Limit to top 5 signals
    
    def _generate_creative_themes(self, theme_name: str, keywords: List[Dict]) -> List[str]:
        """Generate creative themes for asset groups"""
        theme_mapping = {
            'Product Focus': [
                'Product benefits and features',
                'Customer testimonials',
                'Product comparisons',
                'Special offers and discounts'
            ],
            'Service Focus': [
                'Service quality and expertise',
                'Customer success stories',
                'Process explanations',
                'Professional credentials'
            ],
            'Brand Focus': [
                'Brand story and values',
                'Company achievements',
                'Customer loyalty',
                'Brand differentiation'
            ],
            'Local Focus': [
                'Local community connection',
                'Geographic service areas',
                'Local testimonials',
                'Proximity benefits'
            ],
            'Problem Solving': [
                'Solution-focused messaging',
                'Before and after scenarios',
                'Expert guidance',
                'Problem identification'
            ]
        }
        
        return theme_mapping.get(theme_name, ['Generic product/service benefits'])
    
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
                'value': 50.0,  # Estimated lead value
                'attribution_model': 'First-click',
                'priority': 'Secondary'
            }
        ]
    
    def create_shopping_campaign_strategy(self, keyword_data: List[Dict], budget: float) -> Dict[str, Any]:
        """Create Shopping campaign strategy with product prioritization"""
        logger.info("Creating Shopping campaign strategy...")
        
        # Analyze keywords for product insights
        product_keywords = [kw for kw in keyword_data if self._is_product_keyword(kw['keyword'])]
        
        # Create priority tiers
        high_priority_products = self._identify_high_priority_products(product_keywords)
        medium_priority_products = self._identify_medium_priority_products(product_keywords)
        
        campaigns = []
        
        # High Priority Campaign (60% of budget)
        if high_priority_products:
            high_priority_campaign = {
                'campaign_name': 'Shopping - High Priority',
                'campaign_type': 'Shopping',
                'campaign_subtype': 'Standard Shopping',
                'priority': 'High',
                'daily_budget': (budget * 0.6) / 30,
                'bidding_strategy': 'Target ROAS',
                'target_roas': self.target_roas * 1.2,
                'product_groups': self._create_product_groups(high_priority_products, 'high'),
                'negative_keywords': self._generate_shopping_negative_keywords(),
                'bid_adjustments': {
                    'mobile': 0.9,
                    'tablet': 1.0,
                    'desktop': 1.1
                }
            }
            campaigns.append(high_priority_campaign)
        
        # Medium Priority Campaign (40% of budget)
        if medium_priority_products:
            medium_priority_campaign = {
                'campaign_name': 'Shopping - Medium Priority',
                'campaign_type': 'Shopping',
                'campaign_subtype': 'Standard Shopping',
                'priority': 'Medium',
                'daily_budget': (budget * 0.4) / 30,
                'bidding_strategy': 'Enhanced CPC',
                'target_cpa': sum(kw.get('top_page_bid_high', 1.0) for kw in medium_priority_products) / len(medium_priority_products) / self.conversion_rate,
                'product_groups': self._create_product_groups(medium_priority_products, 'medium'),
                'negative_keywords': self._generate_shopping_negative_keywords(),
                'bid_adjustments': {
                    'mobile': 1.0,
                    'tablet': 1.0,
                    'desktop': 1.0
                }
            }
            campaigns.append(medium_priority_campaign)
        
        return {
            'campaigns': campaigns,
            'merchant_center_optimization': self._create_merchant_center_recommendations(keyword_data),
            'feed_optimization': self._create_feed_optimization_recommendations(keyword_data)
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
            
            # High volume, reasonable CPC, not too competitive
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
            # Simple grouping by first word (can be enhanced)
            first_word = kw['keyword'].split()[0].lower()
            if first_word not in grouped_keywords:
                grouped_keywords[first_word] = []
            grouped_keywords[first_word].append(kw)
        
        for group_name, group_keywords in grouped_keywords.items():
            if len(group_keywords) >= 2:  # Only create groups with multiple keywords
                avg_cpc = sum(kw.get('top_page_bid_high', 1.0) for kw in group_keywords) / len(group_keywords)
                
                # Adjust CPC based on priority
                if priority_level == 'high':
                    max_cpc = avg_cpc * 1.1
                else:
                    max_cpc = avg_cpc * 0.9
                
                product_group = {
                    'name': f"{group_name.title()} Products",
                    'max_cpc': round(max_cpc, 2),
                    'keywords': [kw['keyword'] for kw in group_keywords],
                    'estimated_performance': {
                        'avg_monthly_searches': sum(kw.get('avg_monthly_searches', 0) for kw in group_keywords),
                        'competition_level': sum(kw.get('competition_score', 0) for kw in group_keywords) / len(group_keywords)
                    }
                }
                
                product_groups.append(product_group)
        
        return product_groups
    
    def _generate_shopping_negative_keywords(self) -> List[str]:
        """Generate negative keywords for Shopping campaigns"""
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
        # Campaign row
        rows.append({
            'Campaign': campaign.campaign_name,
            'Campaign Type': campaign.campaign_type,
            'Campaign Status': 'Enabled',
            'Budget': campaign.daily_budget * 30,  # Monthly budget
            'Bid Strategy': 'Target ROAS' if campaign.target_roas > 0 else 'Target CPA'
        })
        
        # Ad group rows
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