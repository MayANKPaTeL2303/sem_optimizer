"""
Main execution script for the SEM Campaign Optimizer
Run this script to execute the complete optimization process
"""

import sys
import os
import yaml
import logging
import json
from datetime import datetime
import pandas as pd
from traceback import format_exc

# Import our modules
from sem_optimizer import SEMCampaignOptimizer
from keyword_planner import EnhancedKeywordPlanner, create_keyword_expansion_report
from campaign_builder import AdvancedCampaignBuilder, save_campaign_structure, export_to_google_ads_editor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sem_optimizer.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class SEMOptimizationSuite:
    """
    Complete SEM optimization suite that orchestrates all components
    """
    
    def __init__(self, config_path: str = 'config.yaml'):
        """Initialize the optimization suite"""
        self.config_path = config_path
        self.config = self._load_config()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize components
        self.sem_optimizer = SEMCampaignOptimizer(config_path)
        self.keyword_planner = EnhancedKeywordPlanner()
        self.campaign_builder = AdvancedCampaignBuilder(self.config)
        
    def _load_config(self) -> dict:
        """Load and validate configuration"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
            
            # Validate required fields
            required_fields = [
                'inputs.brand_website',
                'inputs.competitor_website',
                'inputs.service_locations',
                'inputs.budgets'
            ]
            
            for field in required_fields:
                keys = field.split('.')
                current = config
                for key in keys:
                    if key not in current:
                        raise ValueError(f"Missing required config field: {field}")
                    current = current[key]
            
            # Validate budgets
            budgets = config['inputs']['budgets']
            for budget_type, amount in budgets.items():
                if not isinstance(amount, (int, float)) or amount < 0:
                    raise ValueError(f"Invalid budget for {budget_type}: {amount} (must be non-negative number)")
            
            # Validate service_locations
            if not config['inputs']['service_locations']:
                raise ValueError("service_locations cannot be empty")
            
            # Add fallback competitor URL
            config['inputs']['fallback_competitor_website'] = config['inputs'].get(
                'fallback_competitor_website', 'https://www.optimumnutrition.com'
            )
            
            # Add keyword expansion limit
            config['advanced'] = config.get('advanced', {})
            config['advanced']['keyword_expansion_limit'] = config['advanced'].get('keyword_expansion_limit', 100)
            
            logger.info("Configuration loaded and validated successfully")
            return config
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
    
    def run_complete_optimization(self) -> dict:
        """Run the complete SEM optimization process"""
        logger.info("="*60)
        logger.info("STARTING COMPLETE SEM OPTIMIZATION")
        logger.info("="*60)
        
        try:
            results = {}
            
            logger.info("Phase 1: Website Analysis and Keyword Discovery")
            basic_results = self.sem_optimizer.run_campaign_optimization()
            results['basic_analysis'] = basic_results
            
            if basic_results['website_analysis']['competitor_website'].get('error'):
                logger.warning(f"Competitor website scrape failed: {basic_results['website_analysis']['competitor_website']['error']}")
            
            if basic_results['campaign_summary'].get('used_fallback_categorization'):
                logger.warning("Used fallback method for keyword categorization due to LLM failure")
            
            logger.info("Phase 2: Enhanced Keyword Analysis")
            all_keywords = []
            for ad_group in basic_results['search_campaign']['ad_groups']:
                all_keywords.extend([kw['keyword'] for kw in ad_group['keywords']])
            
            max_expansions = self.config['advanced']['keyword_expansion_limit']
            expanded_keywords = self.keyword_planner.expand_keywords(all_keywords, max_expansions=max_expansions)
            
            enhanced_keyword_metrics = self.keyword_planner.get_keyword_metrics(expanded_keywords)
            
            enhanced_keywords_dict = []
            for kw_metric in enhanced_keyword_metrics:
                enhanced_keywords_dict.append({
                    'keyword': kw_metric.keyword,
                    'avg_monthly_searches': kw_metric.search_volume,
                    'competition': kw_metric.competition,
                    'competition_score': kw_metric.competition_score,
                    'top_page_bid_low': kw_metric.cpc_low,
                    'top_page_bid_high': kw_metric.cpc_high,
                    'trend': kw_metric.trend
                })
            
            keyword_report = create_keyword_expansion_report(
                enhanced_keyword_metrics, 
                get_unique_filename(f"keyword_expansion_report_{self.timestamp}.json")
            )
            results['keyword_analysis'] = keyword_report
            
            logger.info("Phase 3: Advanced Campaign Building")
            
            search_campaigns = self.campaign_builder.build_search_campaigns(
                enhanced_keywords_dict, 
                self.config['inputs']['budgets']
            )
            
            pmax_strategy = self.campaign_builder.create_performance_max_strategy(
                enhanced_keywords_dict,
                self.config['inputs']['budgets']['pmax_ads']
            )
            
            shopping_strategy = self.campaign_builder.create_shopping_campaign_strategy(
                enhanced_keywords_dict,
                self.config['inputs']['budgets']['shopping_ads']
            )
            
            logger.info("Phase 4: Generating Comprehensive Report")
            campaign_report = self.campaign_builder.generate_campaign_report(
                search_campaigns,
                pmax_strategy,
                shopping_strategy
            )
            
            results['advanced_campaigns'] = campaign_report
            
            logger.info("Phase 5: Exporting Results")
            self._export_all_results(results, search_campaigns, pmax_strategy, shopping_strategy)
            
            logger.info("="*60)
            logger.info("SEM OPTIMIZATION COMPLETED SUCCESSFULLY")
            logger.info("="*60)
            
            return results
            
        except Exception as e:
            logger.error(f"Error during optimization: {e}")
            logger.error(format_exc())
            raise
    
    def _export_all_results(self, results: dict, search_campaigns, pmax_strategy, shopping_strategy):
        """Export all results to various formats"""
        output_filename = get_unique_filename(f"sem_optimization_results_{self.timestamp}.json")
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Complete results saved to: {output_filename}")
        
        campaign_structure_filename = get_unique_filename(f"campaign_structure_{self.timestamp}.json")
        save_campaign_structure(results['advanced_campaigns'], campaign_structure_filename)
        
        google_ads_filename = get_unique_filename(f"google_ads_import_{self.timestamp}.csv")
        export_to_google_ads_editor(search_campaigns, google_ads_filename)
        
        self._create_keyword_summary_csv(results)
        
        # 5. Create budget allocation summary
        self._create_budget_summary(results)
        
        # 6. Generate executive summary
        self._create_executive_summary(results)
    
    def _create_keyword_summary_csv(self, results: dict):
        all_keywords = []
        
        # Extract keywords from advanced campaigns
        if 'advanced_campaigns' in results and 'search_campaigns' in results['advanced_campaigns']:
            for campaign in results['advanced_campaigns']['search_campaigns']:
                campaign_name = campaign['campaign_name']
                for ad_group in campaign.get('ad_groups', []):
                    ad_group_name = ad_group['name']
                    for keyword in ad_group.get('keywords', []):
                        keyword_data = {
                            'campaign': campaign_name,
                            'ad_group': ad_group_name,
                            'keyword': keyword['keyword'],
                            'search_volume': keyword.get('avg_monthly_searches', 0),
                            'competition': keyword.get('competition', ''),
                            'cpc_low': keyword.get('top_page_bid_low', 0),
                            'cpc_high': keyword.get('top_page_bid_high', 0),
                            'max_cpc': ad_group.get('max_cpc', 0),
                            'match_types': ', '.join(ad_group.get('match_types', [])),
                            'bid_strategy': ad_group.get('bid_strategy', '')
                        }
                        all_keywords.append(keyword_data)
        
        if all_keywords:
            df = pd.DataFrame(all_keywords)
            filename = get_unique_filename(f"keyword_summary_{self.timestamp}.csv")
            df.to_csv(filename, index=False, encoding='utf-8')
            logger.info(f"Keyword summary saved to: {filename}")
        else:
            logger.warning("No keywords found for CSV export")
    
    def _create_budget_summary(self, results: dict):
        """Create budget allocation summary"""
        budget_summary = {
            'timestamp': self.timestamp,
            'total_monthly_budget': sum(self.config['inputs']['budgets'].values()),
            'budget_allocation': self.config['inputs']['budgets'],
            'campaign_budget_distribution': {}
        }
        
        if 'advanced_campaigns' in results:
            exec_summary = results['advanced_campaigns'].get('executive_summary', {})
            budget_summary['campaign_budget_distribution'] = {
                'total_daily_budget': exec_summary.get('total_daily_budget', 0),
                'total_monthly_budget': exec_summary.get('total_monthly_budget', 0),
                'campaign_count': exec_summary.get('total_campaigns', 0)
            }
        
        filename = get_unique_filename(f"budget_summary_{self.timestamp}.json")
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(budget_summary, f, indent=2)
        logger.info(f"Budget summary saved to: {filename}")
    
    def _create_executive_summary(self, results: dict):
        """Create executive summary document"""
        summary_content = []
        
        summary_content.append("# SEM Campaign Optimization - Executive Summary")
        summary_content.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary_content.append(f"**Brand Website:** {self.config['inputs']['brand_website']}")
        summary_content.append(f"**Competitor Website:** {self.config['inputs']['competitor_website']}")
        summary_content.append("")
        
        if 'basic_analysis' in results and 'website_analysis' in results['basic_analysis']:
            if results['basic_analysis']['website_analysis']['competitor_website'].get('error'):
                summary_content.append("**Warning:** Failed to extract content from competitor website. Results may lack competitor insights.")
                logger.warning("Competitor website scrape failed, using brand content only")
        
        if results.get('basic_analysis', {}).get('campaign_summary', {}).get('used_fallback_categorization', False):
            summary_content.append("**Warning:** Used fallback method for keyword categorization due to LLM failure. Keyword groupings may be less precise.")
            logger.warning("Fallback keyword categorization used")
        summary_content.append("")
        
        summary_content.append("## Budget Allocation")
        total_budget = sum(self.config['inputs']['budgets'].values())
        summary_content.append(f"**Total Monthly Budget:** ${total_budget:,.2f}")
        summary_content.append("")
        for budget_type, amount in self.config['inputs']['budgets'].items():
            percentage = (amount / total_budget) * 100
            summary_content.append(f"- **{budget_type.replace('_', ' ').title()}:** ${amount:,.2f} ({percentage:.1f}%)")
        summary_content.append("")
        
        # Campaign Overview
        if 'advanced_campaigns' in results:
            exec_data = results['advanced_campaigns'].get('executive_summary', {})
            summary_content.append("## Campaign Structure")
            summary_content.append(f"**Total Campaigns:** {exec_data.get('total_campaigns', 0)}")
            summary_content.append(f"**Average Target ROAS:** {exec_data.get('average_target_roas', 0):.2f}")
            summary_content.append("")
            
            # Campaign distribution
            dist = exec_data.get('campaign_distribution', {})
            summary_content.append("### Campaign Distribution")
            for camp_type, count in dist.items():
                summary_content.append(f"- **{camp_type.replace('_', ' ').title()}:** {count}")
            summary_content.append("")
        
        # Keyword Insights
        if 'keyword_analysis' in results:
            kw_data = results['keyword_analysis']['summary']
            summary_content.append("## Keyword Analysis")
            summary_content.append(f"**Total Keywords Discovered:** {kw_data.get('total_keywords', 0)}")
            summary_content.append(f"**High Volume Keywords (>5K searches):** {kw_data.get('high_volume_keywords', 0)}")
            summary_content.append(f"**Low CPC Opportunities (<$1.00):** {kw_data.get('low_cpc_opportunities', 0)}")
            summary_content.append(f"**Average Search Volume:** {kw_data.get('avg_search_volume', 0):,.0f}")
            summary_content.append("")
        
        # Recommendations
        if 'advanced_campaigns' in results and 'recommendations' in results['advanced_campaigns']:
            recs = results['advanced_campaigns']['recommendations']
            summary_content.append("## Key Recommendations")
            summary_content.append("")
            
            # Launch sequence
            summary_content.append("### Launch Sequence")
            for step in recs.get('launch_sequence', []):
                summary_content.append(f"- {step}")
            summary_content.append("")
            
            # Success metrics
            summary_content.append("### Success Metrics")
            metrics = recs.get('success_metrics', {})
            summary_content.append(f"**Primary:** {metrics.get('primary', 'N/A')}")
            if 'secondary' in metrics:
                summary_content.append("**Secondary:**")
                for metric in metrics['secondary']:
                    summary_content.append(f"- {metric}")
            summary_content.append("")
        
        # Export Locations
        summary_content.append("## Generated Files")
        summary_content.append("The following files have been generated for your review:")
        summary_content.append(f"- `{get_unique_filename(f'sem_optimization_results_{self.timestamp}.json')}` - Complete results")
        summary_content.append(f"- `{get_unique_filename(f'campaign_structure_{self.timestamp}.json')}` - Detailed campaign structure")
        summary_content.append(f"- `{get_unique_filename(f'google_ads_import_{self.timestamp}.csv')}` - Google Ads Editor import file")
        summary_content.append(f"- `{get_unique_filename(f'keyword_summary_{self.timestamp}.csv')}` - Keyword analysis spreadsheet")
        summary_content.append(f"- `{get_unique_filename(f'budget_summary_{self.timestamp}.json')}` - Budget allocation details")
        
        # Save executive summary
        filename = get_unique_filename(f"executive_summary_{self.timestamp}.md")
        with open(filename, 'w', encoding='utf-8') as f:
            f.write('\n'.join(summary_content))
        logger.info(f"Executive summary saved to: {filename}")
    
    def print_optimization_summary(self, results: dict):
        """Print a nice summary to console"""
        print("\n" + "="*80)
        print(">>> SEM CAMPAIGN OPTIMIZATION COMPLETE!")
        print("="*80)
        
        # Warnings for partial failures
        if 'basic_analysis' in results and 'website_analysis' in results['basic_analysis']:
            if results['basic_analysis']['website_analysis']['competitor_website'].get('error'):
                print("!!! Warning: Failed to extract content from competitor website")
        
        if results.get('basic_analysis', {}).get('campaign_summary', {}).get('used_fallback_categorization', False):
            print("!!! Warning: Used fallback method for keyword categorization")
        
        # Basic stats
        if 'keyword_analysis' in results:
            kw_summary = results['keyword_analysis']['summary']
            print(f">>> Keywords Analyzed: {kw_summary.get('total_keywords', 0)}")
            print(f">>> High Volume Keywords: {kw_summary.get('high_volume_keywords', 0)}")
            print(f">>> Low CPC Opportunities: {kw_summary.get('low_cpc_opportunities', 0)}")
        
        if 'advanced_campaigns' in results:
            exec_data = results['advanced_campaigns'].get('executive_summary', {})
            print(f">>> Total Campaigns Created: {exec_data.get('total_campaigns', 0)}")
            print(f">>> Total Monthly Budget: ${exec_data.get('total_monthly_budget', 0):,.2f}")
            print(f">>> Average Target ROAS: {exec_data.get('average_target_roas', 0):.2f}")
        
        print("\n>>> Generated Files:")
        print(f"   - Executive Summary: {get_unique_filename(f'executive_summary_{self.timestamp}.md')}")
        print(f"   - Keyword Data: {get_unique_filename(f'keyword_summary_{self.timestamp}.csv')}")
        print(f"   - Google Ads Import: {get_unique_filename(f'google_ads_import_{self.timestamp}.csv')}")
        print(f"   - Complete Results: {get_unique_filename(f'sem_optimization_results_{self.timestamp}.json')}")
        
        print("\n>>> Ready to launch your optimized SEM campaigns!")
        print("="*80)

def main():
    """Main function"""
    print(">>> Starting SEM Campaign Optimization Suite...")
    
    # Check if config file exists
    config_file = 'config.yaml'
    if not os.path.exists(config_file):
        print(f"!!! Configuration file '{config_file}' not found!")
        print(">>> Please create a config.yaml file using the provided template.")
        print(">>> See setup instructions for details.")
        sys.exit(1)
    
    try:
        # Initialize optimization suite
        optimizer_suite = SEMOptimizationSuite(config_file)
        
        # Run complete optimization
        results = optimizer_suite.run_complete_optimization()
        
        optimizer_suite.print_optimization_summary(results)
        
        return results
        
    except KeyboardInterrupt:
        print("\n>>> Optimization interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n!!! Optimization failed: {e}")
        logger.error(f"Fatal error: {e}")
        logger.error(format_exc())
        sys.exit(1)

def run_quick_analysis():
    """Quick analysis mode for faster testing"""
    print(">>> Running Quick Analysis Mode...")
    
    try:
        # Initialize with basic optimizer only
        optimizer = SEMCampaignOptimizer('config.yaml')
        
        # Run basic optimization
        results = optimizer.run_campaign_optimization()
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = get_unique_filename(f"quick_analysis_{timestamp}.json")
        optimizer.save_results(results, filename)
        
        print(">>> Quick analysis complete!")
        print(f">>> Keywords found: {results['campaign_summary']['total_keywords_filtered']}")
        print(f">>> Ad groups created: {results['campaign_summary']['total_ad_groups']}")
        
        return results
        
    except Exception as e:
        print(f"!!! Quick analysis failed: {e}")
        sys.exit(1)

def get_unique_filename(base_filename: str) -> str:
    """Generate a unique filename to prevent overwrites"""
    if not os.path.exists(base_filename):
        return base_filename
    base, ext = os.path.splitext(base_filename)
    counter = 1
    while os.path.exists(f"{base}_{counter}{ext}"):
        counter += 1
    return f"{base}_{counter}{ext}"

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SEM Campaign Optimization Suite")
    parser.add_argument('--quick', action='store_true', help='Run quick analysis mode')
    parser.add_argument('--config', default='config.yaml', help='Configuration file path')
    
    args = parser.parse_args()
    
    if args.quick:
        run_quick_analysis()
    else:
        main()