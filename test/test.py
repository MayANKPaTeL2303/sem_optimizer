#!/usr/bin/env python

import argparse
import sys
from google.ads.googleads.client import GoogleAdsClient
from google.ads.googleads.errors import GoogleAdsException

# Default: New York (mock in test account)
_DEFAULT_LOCATION_IDS = ["1023191"]
# Default: English
_DEFAULT_LANGUAGE_ID = "1000"

def main(client, customer_id, location_ids, language_id, keyword_texts, page_url):
    keyword_plan_idea_service = client.get_service("KeywordPlanIdeaService")
    keyword_plan_network = client.enums.KeywordPlanNetworkEnum.GOOGLE_SEARCH_AND_PARTNERS
    location_rns = _map_locations_ids_to_resource_names(client, location_ids)
    
    # Language path for test account still works with GoogleAdsService
    language_rn = client.get_service("GoogleAdsService").language_constant_path(language_id)

    if not (keyword_texts or page_url):
        raise ValueError("At least one of keywords or page URL is required, but neither was specified.")

    request = client.get_type("GenerateKeywordIdeasRequest")
    request.customer_id = customer_id
    request.language = language_rn
    request.geo_target_constants = location_rns
    request.include_adult_keywords = False
    request.keyword_plan_network = keyword_plan_network
    request.keyword_annotation.extend([
        client.enums.KeywordPlanKeywordAnnotationEnum.KEYWORD_CONCEPT
    ])

    if not keyword_texts and page_url:
        url_seed = client.get_type("UrlSeed")
        url_seed.url = page_url
        request.url_seed = url_seed

    if keyword_texts and not page_url:
        keyword_seed = client.get_type("KeywordSeed")
        keyword_seed.keywords.extend(keyword_texts)
        request.keyword_seed = keyword_seed

    if keyword_texts and page_url:
        keyword_and_url_seed = client.get_type("KeywordAndUrlSeed")
        keyword_and_url_seed.url = page_url
        keyword_and_url_seed.keywords.extend(keyword_texts)
        request.keyword_and_url_seed = keyword_and_url_seed

    keyword_ideas = keyword_plan_idea_service.generate_keyword_ideas(request=request)
    
    results = []
    for idea in keyword_ideas:
        keyword_text = idea.text
        avg_monthly_searches = idea.keyword_idea_metrics.avg_monthly_searches
        competition = idea.keyword_idea_metrics.competition.name
        results.append({
            "keyword": keyword_text,
            "avg_monthly_searches": avg_monthly_searches,
            "competition": competition
        })
    return results

def _map_locations_ids_to_resource_names(client, location_ids):
    build_resource_name = client.get_service("GeoTargetConstantService").geo_target_constant_path
    return [build_resource_name(location_id) for location_id in location_ids]

if __name__ == "__main__":
    # Load config from google-ads.yaml (with test account developer token & OAuth creds)
    googleads_client = GoogleAdsClient.load_from_storage("google-ads.yaml")

    parser = argparse.ArgumentParser(description="Generates keyword ideas from a list of seed keywords (Test Account).")
    parser.add_argument("-c", "--customer_id", type=str, required=True, help="Test account customer ID.")
    parser.add_argument("-k", "--keyword_texts", nargs="+", type=str, default=[], help="Space-delimited list of starter keywords")
    parser.add_argument("-l", "--location_ids", nargs="+", type=str, default=_DEFAULT_LOCATION_IDS, help="Location criteria IDs")
    parser.add_argument("-i", "--language_id", type=str, default=_DEFAULT_LANGUAGE_ID, help="Language criterion ID.")
    parser.add_argument("-p", "--page_url", type=str, help="A URL related to your business")

    args = parser.parse_args()

    try:
        keyword_list = main(
            googleads_client,
            args.customer_id,
            args.location_ids,
            args.language_id,
            args.keyword_texts,
            args.page_url,
        )
        for kw in keyword_list:
            print(f"{kw['keyword']}: {kw['avg_monthly_searches']} searches, Competition: {kw['competition']}")
    except GoogleAdsException as ex:
        print(f'Request with ID "{ex.request_id}" failed with status "{ex.error.code().name}" and errors:')
        for error in ex.failure.errors:
            print(f'\t{error.message}')
            if error.location:
                for field_path_element in error.location.field_path_elements:
                    print(f"\t\tOn field: {field_path_element.field_name}")
        sys.exit(1)
