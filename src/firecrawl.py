import os
from firecrawl import FirecrawlApp
from dotenv import load_dotenv

load_dotenv()


class SearchResult:
    def __init__(self, data):
        self.data = data


class FirecrawlService:
    def __init__(self):
        api_key = os.getenv("FIRECRAWL_API_KEY")
        if not api_key:
            raise ValueError("Missing FIRECRAWL_API_KEY environment variable")
        self.app = FirecrawlApp(api_key=api_key)

    def search_companies(self, query: str, num_results: int = 5):
        try:
            result = self.app.search(
                query=f"{query} company pricing",
                limit=num_results,
                scrape_options={
                    "formats": ["markdown"]
                }
            )

            # Handle different result structures silently
            if hasattr(result, 'data'):
                return result
            elif isinstance(result, list):
                return SearchResult(result)
            else:
                # Handle SearchData or other types
                data = []
                if hasattr(result, 'web') and result.web:
                    data = result.web
                elif hasattr(result, 'results'):
                    data = result.results
                elif hasattr(result, 'data'):
                    data = result.data

                return SearchResult(data)

        except Exception as e:
            print(f"❌ Search error: {e}")
            return SearchResult([])

    def scrape_company_pages(self, url: str):
        try:
            result = self.app.scrape_url(
                url,
                params={
                    "formats": ["markdown"]
                }
            )
            return result
        except Exception as e:
            print(f"Scrape error: {e}")
            return None
