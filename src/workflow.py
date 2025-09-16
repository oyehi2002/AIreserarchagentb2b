from typing import Dict, Any
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from .models import ResearchState, CompanyInfo
from .firecrawl import FirecrawlService
from .prompts import B2BBusinessPrompts


class B2BBusinessWorkflow:
    def __init__(self):
        self.firecrawl = FirecrawlService()
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
        self.prompts = B2BBusinessPrompts()
        self.workflow = self._build_workflow()

    def _build_workflow(self):
        graph = StateGraph(ResearchState)
        graph.add_node("extract_tools", self._extract_tools_step)
        graph.add_node("research", self._research_step)
        graph.add_node("analyze", self._analyze_step)
        graph.set_entry_point("extract_tools")
        graph.add_edge("extract_tools", "research")
        graph.add_edge("research", "analyze")
        graph.add_edge("analyze", END)
        return graph.compile()

    def _get_search_data(self, search_results):
        """Helper method to extract data from search results regardless of structure"""
        if not search_results:
            return []

        if hasattr(search_results, 'data'):
            return search_results.data
        elif isinstance(search_results, list):
            return search_results
        elif hasattr(search_results, 'results'):
            return search_results.results
        else:
            return []

    def _generate_alternatives(self, query: str) -> list:
        """Generate alternatives using AI knowledge for any B2B tool"""
        messages = [
            SystemMessage(content="""You are a B2B software expert. Generate 5 direct competitors and alternatives for any given B2B tool/software.

    Rules:
    - Only return actual company/product names
    - One name per line
    - No descriptions or explanations
    - Focus on well-known alternatives
    - Don't include the original tool"""),
            HumanMessage(content=f"List 5 popular alternatives to: {query}")
        ]

        try:
            response = self.llm.invoke(messages)
            alternatives = [
                name.strip()
                for name in response.content.strip().split("\n")
                if name.strip() and not name.strip().lower() == query.lower() and len(name.strip()) > 2
            ]

            return alternatives[:5]

        except Exception as e:
            print("Error")

    def _extract_tools_step(self, state: ResearchState) -> Dict[str, Any]:
        print(f"🔍 Finding articles about: {state.query}")

        article_query = f"{state.query} B2B top comparison best alternatives"
        search_results = self.firecrawl.search_companies(
            article_query, num_results=3)

        all_content = ""
        search_data = self._get_search_data(search_results)

        if search_data:
            for result in search_data:
                url = result.get("url", "") if isinstance(result, dict) else ""
                if url:
                    scraped = self.firecrawl.scrape_company_pages(url)
                    if scraped and hasattr(scraped, 'markdown'):
                        all_content += scraped.markdown[:1500] + "\n\n"
                    elif scraped and isinstance(scraped, dict):
                        all_content += scraped.get('markdown',
                                                   '')[:1500] + "\n\n"

        if not all_content.strip():
            print("⚠️ No content extracted from articles, using query directly")
            all_content = f"Looking for alternatives to {state.query}"

        messages = [
            SystemMessage(content=self.prompts.TOOL_EXTRACTION_SYSTEM),
            HumanMessage(content=self.prompts.tool_extraction_user(
                state.query, all_content))
        ]

        try:
            response = self.llm.invoke(messages)
            tool_names = [
                name.strip()
                for name in response.content.strip().split("\n")
                if name.strip() and not name.strip().lower() == state.query.lower()
            ]
            print(f"🔍 Extracted tools: {', '.join(tool_names[:5])}")
            return {"extracted_tools": tool_names[:5]}
        except Exception as e:
            print(f"❌ Tool extraction error: {e}")
            return {"extracted_tools": []}

    def _research_step(self, state: ResearchState) -> Dict[str, Any]:
        extracted_tools = getattr(state, "extracted_tools", [])

        if not extracted_tools:
            print("⚠️ No extracted tools found, generating alternatives with AI")
            # Generating alternatives using AI instead
            alternatives = self._generate_alternatives(state.query)
            tool_names = alternatives[:4]
        else:
            tool_names = extracted_tools[:4]

        if not tool_names:
            print("❌ No tools found to research")
            return {"companies": []}

        print(
            f"🔬 Generating information for tools: {', '.join(tool_names)}")

        companies = []
        for tool_name in tool_names:
            print(f"  📊 Analyzing: {tool_name}")
            company = self._generate_company_info(tool_name, state.query)
            companies.append(company)

        print(f"✅ Successfully researched {len(companies)} companies")
        return {"companies": companies}

    def _generate_company_info(self, company_name: str, original_query: str) -> CompanyInfo:
        """Generate company information using AI knowledge instead of web scraping"""

        messages = [
            SystemMessage(content="""You are a B2B software expert. Generate accurate information about the given company/tool based on your knowledge. 
            Focus on:
            - Brief description (2-3 sentences)
            - Pricing model (Free/Freemium/Paid/Enterprise)
            - Key integration capabilities
            - Website URL (if known)
            
            Keep descriptions concise and factual."""),

            HumanMessage(content=f"""Generate information for: {company_name}
            
            This is being researched as an alternative to: {original_query}
            
            Please provide:
            1. Company description
            2. Pricing model
            3. Integration capabilities
            4. Website URL""")
        ]

        try:
            structured_llm = self.llm.with_structured_output(CompanyInfo)
            analysis = structured_llm.invoke(messages)
            return analysis
        except Exception as e:
            print(f"❌ AI generation error for {company_name}: {e}")

            # Fallback with basic info
            return CompanyInfo(
                name=company_name,
                description=f"{company_name} is a business communication and collaboration platform that serves as an alternative to {original_query}.",
                website=f"https://{company_name.lower().replace(' ', '')}.com",
                pricing_model="Freemium",
                integration_capabilities="API integrations, third-party apps, webhooks"
            )

    def _analyze_step(self, state: ResearchState) -> Dict[str, Any]:
        print("🧠 Generating recommendations...")

        if not state.companies:
            return {"analysis": f"No alternative tools found for {state.query}. Please try a different search term."}

        company_data = "\n".join([
            f"Company: {company.name}\nWebsite: {company.website}\nDescription: {company.description}\nPricing: {company.pricing_model}\nIntegrations: {company.integration_capabilities}\n---"
            for company in state.companies
        ])

        messages = [
            SystemMessage(content=self.prompts.RECOMMENDATIONS_SYSTEM),
            HumanMessage(content=self.prompts.recommendations_user(
                state.query, company_data))
        ]

        try:
            response = self.llm.invoke(messages)
            return {"analysis": response.content}
        except Exception as e:
            print(f"❌ Analysis generation error: {e}")
            return {"analysis": f"Unable to generate analysis for {state.query} alternatives."}

    def run(self, query: str) -> ResearchState:
        initial_state = ResearchState(query=query)
        try:
            final_state = self.workflow.invoke(initial_state)
            return ResearchState(**final_state)
        except Exception as e:
            print(f"❌ Workflow error: {e}")
            return ResearchState(
                query=query,
                companies=[],
                analysis=f"Workflow failed: {str(e)}"
            )
