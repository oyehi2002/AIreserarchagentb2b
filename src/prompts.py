class B2BBusinessPrompts:
    TOOL_EXTRACTION_SYSTEM = """You are a B2B business analyst and tool researcher. Extract specific business tools, platforms, services, or solutions from articles.
    Focus on actual products/services that businesses can implement or use, including SaaS platforms, enterprise solutions, productivity tools, and business services, not just technical developer tools."""

    @staticmethod
    def tool_extraction_user(query: str, content: str) -> str:
        return f"""Query: {query}
                Article Content: {content}

                Extract a list of specific business tool/service names mentioned in this content that are relevant to "{query}".

                Rules:
                - Only include actual business product names, not generic terms or concepts
                - Focus on tools/services businesses can directly purchase, implement, or use
                - Include SaaS platforms, enterprise software, business services, and productivity tools
                - Include both SMB and enterprise-level solutions  
                - Limit to the 5 most relevant tools for business use
                - Return just the tool names, one per line, no descriptions

                Example format:
                Salesforce
                HubSpot
                Slack
                Asana
                Zendesk"""

    # Recommendation system prompt
    RECOMMENDATIONS_SYSTEM = """You are a B2B business development VP with 20+ years' experience, providing direct and actionable technology recommendations.
                                Keep responses concise but informative. Focus on business impact, cost considerations, and strategic advantage."""

    @staticmethod
    def recommendations_user(query: str, company_data: str) -> str:
        return f"""Business Query: Find alternatives to "{query}"
                   Alternative Tools Analyzed: {company_data}
                   
                   Provide a comprehensive recommendation covering:
                   - Top 2-3 alternatives to "{query}" and why they're better
                   - Key cost and enterprise licensing considerations
                   - Main competitive advantages for B2B buyers
                   - Specific use cases where alternatives excel over "{query}"
                   
                   Be specific about business value and strategic advantages.
                   Do NOT recommend "{query}" itself - only focus on alternatives."""
