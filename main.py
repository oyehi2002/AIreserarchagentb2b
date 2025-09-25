from typing import TypedDict, Sequence, Annotated, List
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from operator import add as add_messages
from dotenv import load_dotenv
from tavily import TavilyClient
import json
import os
from pydantic import BaseModel, Field

load_dotenv()
client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


class TargetCompany(BaseModel):
    """Structure for target company information"""
    company_name: str = Field(description="Name of the target company")
    industry: str = Field(description="Industry/sector of the company")
    description: str = Field(description="What the company does")
    why_good_fit: str = Field(
        description="Why they would need the seller's product/service")
    funding_status: str = Field(
        description="Recent funding or financial status")


class ResearchState(TypedDict):
    """State for the research agent"""
    messages: Annotated[Sequence[BaseMessage], add_messages]


# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", temperature=0.4)


@tool
def search_target_companies(product_description: str, target_industry: str = "") -> str:
    """Search for companies that would be good customers for the given product/service"""

    try:
        # Create search query based on product description
        if target_industry:
            search_query = f"{target_industry} companies funding growth expansion {product_description}"
        else:
            # Extract key terms from product description for search
            search_query = f"companies funding Series A B C growth expansion business {product_description}"

        response = client.search(
            query=search_query,
            search_depth="advanced",
            topic="news",
            days=60,
            max_results=10,
            include_answer=True
        )

        companies_found = []

        for result in response.get("results", []):
            companies_found.append({
                "title": result.get("title", ""),
                "content": result.get("content", "")[:600],
                "url": result.get("url", ""),
            })

        return json.dumps({
            "search_query": search_query,
            "companies": companies_found
        }, indent=2)

    except Exception as e:
        return f"Error in target company search: {str(e)}"


# Tools list
tools = [search_target_companies]
llm_with_tools = llm.bind_tools(tools)


def research_agent(state: ResearchState) -> ResearchState:
    messages = state["messages"]

    system_message = SystemMessage(content="""You are a B2B lead researcher. When given a product/service description:

1. Use the search_target_companies tool to find companies that recently raised funding
2. For each company analyze and find atmost 5 companies and explain:
   - Company name and business
   - Why they would buy this product/service
   - Their funding status
3. Be specific about the business fit

Always search first, then analyze the results.
""")

    response = llm_with_tools.invoke([system_message] + list(messages))
    return {"messages": [response]}


def should_continue(state: ResearchState) -> str:
    """Determines if we should continue with tool calls or end"""
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    else:
        return "end"


tool_node = ToolNode(tools=tools)

graph = StateGraph(ResearchState)
graph.add_node("research_agent", research_agent)
graph.add_node("tools", tool_node)

graph.set_entry_point("research_agent")
graph.add_conditional_edges(
    "research_agent",
    should_continue,
    {
        "tools": "tools",
        "end": END
    }
)

graph.add_edge("tools", "research_agent")
agent = graph.compile()

while True:
    user_input = input(
        "Please describe what your business does: ")

    if user_input.lower().strip() in ['exit', 'quit']:
        print("\nThanks for using the B2B Lead Generation Agent!")
        break

    print(f"\n🔍 Researching leads for: {user_input}")
    print("⏳ This may take some time..")

    messages = [HumanMessage(content=user_input)]

    try:
        result = agent.invoke({"messages": messages})

        if result and "messages" in result:
            print("\n" + "=" * 80)
            print("📋 BUSINESS DEVELOPMENT RESEARCH REPORT")
            print("=" * 80)

            # Get the final report (last message)
            final_message = result["messages"][-1]
            print(final_message.content)
        else:
            print("❌ No results generated. Please try a more specific query.")

    except Exception as e:
        print(e)
