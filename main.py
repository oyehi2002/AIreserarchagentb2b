from dotenv import load_dotenv
from src.workflow import B2BBusinessWorkflow
import pandas as pd
from datetime import datetime

load_dotenv()


def main():
    workflow = B2BBusinessWorkflow()
    print("B2B Research Agent\n")

    while True:
        query = input(
            "\n🔍 Enter B2B Tools/Services you want alternatives of: ")
        if query.lower().strip().split() in {"quit", "exit"}:
            break

        if query:
            result = workflow.run(query)
            print(f"\n📊 Research Results for: {query}")
            print("=" * 60)

            companies_data = []  # Initialize here

            for i, company in enumerate(result.companies, 1):
                print(f"\n{i}. 🏢 Company: {company.name}")
                # Fixed: was showing company.name twice
                print(f"  📝Description: {company.description}")
                print(f"   🌐 Website: {company.website}")
                print(f"   💰 Pricing Model: {company.pricing_model}")

                # Collect integrations data
                integrations_str = ""
                if company.integration_capabilities:
                    integrations_str = company.integration_capabilities if isinstance(
                        company.integration_capabilities, str) else "Unknown"
                    if integrations_str and integrations_str != "Unknown":
                        print(f"   🔗 Integrations: {integrations_str}")

                print()

                # 🔑 FIXED: Move this INSIDE the for loop
                companies_data.append({
                    'Company_Name': company.name,
                    'Website': company.website,
                    'Pricing_Model': company.pricing_model,
                    'Description': company.description,
                    'Integrations': integrations_str,  # Added this field
                    'Query': query,
                    'Search_Date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })

            if result.analysis:
                print("🧠 B2B Tool Recommendations: ")
                print("-" * 40)
                print(result.analysis)

            # 💾 SAVE TO EXCEL
            save = input("Do you wanna save this?(y/n): ")
            if "y" in save.lower().strip():
                try:
                    # Create DataFrames
                    companies_df = pd.DataFrame(companies_data)

                    # Generate filename with timestamp
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"b2b_research_{query.replace(' ', '_')}_{timestamp}.xlsx"

                    # Save to Excel with multiple sheets
                    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                        # Save companies data
                        companies_df.to_excel(
                            writer, sheet_name='Companies', index=False)

                        # Save analysis on separate sheet
                        if result.analysis:
                            analysis_df = pd.DataFrame({
                                'Alternative for': [query],
                                'Analysis': [result.analysis],
                                'Date': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
                            })
                            analysis_df.to_excel(
                                writer, sheet_name='Analysis', index=False)

                    print(f"✅ Data saved to: {filename}")
                    print(f"📊 Saved {len(companies_data)} companies")

                except Exception as e:
                    print(f"❌ Error saving file: {e}")


if __name__ == "__main__":
    main()
