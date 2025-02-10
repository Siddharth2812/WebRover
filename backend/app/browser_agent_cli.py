import asyncio
from pathlib import Path
import json
from webrover import setup_browser_2, main_agent_graph

async def setup_browser_with_cookies(url: str):
    try:
        # Setup new browser session with a context
        playwright, browser, page = await setup_browser_2(url)
        context = page.context
        
        # Load and set cookies from linkedin_cookies.json if it exists
        cookies_path = Path(__file__).parent / "cookies.json"
        if cookies_path.exists():
            try:
                with open(cookies_path, 'r') as f:
                    cookies = json.load(f)
                # Add cookies to the context
                await context.add_cookies(cookies)
                # Verify cookies were set
                current_cookies = await context.cookies()
                print(f"Loaded {len(current_cookies)} cookies")
                print("Current cookies:", current_cookies)
            except Exception as e:
                print(f"Error loading cookies: {str(e)}")
                print(f"Cookies path: {cookies_path}")
        else:
            print(f"Cookies file not found at: {cookies_path}")
        
        return playwright, browser, context, page
    except Exception as e:
        print(f"Failed to setup browser: {str(e)}")
        return None, None, None, None

async def cleanup_browser(playwright, browser, context, page):
    try:
        if page:
            await page.close()
        if context:
            await context.close()
        if browser:
            await browser.close()
        if playwright:
            await playwright.stop()
    except Exception as e:
        print(f"Error during cleanup: {str(e)}")

async def run_browser_agent():
    try:
        # Get user input
        query = input("Enter your query: ")
        url = "https://www.google.com"  # Starting with Google
        
        # Setup browser
        playwright, browser, context, page = await setup_browser_with_cookies(url)
        if not page:
            print("Failed to setup browser")
            return
        
        # Prepare initial state
        initial_state = {
            "input": query,
            "page": page,
            "image": "",
            "master_plan": None,
            "bboxes": [],
            "actions_taken": [],
            "action": None,
            "last_action": "",
            "notes": [],
            "answer": ""
        }
        
        try:
            # Run the agent graph
            async for event in main_agent_graph.astream(
                initial_state,
                {"recursion_limit": 400}
            ):
                if isinstance(event, dict):
                    if "parse_action" in event:
                        action = event["parse_action"]["action"]
                        thought = event["parse_action"]["notes"][-1]
                        print("\nThought:", thought)
                        print("Action:", action)
                    elif "answer" in event:
                        print("\nFinal Answer:")
                        print(event["answer"])
                        break
        except Exception as e:
            print(f"Error during agent execution: {str(e)}")
        
        # Cleanup
        await cleanup_browser(playwright, browser, context, page)
        
    except Exception as e:
        print(f"Error: {str(e)}")

def main():
    # Run the async function
    asyncio.run(run_browser_agent())

if __name__ == "__main__":
    main() 