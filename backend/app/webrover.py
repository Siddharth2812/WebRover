from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
#from langchain_anthropic import ChatAnthropic
import os
#import nest_asyncio
from typing import TypedDict, List, Annotated, Optional, Dict, Type
from playwright.async_api import Page
from langchain_core.messages import BaseMessage
from operator import add
import base64
import asyncio
from playwright.async_api import async_playwright
import asyncio
from PIL import Image as PILImage
import io
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
import platform
# from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END
from datetime import datetime
import logging

logger = logging.getLogger(__name__)




#nest_asyncio.apply()


load_dotenv()

def set_env_vars(var):
    value = os.getenv(var)
    if value is not None:
        os.environ[var] = value


vars = ["OPENAI_API_KEY", "LANGCHAIN_API_KEY", "LANGCHAIN_TRACING_V2", "LANGCHAIN_ENDPOINT", "LANGCHAIN_PROJECT"]

for var in vars:
    set_env_vars(var)

llm_openai = ChatOpenAI(model="gpt-4o", temperature=0)
llm_mini = ChatOpenAI(model="gpt-4o-mini", temperature=0)
#llm_anthropic = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=0)
llm_openai_o1 = ChatOpenAI(model="o1-preview", temperature=1)
llm = llm_openai

current_dir = os.path.dirname(os.path.abspath(__file__))

mark_page_path = os.path.join(current_dir, "static", "mark_page.js")



class Bbox(TypedDict):
    x: int
    y: int
    text: str
    type: str
    ariaLabel: str

class Action(TypedDict):
    action: str
    args: str | Bbox

class MasterPlanState(TypedDict):
    plan: List[str]



class AgentState(TypedDict):
    input: str
    page: Page
    image: str
    master_plan: MasterPlanState
    bboxes: List[Bbox]
    actions_taken: Annotated[List[str], add]
    action: Action | str
    last_action: str
    notes: Annotated[List[str], add]
    answer: str
    sensitive_data: Optional[Dict[str, str]]
    use_vision: bool
    ratio_width: int = 1076
    ratio_height: int = 1076



with open(mark_page_path) as f:
    mark_page_script = f.read()

async def is_image_blank(image_bytes: bytes) -> bool:
    """Return True if the screenshot is fully blank (e.g. all white), else False."""
    if not image_bytes:
        return True
    img = PILImage.open(io.BytesIO(image_bytes)).convert("L")
    # If getbbox() returns None, the image is entirely one color
    return img.getbbox() is None

async def capture_screenshot(page: Page, max_retries=3, wait_seconds=2) -> bytes:
    """Take a screenshot, retry if blank (completely white)."""
    screenshot_bytes = b""
    for attempt in range(max_retries):
        # Wait for the page to be fully loaded
        await page.wait_for_load_state("networkidle")
        
        # Take screenshot
        screenshot_bytes = await page.screenshot(path="screenshot.jpg", type="jpeg", quality=60, scale="device")
        
        # Check if it's blank
        if not await is_image_blank(screenshot_bytes):
            return screenshot_bytes
        
        # If blank, wait a bit and retry
        print(f"[capture_screenshot] Screenshot is blank (attempt {attempt+1}/{max_retries}). Retrying...")
        await asyncio.sleep(wait_seconds)
    
    # If we get here, all attempts yielded a blank screenshot
    print("[capture_screenshot] All screenshot attempts were blank.")
    return screenshot_bytes  # Return whatever we got last


async def mark_page(state: AgentState, max_retries=3, wait_seconds=2) -> dict:
    page = state["page"]
    ratio_width = state["ratio_width"]
    ratio_height = state["ratio_height"]
    use_vision = state["use_vision"]

    # Apply custom viewport size
    await page.set_viewport_size({"width": ratio_width, "height": ratio_height})

    if not use_vision:
        return {
            "image": "",
            "bboxes": []
        }

    bboxes = []
    
    for attempt in range (3):
        
        try: 
            
            await page.wait_for_load_state("domcontentloaded")
            await page.evaluate(mark_page_script)
            bboxes = await page.evaluate("markPage()")
            
            break
        except Exception as e:
            print(f"[mark_page] Attempt {attempt+1}/3 failed to mark page: {e}")
            await asyncio.sleep(3)
    # Get screenshot as bytes
    await page.wait_for_load_state("networkidle")
    screenshot_bytes = await capture_screenshot(page, max_retries=3)
    
    # Process screenshot if we have any bytes
    if screenshot_bytes:
        img = PILImage.open(io.BytesIO(screenshot_bytes))
        # Convert to grayscale
        img = img.convert('L')
        # Resize
        max_size = (600, 600)
        img.thumbnail(max_size, PILImage.Resampling.LANCZOS)
        # Quantize and convert back to grayscale
        img = img.quantize(colors=256).convert('RGB')
        
        # Compress
        buffer = io.BytesIO()
        img.save(
            buffer,
            format='JPEG',
            quality=50,      # Low quality -> smaller size
            optimize=True,
            progressive=True
        )
        compressed_bytes = buffer.getvalue()
    else:
        # If screenshot is empty or never taken, handle gracefully
        print("[mark_page] Using empty screenshot due to failure or blank screenshot.")
        compressed_bytes = b""

    await page.wait_for_load_state("networkidle")
    try:
        await page.evaluate("unmarkPage()")
    except Exception as e:
        print(f"[mark_page] Could not unmark page: {e}")

    # Reset viewport before returning
    await page.set_viewport_size({"width": 1076, "height": 1076})

    return {
        "image": base64.b64encode(compressed_bytes).decode("utf-8"),
        "bboxes": bboxes
    }


async def setup_browser_2(go_to_page: str):
    playwright = await async_playwright().start()
    
    # Add browser arguments to appear more human-like
    browser_args = [
        '--disable-dev-shm-usage',
        '--disable-blink-features=AutomationControlled',  # Hide automation
        '--no-sandbox',
        '--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',  # Use a common user agent
    ]
    
    # Add browser context options
    context_options = {
        #"viewport": {"width": 1076, "height": 1076},  # Standard desktop resolution
        "viewport": {"width": 1280, "height": 720},
        "user_agent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
        "permissions": ['geolocation'],
        "geolocation": {"latitude": 37.7749, "longitude": -122.4194},  # Set a fixed location
        "locale": 'en-US',
        "timezone_id": 'America/Los_Angeles',
    }
    
    browser = await playwright.chromium.launch(
        headless=False,
        args=browser_args
    )
    
    # Create context with the specified options
    context = await browser.new_context(**context_options)
    
    # Enable JavaScript and cookies
    await context.add_init_script("""
        Object.defineProperty(navigator, 'webdriver', {
            get: () => undefined
        });
    """)
    
    page = await context.new_page()
    
    try:
        await page.goto(go_to_page, timeout=80000, wait_until="domcontentloaded")
    except Exception as e:
        print(f"Error loading page: {e}")
        # Fallback to Google if the original page fails to load
        await page.goto("https://www.google.com", timeout=60000, wait_until="domcontentloaded")
    
    return playwright, browser, page



class MessageMetadata(TypedDict):
    input_tokens: int
    timestamp: str = datetime.now().isoformat()

class MessageHistory:
    def __init__(self):
        self.messages: List[BaseMessage] = []
        self.total_tokens: int = 0
    
    def add_message(self, message: BaseMessage, metadata: MessageMetadata, position: Optional[int] = None) -> None:
        if position is not None:
            self.messages.insert(position, message)
        else:
            self.messages.append(message)
        self.total_tokens += metadata.input_tokens
    
    def remove_message(self, index: int = -1) -> None:
        if self.messages:
            removed_message = self.messages.pop(index)
            self.total_tokens -= getattr(removed_message, 'input_tokens', 0)


def _filter_sensitive_data(text: str, sensitive_data: Dict[str, str]) -> str:
    """Filter out sensitive data from the message"""
    if not sensitive_data:
        return text
    filtered = text
    for key, val in sensitive_data.items():
        filtered = filtered.replace(val, f'<secret>{key}</secret>')
    return filtered

# Update MAX_PROMPT_TOKENS to match MessageManager's max_input_tokens
MAX_PROMPT_TOKENS = 128000

async def master_plan_node(state: AgentState):
    try:
        # Initialize ratio_width and ratio_height if not present
        if "ratio_width" not in state:
            state["ratio_width"] = 1076
        if "ratio_height" not in state:
            state["ratio_height"] = 1076
        if "use_vision" not in state:
            state["use_vision"] = True

        screen_shot = await mark_page(state)
        
        system_message = """
        You are an expert a preparing a step by step plan to complete a task.
        You will be given a task provided by the user. The task might also be a question.
        You will need to prepare a plan to complete the task. In case its a question, you will need to prepare a plan to answer the question.

        You will be also provided the screenshot of the current web page.
        - If the current page is google home page or any other search engine, create a plan that basically searches the keyword and continues to the next step.
        - If the current page is not a some other web page, create a plan to scroll through the page and relevant collect information. 

        For eg if the task is "What is the lastest news on Apple's stock price?", you will need to prepare a plan to answer the question.
        You will need to prepare a plan to complete the task.

        For example, if the task is "What is the latest news on Apple's stock price?", your plan might look like this:
        1. Go to Google
        2. Type "Apple stock price news today" in the search bar and press enter
        3. Click on the link to the reliable financial news source (like Reuters, Bloomberg, or CNBC).
        4. Scan the article for current stock price and recent developments
        5. If you have enough information, prepare a concise summary of the latest news and price movement
        6. If you do not have enough information, go back to the previous page and try a different source and collect more data until you have enough information to answer the question.

        Your plan should be clear, sequential, and focused on achieving the user's goal efficiently. 

        --Notes--
        The browser is already open. First page will always be google, so plan accordingly with a search term.
        For any question, you will need to go to google and search for the question.
        """ 

        human_prompt = """ This is the task that needs to be performed/question that needs to be answered: {input} \n This is the screenshot of the current web page: {screenshot}"""

        input = state["input"]

        human_message = human_prompt.format(input=input, screenshot=screen_shot)

        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=human_message)
        ]

        # Count tokens for all messages
        total_tokens = sum(_count_message_tokens(msg) for msg in messages)
        print(f"Total tokens in master plan: {total_tokens}")

        if total_tokens > MAX_PROMPT_TOKENS:
            print(f"Truncating master plan from {total_tokens} to {MAX_PROMPT_TOKENS} tokens")
            try:
                # Keep system message intact and truncate human message
                system_tokens = _count_message_tokens(messages[0])
                available_tokens = MAX_PROMPT_TOKENS - system_tokens
                
                if available_tokens > 0:
                    # Truncate human message content
                    human_content = human_message[:int(len(human_message) * (available_tokens / (total_tokens - system_tokens)))]
                    messages = [
                        SystemMessage(content=system_message),
                        HumanMessage(content=human_content)
                    ]
                else:
                    print("Warning: System message alone exceeds token limit")
                    # If we must truncate system message
                    truncated_system = system_message[:int(len(system_message) * (MAX_PROMPT_TOKENS / total_tokens))]
                    messages = [SystemMessage(content=truncated_system)]
            except Exception as e:
                print(f"Error during master plan truncation: {str(e)}")
                raise

        try:
            structured_llm = llm.with_structured_output(MasterPlanState)
            response = structured_llm.invoke(messages)
        except Exception as e:
            print(f"Error during master plan LLM call: {str(e)}")
            raise
        
        return {"master_plan": [response]}
    except Exception as e:
        print(f"Critical error in master_plan_node: {str(e)}")
        raise

def _count_tokens(text: str, estimated_characters_per_token: int = 3) -> int:
    """Count tokens in a text string using character estimation"""
    return len(text) // estimated_characters_per_token

def _count_message_tokens(message: BaseMessage, img_tokens: int = 800, estimated_characters_per_token: int = 3) -> int:
    """Count tokens in a message including both text and images"""
    tokens = 0
    if isinstance(message.content, list):
        for item in message.content:
            if 'image_url' in item:
                tokens += img_tokens
            elif isinstance(item, dict) and 'text' in item:
                tokens += _count_tokens(item['text'], estimated_characters_per_token)
    else:
        msg = message.content
        if hasattr(message, 'tool_calls'):
            msg += str(message.tool_calls)  # type: ignore
        tokens += _count_tokens(msg, estimated_characters_per_token)
    return tokens

async def llm_call_node(state: AgentState):   
    try:
        template = """Imagine you are a robot browsing the web, just like humans. Now you need to complete a task. In each iteration,
        you will receive an Observation that includes a screenshot of a webpage and some texts. 
        Carefully analyze the bounding box information and the web page contents to identify the Numerical Label corresponding 
        to the Web Element that requires interaction, then follow
        the guidelines and choose one of the following actions:

        1. Click a Web Element.
        2. Delete existing content in a textbox and then type content.
        3. Scroll up or down.
        4. Wait 
        5. Go back
        7. Return to google to start over.
        8. Respond with the final answer

        Correspondingly, Action should STRICTLY follow the format:

        - Click [Numerical_Label] 
        - Type [Numerical_Label]; [Content] 
        - Scroll [Numerical_Label or WINDOW]; [up or down] 
        - Wait 
        - GoBack
        - Google
        - Respond 

        Key Guidelines You MUST follow:

        * Action guidelines *
        1) Execute only one action per iteration.
        2) Always click close on the popups.
        3) When clicking or typing, ensure to select the correct bounding box.
        4) Numeric labels lie in the top-left corner of their corresponding bounding boxes and are colored the same.
        5) Try to scroll down if a pdf or a document is opeened to read the entire document., if you dont find the information you need, go back to the previous page and try a different source and collect more data until you have enough information to answer the question.

        * Web Browsing Guidelines *
        1) Don't interact with useless web elements like Login, Sign-in, donation that appear in Webpages
        2) Select strategically to minimize time wasted.

        Your reply should strictly follow the format:
        Thought: {{Your brief thoughts (briefly summarize the info that will help ANSWER)}}
        Action: {{One Action format you choose}} (Make sure to enclose the bbox id in [] , for eg  Click [1], Type [5], Scroll [10] or Scroll [WINDOW])
        
        Then the User will provide:
        Observation: {{A labeled bounding boxes and contents given by User}}"
        Actions Taken: {{A list of actions taken so far}} (Could be empty, if it is the first iteration)
        Master Plan: {{A set of steps that you can use as a reference to complete the task}}

        Observation including a screenshot of a webpage with bounding boxes and the text related to it: {{result}}"""



        prompt = ChatPromptTemplate(
            messages=[
                ("system", template),
                ("human", "Input: {input}"),
                ("human", "Actions Taken So far: {actions_taken}"),
                ("human", "Observation: Screenshot: {image}"),
                ("human", "Observation: Bounding Boxes: {bboxes}"),
                
            ],
            input_variables=["image", "bboxes", "input"],
            partial_variables={"actions_taken": []},
            optional_variables=["actions_taken"]
        )
        
        

        actions_taken = state.get("actions_taken", [])
        image = state["image"]
        bboxes = state["bboxes"]
        input = state["input"]
        master_plan = state["master_plan"]
        sensitive_data = state.get("sensitive_data", {})
        
        if sensitive_data:
            try:
                image = _filter_sensitive_data(image, sensitive_data)
                bboxes = _filter_sensitive_data(str(bboxes), sensitive_data)
            except Exception as e:
                print(f"Error filtering sensitive data: {str(e)}")

        try:
            prompt_value = prompt.invoke({
                "actions_taken": actions_taken, 
                "image": image, 
                "bboxes": bboxes, 
                "input": input, 
                "master_plan": master_plan
            })
        except Exception as e:
            print(f"Error creating prompt: {str(e)}")
            raise

        # Convert to messages and count tokens
        try:
            messages = prompt_value.to_messages()
            total_tokens = sum(_count_message_tokens(msg) for msg in messages)
            print(f"Total tokens in prompt: {total_tokens}")
        except Exception as e:
            print(f"Error counting tokens: {str(e)}")
            total_tokens = MAX_PROMPT_TOKENS + 1  # Force truncation as fallback

        if total_tokens > MAX_PROMPT_TOKENS:
            print(f"Truncating prompt from {total_tokens} to {MAX_PROMPT_TOKENS} tokens")
            try:
                # Calculate proportion to keep
                keep_ratio = MAX_PROMPT_TOKENS / total_tokens
                # Truncate the last message (usually the observation)
                last_message = messages[-1]
                if isinstance(last_message.content, str):
                    truncated_content = last_message.content[:int(len(last_message.content) * keep_ratio)]
                    messages[-1] = HumanMessage(content=truncated_content)
                response = llm.invoke(messages)
            except Exception as e:
                print(f"Error during truncation and LLM call: {str(e)}")
                raise
        else:
            try:
                response = llm.invoke(prompt_value)
            except Exception as e:
                print(f"Error during LLM call: {str(e)}")
                raise

        action = response.content
        return {"action": action}
    except Exception as e:
        print(f"Critical error in llm_call_node: {str(e)}")
        raise



async def parse_action_node(state: AgentState):
    action_prefix = "Action: "
    text = state["action"]
    if not text.strip().split("Action:")[-1]:
        return {"action": "retry", "args": f"Could not parse LLM Output: {text}"}
    action_block = text.strip().split("Action: ")[-1]
    thought_block = text.strip().split("Action: ")[0].split("Thought: ")[-1]
    split_output = action_block.split("; ", 1)
    if len(split_output) == 1:
        action, args = split_output[0], None
    else:
        action, args = split_output

    return {"action": Action(action=action, args=args), "notes": [thought_block]}



tools = {
    "Click": "click",
    "Type": "type",
    "Scroll": "scroll",
    "Wait": "wait",
    "GoBack": "go_back",
    "Google": "go_to_search_engine"
}



def tool_router(state: AgentState):
    action = state["action"]["action"]
    action_type = action.split(" ")[0]
    if action_type == "retry":
        return "mark_page"
    if action_type == "Respond":
        return "answer_node"
    return tools[action_type]


async def answer_node(state: AgentState):
    system_message_answer = """ You are an assistant who is expert at answering the user input based on the notes.
    You will be given:
    Notes: {notes}
    User Input: {input}

    Breakdown the Answer in to two sections:
    1. Steps: A list of steps were taken to surf the web and provide answer to the user input.
    2. Final Answer: Should only contain the final answer that directly provides answer to the user input.

    Provide the answer in proper markdown format. Use proper markdown formatting for the steps and final answer.
    """

    prompt_answer = ChatPromptTemplate(
        messages=[
            ("system", system_message_answer),
        ],
        input_variables=["notes", "input"],
    )   

    notes = state["notes"]
    input = state["input"]

    prompt_value_answer = prompt_answer.invoke({"notes": notes, "input": input})
    response_answer = llm_mini.invoke(prompt_value_answer)
    answer = response_answer.content
    
    return {"answer": answer}



async def click(state: AgentState):
    page = state["page"]
    action = state["action"]
    bbox_id = int(action["action"].split(" ")[1].split("[")[1].split("]")[0])
    if bbox_id not in [bbox["id"] for bbox in state["bboxes"]]:
        return {"action": "retry", "args": f"Could not find bbox with id {bbox_id}"}
    bbox = state["bboxes"][bbox_id]
    await asyncio.sleep(2)
    await page.mouse.click(bbox["x"], bbox["y"])
    await asyncio.sleep(4)
    return {"last_action": f"Click : clicked on {bbox_id}", "actions_taken": [f"Click : clicked on {bbox_id}"]}



async def scroll_2(state: AgentState):
    page = state["page"]
    action = state["action"]
    scroll_type = action["action"].split(" ")[1].split("[")[1].split("]")[0]
    direction = action["args"]
    
    async def is_pdf_page():
        current_url = page.url
        return (
            current_url.lower().endswith('.pdf') or
            'pdf' in current_url.lower() or
            '/pdf/' in current_url.lower()
        )

    async def try_scroll_methods(is_down: bool):
        keys = ["PageDown", "Space", "ArrowDown", "j"] if is_down else ["PageUp", "ArrowUp", "k"]
        # Reduced to just one key press for more controlled scrolling
        for key in keys:
            try:
                await page.keyboard.press(key)
                await page.wait_for_timeout(100)
                break  # Exit after first successful key press
            except Exception as e:
                print(f"Failed with key {key}: {str(e)}")
                continue

    if scroll_type == "WINDOW":
        is_pdf = await is_pdf_page()
      
        
        if is_pdf:
            try:
                # Wait for PDF to load
                await page.wait_for_load_state('networkidle')
                await page.wait_for_timeout(1000)

                # Try to click on the PDF to ensure focus
                try:
                    await page.mouse.click(300, 300)  # Click somewhere in the middle of the page
                except Exception:
                    pass

                # Single scroll attempt
                await try_scroll_methods(direction.lower() == "down")
                await page.wait_for_timeout(500)
                
                return {
                    "last_action": f"Scroll : scrolled {direction} on PDF document",
                    "actions_taken": [f"Scroll : scrolled {direction} on PDF document"]
                }
                
            except Exception as e:
                print(f"PDF scrolling error: {str(e)}")
                return {
                    "action": "retry",
                    "args": f"Error scrolling PDF: {str(e)}",
                    "actions_taken": [f"Error scrolling PDF: {str(e)}"]
                }
        else:
            # Regular webpage scrolling with exactly 500px
            scroll_amount = 500  # Changed from 800 to 500
            scroll_direction = -scroll_amount if direction.lower() == "up" else scroll_amount
            try:
                await page.evaluate(f"""
                    window.scrollBy({{
                        top: {scroll_direction},
                        left: 0,
                        behavior: 'smooth'
                    }});
                """)
            except Exception:
                await page.evaluate(f"window.scrollBy(0, {scroll_direction})")
        
        await page.wait_for_timeout(500)
        
        return {
            "last_action": f"Scroll : scrolled {direction}",
            "actions_taken": [f"Scroll : scrolled {direction}"]
        }
    
    else:
        # Element-specific scrolling
        try:
            bbox_id = int(action["action"].split("[")[1].split("]")[0])
            if bbox_id not in [bbox["id"] for bbox in state["bboxes"]]:
                return {
                    "action": "retry",
                    "args": f"Could not find bbox with id {bbox_id}",
                    "actions_taken": [f"Could not find bbox with id {bbox_id}"]
                }
                
            bbox = state["bboxes"][bbox_id]
            scroll_amount = 200
            scroll_direction = -scroll_amount if direction.lower() == "up" else scroll_amount
            
            await page.mouse.move(bbox["x"], bbox["y"])
            await page.mouse.wheel(0, scroll_direction)
            
            return {
                "last_action": f"Scroll : scrolled {direction} at element {bbox_id}",
                "actions_taken": [f"Scroll : scrolled {direction} at element {bbox_id}"]
            }
            
        except Exception as e:
            return {
                "action": "retry",
                "args": f"Error scrolling element: {str(e)}",
                "actions_taken": [f"Error scrolling element: {str(e)}"]
            }
        

async def type(state: AgentState):
    page = state["page"]
    action = state["action"]
    bbox_id = int(action["action"].split("[")[1].split("]")[0])
    if bbox_id not in [bbox["id"] for bbox in state["bboxes"]]:
        return {"action": "retry", "args": f"Could not find bbox with id {bbox_id}"}
    bbox = state["bboxes"][bbox_id]
    await page.mouse.click(bbox["x"], bbox["y"])
    select_all = "Meta+A" if platform.system() == "Darwin" else "Control+A"
    await page.keyboard.press(select_all)
    await asyncio.sleep(2)
    select_all = "Meta+A" if platform.system() == "Darwin" else "Control+A"
    await page.keyboard.press(select_all)
    await asyncio.sleep(1)
    await page.keyboard.press("Backspace")
    await asyncio.sleep(1)
    await page.keyboard.type(action["args"])
    await asyncio.sleep(3)
    await page.keyboard.press("Enter")
    await asyncio.sleep(3)
    return {"last_action": f"Type : typed {action['args']} into {bbox_id}", "actions_taken": [f"Type : typed {action['args']} into {bbox_id}"]}



async def wait(state: AgentState):
    await asyncio.sleep(3)
    return {"last_action": "Wait : waited for 3 seconds", "actions_taken": ["Wait : waited for 3 seconds"]}


async def go_back(state: AgentState):
    page = state["page"]
    await page.go_back()
    return {"last_action": f"Go Back : Navigated back to page {page.url}", "actions_taken": [f"Go Back : Navigated back to page {page.url}"]}



async def go_to_search_engine(state: AgentState):
    page = state["page"]
    await page.goto("https://www.google.com")
    return {"last_action": "Go to Search Engine : Navigated to Google", "actions_taken": ["Go to Search Engine : Navigated to Google"]}


async def control_image_processing(state: AgentState):
    action = state["action"]["action"]
    page = state["page"]
    use_vision = state["use_vision"]
    
    # Disable images for text-heavy operations
    if "extract" in action.lower() or "text" in action.lower():
        return {"use_vision": False}
    else:
        return {"use_vision": True}


builder = StateGraph(AgentState)

builder.add_node("control_image_processing", control_image_processing)
builder.add_node("mark_page", mark_page)
builder.add_node("llm_call", llm_call_node)
builder.add_node("parse_action", parse_action_node)
builder.add_node("master_plan_node", master_plan_node)
builder.add_node("click", click)
builder.add_node("type", type)
builder.add_node("scroll", scroll_2)
builder.add_node("wait", wait)
builder.add_node("go_back", go_back)
builder.add_node("go_to_search_engine", go_to_search_engine)
builder.add_node("answer_node", answer_node)

builder.add_edge(START, "master_plan_node")
builder.add_edge("master_plan_node", "control_image_processing")
builder.add_edge("control_image_processing", "mark_page")
builder.add_edge("mark_page", "llm_call")
builder.add_edge("llm_call", "parse_action")
builder.add_conditional_edges("parse_action", tool_router, ["mark_page",  "click", "type", "scroll", "wait", "go_back", "go_to_search_engine", "answer_node"])
builder.add_edge("click", "mark_page")
builder.add_edge("type", "mark_page")
builder.add_edge("scroll", "mark_page")
builder.add_edge("wait", "mark_page")
builder.add_edge("go_back", "mark_page")
builder.add_edge("go_to_search_engine", "mark_page")
builder.add_edge("answer_node", END)

main_agent_graph = builder.compile()

