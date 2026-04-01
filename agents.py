import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from langchain_openrouter import ChatOpenRouter
import litellm

load_dotenv()

# === Correct LLM Setup for OpenRouter with CrewAI ===
llm = ChatOpenRouter(
    model="openrouter/meta-llama/llama-3.3-70b-instruct",   # Important: prefix with "openrouter/"
    temperature=0.6,
    max_tokens=1500,
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

# ==================== 5 AGENTS (now with test modifications) ====================

cleaner = Agent(
    role="Data Cleaning Specialist (TEST MODE)",
    goal="Add 'TEST: ' to every cleaned comment and return the list",
    backstory="You are in test mode. You always add 'TEST: ' at the start of each cleaned comment.",
    llm=llm,
    verbose=True
)

analyzer = Agent(
    role="Feedback Analyst (TEST MODE)",
    goal="Always output a fake theme named 'TEST THEME' with count 999 and sentiment 'positive'",
    backstory="You are in test mode. You ignore the actual comments and output a single fake theme.",
    llm=llm,
    verbose=True
)

summarizer = Agent(
    role="Professional Summarizer (TEST MODE)",
    goal="Always output the exact string: 'TEST: This is a test summary from the summarizer agent.'",
    backstory="You are in test mode. You ignore the analysis and just output the test sentence.",
    llm=llm,
    verbose=True
)

recommender = Agent(
    role="Product Improvement Expert (TEST MODE)",
    goal="Always output a single test recommendation: '1. [HIGH] TEST recommendation from the recommender agent.'",
    backstory="You are in test mode. You ignore insights and output only the test recommendation.",
    llm=llm,
    verbose=True
)

reporter = Agent(
    role="Report Writer (TEST MODE)",
    goal="Always output a test report that clearly indicates the multi‑agent pipeline was used.",
    backstory="You are in test mode. You combine the previous test outputs into a simple test report.",
    llm=llm,
    verbose=True
)

# ==================== TASKS & CREW (modified to produce test outputs) ====================

def create_crew(user_comments: str):
    task1 = Task(
        description=f"""You are in test mode. Ignore the actual user comments.
        Instead, output a bulleted list with exactly one item: "- TEST: Cleaner agent ran successfully."

        Comments (ignored): {user_comments}
        """,
        agent=cleaner,
        expected_output="Bulleted list with test marker"
    )

    task2 = Task(
        description="""You are in test mode. Ignore the cleaned comments from the previous task.
        Output exactly this structure:
        **Theme:** TEST THEME
        - **Count:** 999
        - **Sentiment:** Positive
        - **Quotes:** "This is a test quote"
        """,
        agent=analyzer,
        expected_output="Fake theme analysis",
        context=[task1]
    )

    task3 = Task(
        description="""You are in test mode. Ignore the analysis.
        Output exactly: "TEST: This is a test summary from the summarizer agent."
        """,
        agent=summarizer,
        expected_output="Test summary sentence",
        context=[task2]
    )

    task4 = Task(
        description="""You are in test mode. Ignore the summary.
        Output exactly: "1. [HIGH] TEST recommendation from the recommender agent."
        """,
        agent=recommender,
        expected_output="Test recommendation",
        context=[task3]
    )

    task5 = Task(
        description="""You are in test mode. Combine the outputs from previous agents into a simple report.
        Use this format:
        # TEST REPORT
        ## Cleaner Output
        (copy the output from task1)
        ## Analyzer Output
        (copy the output from task2)
        ## Summarizer Output
        (copy the output from task3)
        ## Recommender Output
        (copy the output from task4)

        Output only the report, no extra text.
        """,
        agent=reporter,
        expected_output="Test report",
        context=[task4]
    )

    crew = Crew(
        agents=[cleaner, analyzer, summarizer, recommender, reporter],
        tasks=[task1, task2, task3, task4, task5],
        verbose=True
    )

    return crew
