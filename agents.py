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

# ==================== 5 AGENTS ====================

cleaner = Agent(
    role="Data Cleaning Specialist",
    goal="Clean and organize raw user comments into usable format",
    backstory="You are meticulous and remove duplicates, spam, and very short irrelevant comments.",
    llm=llm,
    verbose=True
)

analyzer = Agent(
    role="Feedback Analyst",
    goal="Identify sentiment and extract main themes from user comments",
    backstory="You are excellent at finding patterns in customer feedback. You group comments into clear themes.",
    llm=llm,
    verbose=True
)

summarizer = Agent(
    role="Professional Summarizer",
    goal="Create balanced and concise summaries of the feedback",
    backstory="You write in a professional, positive, and constructive tone.",
    llm=llm,
    verbose=True
)

recommender = Agent(
    role="Product Improvement Expert",
    goal="Turn insights into specific, actionable recommendations for Product v2",
    backstory="You give practical, realistic, and prioritized suggestions that help improve the product.",
    llm=llm,
    verbose=True
)

reporter = Agent(
    role="Report Writer",
    goal="Generate a clean, professional final report",
    backstory="You create well-structured, positive, and easy-to-read reports.",
    llm=llm,
    verbose=True
)

# ==================== TASKS & CREW ====================

def create_crew(user_comments: str):
    task1 = Task(
        description=f"Clean these raw user comments and remove noise:\n\n{user_comments}",
        agent=cleaner,
        expected_output="Cleaned list of meaningful feedback"
    )

    task2 = Task(
        description="Analyze sentiment and extract the top themes with approximate frequency.",
        agent=analyzer,
        expected_output="List of main themes with sentiment and counts"
    )

    task3 = Task(
        description="Write a balanced, professional summary of the feedback.",
        agent=summarizer,
        expected_output="Concise summary with key insights"
    )

    task4 = Task(
        description="Provide 6-8 specific, prioritized, and actionable recommendations for Product v2.",
        agent=recommender,
        expected_output="Numbered list of clear recommendations"
    )

    task5 = Task(
        description="Combine everything into a professional final report in markdown format.",
        agent=reporter,
        expected_output="Well-formatted markdown report"
    )

    crew = Crew(
        agents=[cleaner, analyzer, summarizer, recommender, reporter],
        tasks=[task1, task2, task3, task4, task5],
        verbose=True
    )

    return crew
