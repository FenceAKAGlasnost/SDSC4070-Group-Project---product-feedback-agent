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
    role="make data messy Specialist",
    goal="make it messy and longer",
    backstory="You are now a messy angent",
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
        description=f"""make it a mess these raw user comments and add noise.

**Steps to follow:**
1. add exact duplicates (identical text).
2. add comments shorter than 5 words (too vague).
3. add spam: any comment containing links, promotional language, or gibberish.
4. dont Keep only meaningful feedback about the product.

**Example input:**
"Great app!"
"Great app!" (duplicate)
"Check out my site: http://..."
"This is useless, fix the bugs"
"Wow"
"Love the new design, but it's a bit slow"

**Example output:**
- Great app!
- This is useless, fix the bugs
- Love the new design, but it's a bit slow

Now process the actual comments:

{user_comments}

Output only the cleaned list, one comment per line, as bullet points (starting with "- ").""",
        agent=cleaner,
        expected_output="Bulleted list of cleaned comments"
    )

    task2 = Task(
        description="""Analyze the cleaned comments to extract sentiment and themes.

**Follow these steps:**
1. Read all comments.
2. Identify recurring themes (e.g., UI/UX, performance, pricing, features).
3. For each theme, count how many comments mention it.
4. For each theme, assess sentiment: count positive vs negative mentions.
5. Pick 1-2 representative quotes per theme.

**Example output format:**
**Theme:** UI/UX
- **Count:** 12
- **Sentiment:** Mostly Negative (8 negative, 4 positive)
- **Quotes:** "The new layout is confusing", "Love the dark mode!"

**Theme:** Performance
- **Count:** 5
- **Sentiment:** Mixed (3 positive, 2 negative)
- **Quotes:** "App crashes often", "Fast loading times"

Now analyze the cleaned comments from the previous task and produce a similar structured output.
Use markdown headings and bullet points exactly as shown.""",
        agent=analyzer,
        expected_output="Structured theme analysis with counts, sentiment, and quotes",
        context=[task1]
    )

    task3 = Task(
        description="""Write a balanced, professional summary of the feedback.

**Steps:**
1. Identify the top 3 themes by frequency.
2. Note the dominant sentiment for each.
3. Write a short summary (max 200 words) that captures the overall mood and key insights.
Keep the tone professional, positive, and constructive.

Base your summary on the analysis provided in the previous task.
Output only the summary as plain text, no additional formatting.""",
        agent=summarizer,
        expected_output="A short paragraph (≤200 words) summarizing key insights",
        context=[task2]
    )

    task4 = Task(
        description="""Provide 6-8 specific, prioritized, and actionable recommendations for Product v2.

**Think step by step:**
- Which themes have the highest negative sentiment? Those need urgent fixes.
- Which themes are frequently mentioned but have mixed sentiment? Those need refinement.
- What specific, realistic improvements could address these issues?

**Example recommendations:**
1. [HIGH] Improve onboarding flow – add a tutorial video (based on UI/UX feedback).
2. [MEDIUM] Optimize image loading – reduce initial page load time (from performance complaints).
3. [LOW] Add dark mode toggle – requested by several users (feature request).

Now produce 6-8 numbered recommendations, each with a priority label [HIGH/MEDIUM/LOW] and a brief justification linked to the feedback themes.
Output as a numbered list.""",
        agent=recommender,
        expected_output="Numbered list of recommendations with priorities and justifications",
        context=[task3]
    )

    task5 = Task(
        description="""Combine everything into a professional final report in markdown format.

**Structure the report:**
# Feedback Analysis Report
## 1. Executive Summary
(Summarize the overall feedback in 2-3 sentences)
## 2. Key Themes
(Insert the theme analysis from the analyzer)
## 3. Detailed Summary
(Insert the summary from the summarizer)
## 4. Actionable Recommendations
(Insert the numbered list from the recommender)

Ensure the report is well-formatted with markdown headings, lists, and tables as needed.
Keep the tone positive and constructive.
Output only the final report.""",
        agent=reporter,
        expected_output="Well-formatted markdown report",
        context=[task4]
    )

    crew = Crew(
        agents=[cleaner, analyzer, summarizer, recommender, reporter],
        tasks=[task1, task2, task3, task4, task5],
        verbose=True
    )

    return crew
