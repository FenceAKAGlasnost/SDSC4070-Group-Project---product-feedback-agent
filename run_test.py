from agents import create_crew

# Sample test comments
sample_comments = """
The app is beautiful but it crashes a lot.
Love the new design!
Battery drains too fast.
UI is confusing for new users.
Great product, but too expensive.
Feature request: dark mode please.
It keeps logging me out randomly.
"""

crew = create_crew(sample_comments)
result = crew.kickoff()

print(result)