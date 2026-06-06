"""Entry point for the Autonomous Wikipedia Researcher.

Thin wrapper that delegates to the package CLI so the agent can be launched
with `python main.py` or `python -m wiki_researcher`.
"""

from wiki_researcher.cli import main

if __name__ == "__main__":
    main()
