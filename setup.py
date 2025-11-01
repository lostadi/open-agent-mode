from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="open-agent-mode",
    version="0.1.0",
    author="Open Agent Contributors",
    description="Open-source implementation of ChatGPT's agent mode",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/open-agent-mode",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "openai>=1.0.0",
        "anthropic>=0.7.0",
        "aiohttp>=3.8.0",
        "aiofiles>=23.0.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0",
        "click>=8.0.0",
        "rich>=13.0.0",
        "python-dotenv>=1.0.0",
        "pydantic>=2.0.0",
        "jinja2>=3.0.0",
        "websockets>=11.0.0",
        "beautifulsoup4>=4.12.0",
        "requests>=2.31.0",
        "tiktoken>=0.5.0",
        "PyYAML>=6.0",
        "docker>=6.1.0",
        "aiodocker>=0.21.0",
        "black>=23.0.0",
        "autopep8>=2.0.0",
        "google-generativeai>=0.3.0",
        "groq>=0.4.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "pre-commit>=3.0.0",
        ],
        "ollama": ["ollama>=0.1.0"],
    },
    entry_points={
        "console_scripts": [
            "open-agent=open_agent.cli:main",
        ],
    },
)
