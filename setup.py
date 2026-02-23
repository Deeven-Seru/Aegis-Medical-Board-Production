from setuptools import setup, find_packages

setup(
    name="aegis-mdt",
    version="2.0.0",
    description="Autonomous Multi-Agent Diagnostic Board powered by HAI-DEF Models",
    author="Deeven Seru & Harvey",
    author_email="deevenseru11@gmail.com",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "uvicorn",
        "pydantic",
        "requests",
        "asyncio",
        "loguru",
        "prometheus-client",
        "transformers",
        "torch",
        "accelerate",
        "bitsandbytes",
        "outlines"
    ],
    entry_points={
        "console_scripts": [
            "aegis=aegis.cli:main",
        ]
    }
)
