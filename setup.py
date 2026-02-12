from setuptools import setup
setup(name="multi-llm-router",version="0.1.0",
    description="Smart routing across LLM providers",
    long_description=open("README.md").read(),long_description_content_type="text/markdown",
    author="Lei Hua",url="https://github.com/leiMizzou/multi-llm-router",
    py_modules=["llm_router"],python_requires=">=3.8",
    entry_points={"console_scripts":["llm-router=llm_router:main"]})
