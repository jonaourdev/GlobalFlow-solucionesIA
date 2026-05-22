"""Compatibilidad de imports para LangChain 1.x."""

try:
    from langchain_classic.prompts import ChatPromptTemplate
except ImportError:  
    from langchain_core.prompts import ChatPromptTemplate

try:
    from langchain_classic.tools import tool
except ImportError:  
    from langchain_core.tools import tool

try:
    from langchain_classic.memory import ConversationBufferWindowMemory
except ImportError:
    ConversationBufferWindowMemory = None
