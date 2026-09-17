from typing import List, Dict, Tuple
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI

def format_docs(docs: List[Document]) -> str:
    """Format documents into a single string for the prompt context."""
    formatted = []
    for i, doc in enumerate(docs):
        # We also pass the citation metadata to the model so it understands the source
        source = doc.metadata.get("source", "Unknown Source")
        page = doc.metadata.get("page", "Unknown Page")
        formatted.append(f"--- Document {i+1} [Source: {source}, Page: {page}] ---\n{doc.page_content}")
    return "\n\n".join(formatted)

def get_rag_chain(api_key: str, model_name: str = "gemini-3.1-flash-lite-preview", retriever=None):
    """
    Constructs the conversational RAG chain using LCEL (LangChain Expression Language).
    Requires a retriever to fetch context before calling the LLM.
    """
    
    # Initialize the LLM
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0.0 # Lowest temperature for groundedness
    )
    
    # Custom rules for explanation and external knowledge
    system_prompt = (
        "You are 'PaperMindAI', an AI assistant designed to answer questions based on the provided research context.\n"
        "Instructions:\n"
        "1. Primarily use the provided context to answer the user's questions.\n"
        "2. If the question is out of the pdf or the context does not contain the answer, you MUST respond EXACTLY with 'I don't know'. Do not provide any answer for that question.\n"
        "3. If the context contains mangled text, code blocks missing spaces (e.g. 'defget_or_create'), or broken formatting from PDF extraction, you MUST naturally reconstruct the proper spaces, indentation, and formatting in your final answer so it is accurate and readable.\n"
        "\nContext:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])
    
    # Construct chain
    # We will pass the pre-retrieved context so we can also display the snippets in the UI.
    # Therefore, the chain will just text text formatting.
    
    rag_chain = prompt | llm | StrOutputParser()
    
    return rag_chain

def generate_answer(query: str, retriever, api_key: str, model_name: str = "gemini-3.1-flash-lite-preview", chat_history: List[Dict] = None) -> Tuple[str, List[Document]]:
    """
    Orchestrator function that retrieves chunks and generates the final answer.
    Takes optional chat_history to enable conversational memory and context-aware retrieval.
    Returns the answer string and the list of Document chunks used as context.
    """
    if not api_key:
        raise ValueError("Google Gemini API Key is missing.")
        
    search_query = query
    lc_messages = []
    
    if chat_history and len(chat_history) > 1:
        for msg in chat_history[:-1]:
            if msg["role"] == "user":
                lc_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                lc_messages.append(AIMessage(content=msg["content"]))
                
    # 2. Retrieve Context using Standalone Query
    retrieved_docs = retriever.invoke(search_query)
    
    # 3. Format Context
    context_str = format_docs(retrieved_docs)
    
    # 4. Generate Answer with Full Conversation History
    chain = get_rag_chain(api_key, model_name)
    answer = chain.invoke({
        "context": context_str,
        "chat_history": lc_messages,
        "question": query
    })
    
    return answer, retrieved_docs