from src.config import settings
from src.paths import CONSULTORIO_PDF_PATH, WORKFLOW_JPG_PATH
from src.chromadb_manager import ChromaDBManager

from pydantic import BaseModel
from typing import Annotated, Literal
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AnyMessage, AIMessage, HumanMessage
from langchain_core.tools import tool
from fastapi import FastAPI
from fastapi.responses import JSONResponse

api_key = settings['openai']
model = 'gpt-4o-mini'
llm = ChatOpenAI(api_key=api_key, model=model)

class AgentState(BaseModel):
    user_message: str = ""
    query: str = ""
    context: str = ""
    language: Literal["es", "en"] = "es"
    messages: Annotated[list[AnyMessage], add_messages] = []
    question_type: Literal["appointment", "question"] = "question"

class LanguageOutput(BaseModel):
    language: Literal["es", "en"]

class QuestionType(BaseModel):
    question_type: Literal["appointment", "question"]

def get_language_node(state: AgentState) -> AgentState:
    llm_parsed = llm.with_structured_output(LanguageOutput)
    response: LanguageOutput = llm_parsed.invoke(f"Detect the language of the following text, answer 'es' or 'en': '{state.user_message}'")
    state.language = response.language
    print(f"Detected language: {state.language}")
    return state

def get_question_type_node(state: AgentState) -> Literal["appointment_node", "query_node"]:
    user_message = state.user_message[-1].content
    llm_parsed = llm.with_structured_output(QuestionType)
    response: QuestionType = llm_parsed.invoke(f"Is the following message a question or an appointment? User query: '{user_message}'")
    state.question_type = response.question_type
    print(f"Detected question type: {state.question_type}")
    if state.question_type == "appointment":
        return "appointment_node"
    else:
        return "query_node"

@tool  
def create_appointment(name: str, date: str, time: str, description: str):
    """
    Schedule an appointment for a patient by name, date, and time.
    """
    return f"Appointment scheduled for {name} on {date} at {time} with description: {description}"

    
def appointment_node(state: AgentState) -> AgentState:
    tools = [create_appointment]
    llm_with_tools = llm.bind_tools(tools)
    response = llm_with_tools.invoke(state.messages)
    if response.tool_calls:
        function_result = None
        for call in response.tool_calls:
            if call["name"] == "create_appointment":
                args = call["args"]
                function_result  = create_appointment.invoke(args)
        state.messages = [AIMessage(content=function_result)]
    else:
        state.messages = [AIMessage(content=response.content)]
    return state

def query_node(state: AgentState) -> AgentState:
    history = state.messages[:-1]
    user_message = state.user_message[-1].content
    response = llm.invoke(f"""
    You are an agent who must generate a query to perform a vector search.
    You should not add words that could cause unwanted vectors to be searched.
    You must use the most recent message to generate the query, and you can also
    use the history to gain greater context about the user's question.
    
    Conversation history:
    {history}

    New message from user:
    {user_message}

    Query:
    """ )
    state.query = response.content
    print(f"Generated query: {state.query}")
    return state

def rag_node(state: AgentState) -> AgentState:
    chromadb_manager = ChromaDBManager()
    response = chromadb_manager.query(state.query, metadata={'filename': str(CONSULTORIO_PDF_PATH)}, k=2)
    state.context = "\n\n".join([doc.page_content for doc in response])
    return state

def response_node(state: AgentState) -> AgentState:
    history = state.messages[:-1]
    user_message = state.messages[-1]
    llm = ChatOpenAI(model="gpt-4o-mini")
    llm_response = llm.invoke(
    f"""
    You are an assistant who answers user questions related to a medical office, using contextual information and history to respond.

    Context:
    {state.context}

    Conversation history:
    {history}

    user query:
    {user_message}
    """
    )
    new_message = AIMessage(content=llm_response.content)
    state.messages = [new_message]
    return state

## FastAPI setup
app = FastAPI()

graph = StateGraph(AgentState)
graph.add_node(get_language_node)
graph.add_node(appointment_node)
graph.add_node(query_node)
graph.add_node(rag_node)
graph.add_node(response_node)

graph.add_edge(START, "get_language_node")
graph.add_conditional_edges("get_language_node", get_question_type_node)
graph.add_edge("appointment_node", END)
graph.add_edge("query_node", "rag_node")
graph.add_edge("rag_node", "response_node")
graph.add_edge("response_node", END)

initial_state = AgentState()
checkpointer = MemorySaver()
compiled_graph = graph.compile(checkpointer=checkpointer)

# _ = compiled_graph.get_graph().draw_mermaid_png(output_file_path=str(WORKFLOW_JPG_PATH))

class AgentInput(BaseModel):
    question: str

@app.post("/run")
def run(agent_input: AgentInput):
    # question = "Jason, tomorrow at 10:00 am, with a pediatrist, please."
    user_message = HumanMessage(content=agent_input.question)
    initial_state.user_message = [user_message]
    initial_state.messages = [user_message]

    agent_response = compiled_graph.invoke(
        initial_state,
        config={"configurable": {"thread_id": 1}})
    print(agent_response["messages"][-1].content)
    return JSONResponse(content=agent_response["messages"][-1].content)