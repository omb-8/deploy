import os
import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from typing import Sequence
from typing_extensions import Annotated, TypedDict
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from dotenv import load_dotenv

# Configure API key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = api_key

# Initialize the Google Generative AI model
gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
model = ChatGoogleGenerativeAI(model="gemini-1.0-pro", convert_system_message_to_human=True)

# Load the document
document_loader = PyPDFLoader("/Users/maryam/Documents/UWF/our/chatbot/22_studenthandbook-22-23_f2.pdf")
doc = document_loader.load()

# Split documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(doc)

# Create a vector store and retriever
vectorstore = Chroma.from_documents(documents=splits, embedding=gemini_embeddings)
retriever = vectorstore.as_retriever()

# Set up prompts
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(model, retriever, contextualize_q_prompt)

# Create the question-answer chain
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know."
    "\n\n"
    "{context}"
)
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(model, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# State management with LangGraph
class State(TypedDict):
    input: str
    chat_history: Annotated[Sequence[BaseMessage], "add_messages"]
    context: str
    answer: str

def call_model(state: State):
    response = rag_chain.invoke(state)
    return {
        "chat_history": [
            HumanMessage(state["input"]),
            AIMessage(response["answer"]),
        ],
        "context": response["context"],
        "answer": response["answer"],
    }

workflow = StateGraph(state_schema=State)
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# Streamlit User Interface
st.title("Custom Question-Answering Chatbot")
st.write("Ask questions based on the loaded document.")

# Maintain chat history using Streamlit session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User input section
user_input = st.text_input("Enter your question here:")

# Submit button
if st.button("Submit"):
    if user_input:
        # Prepare state and invoke the model
        state = {"input": user_input, "chat_history": st.session_state.chat_history, "context": "", "answer": ""}
        config = {"configurable": {"thread_id": "246"}}
        
        result = app.invoke(state, config=config)
        
        # Display response and update chat history
        st.session_state.chat_history.append(HumanMessage(user_input))
        st.session_state.chat_history.append(AIMessage(result["answer"]))
        
        st.write("Chatbot:", result["answer"])
    else:
        st.write("Please enter a question.")

