import streamlit as st
from sqlalchemy.orm import Session
from database.schema import Users, Sessions, VectorDB
from database.setup import DB_setup
from typing import TypedDict
from langgraph.graph import StateGraph, END, START

import PyPDF2
import hashlib
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv(".env")

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def embed(text: str):
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )
    return response.embeddings[0].values

@st.cache_resource
def get_db_setup():
    db = DB_setup()
    # db.drop_tables()
    db.create_tables()
    return db

db = get_db_setup()

class RAGState(TypedDict):
    question: str
    history: str
    context: str
    found_docs: str
    answer: str

def retrieve_documents(state: RAGState):
    with Session(db.engine) as session:
        query_embedding = embed(state["question"])
        DISTANCE_THRESHOLD = 0.55
        distance_col = VectorDB.embedding.cosine_distance(query_embedding)
        
        hits = (
            session.query(VectorDB)
            .filter(distance_col < DISTANCE_THRESHOLD)
            .order_by(distance_col)
            .limit(5)
            .all()
        )

        if not hits:
            return {"context": "", "found_docs": False}
        
        context = "\n\n".join(h.chunk for h in hits)
        return {"context": context, "found_docs": True}

# 3. Node: Generate Answer
def generate_answer(state: RAGState):
    full_prompt = f"""
    Document Context (ONLY source of truth):
    {state["context"]}
    
    Past Conversation:
    {state["history"]}
    
    User Question: {state["question"]}
    """
    
    reply_response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=full_prompt,
        config=types.GenerateContentConfig(
            system_instruction=(
                "You are a strict document QA assistant. Answer the user's question "
                "using ONLY the provided Document Context. Do NOT use any outside knowledge. "
                "If the answer cannot be found in the Document Context, reply exactly with: "
                "'I cannot answer this based on the uploaded documents.'"
            ),
            temperature=0.1,
        )
    )
    return {"answer": reply_response.text}

# 4. Node: Fallback (No Context)
def handle_no_docs(state: RAGState):
    return {"answer": "I cannot answer this based on the uploaded documents."}

# 5. Conditional Routing Logic
def route_after_retrieval(state: RAGState):
    if state["found_docs"]:
        return "generate"
    return "fallback"

# --- Compile the LangGraph Workflow ---
workflow = StateGraph(RAGState)

workflow.add_node("retrieve", retrieve_documents)
workflow.add_node("generate", generate_answer)
workflow.add_node("fallback", handle_no_docs)

workflow.set_entry_point("retrieve")
workflow.add_conditional_edges(
    "retrieve",
    route_after_retrieval,
    {
        "generate": "generate",
        "fallback": "fallback"
    }
)
workflow.add_edge("generate", END)
workflow.add_edge("fallback", END)

rag_graph = workflow.compile()

# --- Page Definitions ---
def login_page():
    with st.form("login", clear_on_submit=False, border=False):
        st.title("Login into RAG System")
        st.text("Login using your organizations id to use this system")
        email = st.text_input(placeholder="Enter email", label="Email", type="email")
        password = st.text_input(placeholder="Enter password", label="Password", type="password")
        submit = st.form_submit_button(label="Submit", use_container_width=True)
        signup = st.form_submit_button(label="No account? Sign Up", type="tertiary")

        if submit:
            with Session(db.engine) as session:
                user = session.query(Users).filter_by(email_address=email).first()
                if not user:
                    st.error("User not found !!")
                elif user.password == password:
                    st.session_state.email_address = user.email_address
                    st.session_state.name = user.name
                    st.session_state.user_id = user.id
                    st.success("Login Successfull!!")
                    st.switch_page(rag_route)
                else:
                    st.error("Invalid password entered !!")

        if signup:
            st.switch_page(signup_route)

def signup_page():
    with st.form("signup", clear_on_submit=False, border=False):
        st.title("Sign Up into RAG System")
        st.text("Add account to our RAG System")
        name = st.text_input(placeholder="Enter your name", label="Name")
        input_email = st.text_input(placeholder="Enter email", label="Email", type="email")
        input_password = st.text_input(placeholder="Enter password", label="Password", type="password")
        submit = st.form_submit_button(label="Submit", use_container_width=True)
        signin = st.form_submit_button(label="Already have an account? Sign In", type="tertiary")

        if submit:
            try:
                with Session(db.engine) as session:
                    user_input = Users(name=name, email_address=input_email, password=input_password)
                    session.add(user_input)
                    session.commit()
                    session.refresh(user_input)
                    
                    st.session_state.email_address = input_email
                    st.session_state.name = name
                    st.session_state.user_id = user_input.id
                    st.success("Account creation successfull !!")
                    st.switch_page(rag_route)
            except Exception as e:
                st.error(f"Unable to create account!!. {e}")

        if signin:
            st.switch_page(login_route)

def retrieve_history():
    with Session(db.engine) as session:
        user_sessions = session.query(Sessions).filter_by(user_id=st.session_state.user_id).all()
        if not user_sessions:
            st.text("No Chat history")
            return
            
        for s in user_sessions:
            if st.button(f"{s.title}", key=f"session_{s.id}", type="secondary", use_container_width=True):
                st.session_state.active_session_id = s.id
                st.session_state.messages = s.messages
                st.rerun()

def RAG_page():
    if "user_id" not in st.session_state:
        st.switch_page(login_route)

    st.title("RAG QA System")
    st.text("A RAG system designed for organization specific document data retreival")
    
    with st.sidebar:
        st.markdown(f"### 👤 {st.session_state.name}")
        st.caption(st.session_state.get("email_address", ""))
        
        if st.button("Logout", type="tertiary", use_container_width=True):
            st.session_state.clear()
            st.switch_page(login_route)
            
        st.divider()

        if st.button("New Chat", type="primary", use_container_width=True):
            st.session_state.messages = []
            st.session_state.pop("active_session_id", None)
            st.rerun()

        st.text("History")
        retrieve_history()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Message here", accept_file="multiple", file_type=["pdf"]):
        
        # 1. Process Files
        if prompt.files:
            with st.spinner("Chunking and saving to VectorDB..."):
                with Session(db.engine) as session:
                    for file in prompt.files:
                        file_bytes = file.getvalue()
                        file_hash = hashlib.sha256(file_bytes).hexdigest()
                        
                        existing = session.query(VectorDB).filter_by(file_hash=file_hash).first()
                        if existing:
                            st.warning(f"'{file.name}' is already in the database. Skipping.")
                            continue
                        
                        reader = PyPDF2.PdfReader(file)
                        text = "".join(page.extract_text() or "" for page in reader.pages)
                        chunks = [text[i:i+1000] for i in range(0, len(text), 1000) if text[i:i+1000].strip()]
                        
                        for chunk in chunks:
                            vector = embed(chunk)
                            session.add(VectorDB(
                                filename=file.name, 
                                file_hash=file_hash, 
                                chunk=chunk, 
                                embedding=vector
                            ))
                            
                    session.commit()
                    st.success("Files processed!")

        # 2. Process Chat Text
        if prompt.text:
            st.chat_message("user").write(prompt.text)
            st.session_state.messages.append({"role": "user", "content": prompt.text})
            
            with Session(db.engine) as session:
                # Retrieve matching chunks
                query_embedding = embed(prompt.text)
                hits = session.query(VectorDB).order_by(
                    VectorDB.embedding.cosine_distance(query_embedding)
                ).limit(5).all()

                DISTANCE_THRESHOLD = 0.55
                distance_col = VectorDB.embedding.cosine_distance(query_embedding)
                
                hits = (
                    session.query(VectorDB)
                    .filter(distance_col < DISTANCE_THRESHOLD)
                    .order_by(distance_col)
                    .limit(5)
                    .all()
                )

                if not hits:
                    reply = "No such information is present in my database."
                else:
                    
                    context = "\n\n".join(h.chunk for h in hits)
                    
                    history_text = "\n".join(
                        f"{m['role'].capitalize()}: {m['content']}" 
                        for m in st.session_state.messages[:-1]
                    )

                    full_prompt = f"""
                    Document Context (ONLY source of truth):
                    {context}
                    
                    Past Conversation:
                    {history_text}
                    
                    User Question: {prompt.text}
                    """

                    with st.chat_message("assistant"):
                        with st.spinner("Thinking..."):
                            reply_response = client.models.generate_content(
                                model="gemini-3.5-flash",
                                contents=full_prompt,
                                config=types.GenerateContentConfig(
                                    system_instruction=(
                                    "You are a strict document QA assistant. Answer the user's question "
                                    "using ONLY the provided Document Context. Do NOT use any outside knowledge. "
                                    "If the answer cannot be found in the Document Context, reply exactly with: "
                                    "'I cannot answer this based on the uploaded documents.'"
                                    ),
                                    temperature=0.1,
                                ),
                            )
                            reply = reply_response.text

                        st.write(reply)

                # st.chat_message("assistant").write(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})

                # Save session history to database
                if "active_session_id" in st.session_state:
                    active = session.get(Sessions, st.session_state.active_session_id)
                    active.messages = list(st.session_state.messages)
                else:
                    new_sess = Sessions(
                        user_id=st.session_state.user_id, 
                        title=prompt.text[:25], 
                        messages=st.session_state.messages
                    )
                    session.add(new_sess)
                    session.flush()
                    st.session_state.active_session_id = new_sess.id
                session.commit()

# --- Global Page Routing ---
login_route = st.Page(login_page, title="login", url_path="login")
signup_route = st.Page(signup_page, title="signup", url_path="signup")
rag_route = st.Page(RAG_page, title="RAG", url_path="rag")

pg = st.navigation([login_route, signup_route, rag_route], position="hidden")
pg.run()