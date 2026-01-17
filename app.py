from langchain_groq import ChatGroq
import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from htmlTemplates import css, bot_template, user_template
import json
import os
from dotenv import load_dotenv
import time

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("⚠️ GROQ_API_KEY not found! Please set it in your environment variables or Hugging Face secrets.")
    st.stop()


def call_groq_with_retry(llm, prompt, max_retries=3, initial_delay=2):
    for attempt in range(max_retries):
        try:
            response = llm.invoke(prompt)
            return response, None
        except Exception as e:
            error_msg = str(e).lower()
            
            if "rate limit" in error_msg or "429" in error_msg or "quota" in error_msg:
                if attempt < max_retries - 1:
                    wait_time = initial_delay * (2 ** attempt)  
                    st.warning(f"⏳ Rate limit reached. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    return None, " **API rate limit reached.** Please try again in a few minutes. Groq's free tier has usage limits."
            else:
                return None, f"❌ Error: {str(e)}"
    
    return None, 


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(raw_text):
    splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_text(raw_text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    vector_store = FAISS.from_texts(text_chunks, embeddings)
    return vector_store


def get_conversation_chain(vector_store):
    llm = ChatGroq(model='llama-3.3-70b-versatile', api_key=GROQ_API_KEY)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer questions based on the provided context and the answer the question in user asked question language only. If you don't know, say so."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "Context: {context}\n\nQuestion: {question}")
    ])
    
    chain = prompt | llm | StrOutputParser()
    
    return {
        "chain": chain,
        "vector_store": vector_store,
        "llm": llm
    }


def generate_quiz(vector_store):
    llm = ChatGroq(model='llama-3.3-70b-versatile', api_key=GROQ_API_KEY)
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 10})
    docs = retriever.invoke("Generate questions about key concepts")
    context = "\n\n".join([doc.page_content for doc in docs])
    
    quiz_prompt = f"""Based on the following context, generate exactly 5 multiple-choice questions to test understanding.

Context: {context}

Return ONLY a valid JSON array with this exact structure (no markdown, no extra text):
[
  {{
    "question": "Question text here?",
    "options": ["Option A", "Option B", "Option C", "Option D"],
    "correct": 0
  }}
]

Rules:
- Generate exactly 5 questions
- Each question must have exactly 4 options
- The "correct" field must be the index (0-3) of the correct option
- Make questions clear and options distinct
- Return ONLY the JSON array, nothing else"""
    
    response, error = call_groq_with_retry(llm, quiz_prompt)
    
    if error:
        st.error(error)
        return None
    
    try:
        response_text = response.content if hasattr(response, 'content') else str(response)
        response_text = response_text.strip()
        
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
            response_text = response_text.strip()
        
        quiz_data = json.loads(response_text)
        return quiz_data
    except json.JSONDecodeError as e:
        st.error(f"Error parsing quiz data: {e}")
        st.text(f"Response received: {response_text[:500]}")
        return None


if "conversation" not in st.session_state:
    st.session_state.conversation = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "mode" not in st.session_state:
    st.session_state.mode = "chat"

if "quiz_data" not in st.session_state:
    st.session_state.quiz_data = None

if "current_question" not in st.session_state:
    st.session_state.current_question = 0

if "score" not in st.session_state:
    st.session_state.score = 0

if "answers" not in st.session_state:
    st.session_state.answers = []

if "quiz_completed" not in st.session_state:
    st.session_state.quiz_completed = False


def handle_userinput(user_question):
    if st.session_state.conversation is None:
        st.warning("Please upload and process PDF documents first.")
        return
    
    chain = st.session_state.conversation["chain"]
    vector_store = st.session_state.conversation["vector_store"]
    llm = st.session_state.conversation["llm"]
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    docs = retriever.invoke(user_question)
    
    context = "\n\n".join([doc.page_content for doc in docs])
    
    full_prompt = {
        "context": context,
        "question": user_question,
        "chat_history": st.session_state.chat_history
    }
    try:
        response = chain.invoke(full_prompt)
        
        st.session_state.chat_history.append(HumanMessage(content=user_question))
        st.session_state.chat_history.append(AIMessage(content=response))
        
        st.write(user_template.replace("{{MSG}}", user_question), unsafe_allow_html=True)
        st.write(bot_template.replace("{{MSG}}", response), unsafe_allow_html=True)
        
    except Exception as e:
        error_msg = str(e).lower()
        if "rate limit" in error_msg or "429" in error_msg or "quota" in error_msg:
            st.error(" **API rate limit reached.** Please try again in a few minutes. Groq's free tier has usage limits.")
        else:
            st.error(f"❌ An error occurred: {str(e)}")


def reset_quiz():
    st.session_state.current_question = 0
    st.session_state.score = 0
    st.session_state.answers = []
    st.session_state.quiz_completed = False
    st.session_state.quiz_data = None


st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
st.header("Multilingual RAG")

col1, col2 = st.columns(2)
with col1:
    if st.button("Chat Mode", use_container_width=True, type="primary" if st.session_state.mode == "chat" else "secondary"):
        st.session_state.mode = "chat"
        st.rerun()
        
with col2:
    if st.button("Quiz Mode", use_container_width=True, type="primary" if st.session_state.mode == "quiz" else "secondary"):
        if st.session_state.conversation is None:
            st.warning("Please upload and process PDFs first!")
        else:
            st.session_state.mode = "quiz"
            reset_quiz()
            st.rerun()

st.markdown("---")

if st.session_state.mode == "chat":
    for msg in st.session_state.chat_history:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(msg.content)

    user_question = st.chat_input("Ask something about your documents…")
    if user_question:
        handle_userinput(user_question)

elif st.session_state.mode == "quiz":
    if st.session_state.quiz_data is None and not st.session_state.quiz_completed:
        st.info("Generate a quiz to test your knowledge about the uploaded PDFs!")
        if st.button("Generate Quiz", type="primary"):
            with st.spinner("Generating quiz questions..."):
                quiz_data = generate_quiz(st.session_state.conversation["vector_store"])
                if quiz_data:
                    st.session_state.quiz_data = quiz_data
                    st.rerun()
    
    elif st.session_state.quiz_data and not st.session_state.quiz_completed:
        quiz = st.session_state.quiz_data
        current = st.session_state.current_question
        
        if current < len(quiz):
            st.subheader(f"Question {current + 1} of {len(quiz)}")
            st.progress((current) / len(quiz))
            
            question_data = quiz[current]
            st.markdown(f"### {question_data['question']}")
            
            selected = st.radio(
                "Select your answer:",
                options=range(len(question_data['options'])),
                format_func=lambda x: question_data['options'][x],
                key=f"q_{current}"
            )
            
            if st.button("Submit Answer", type="primary"):
                is_correct = selected == question_data['correct']
                
                if is_correct:
                    st.success("✅ Correct!")
                    st.session_state.score += 1
                else:
                    st.error(f"❌ Incorrect. The correct answer was: {question_data['options'][question_data['correct']]}")
                
                st.session_state.answers.append({
                    "question": question_data['question'],
                    "selected": selected,
                    "correct": question_data['correct'],
                    "is_correct": is_correct
                })
                
                st.session_state.current_question += 1
                
                if st.session_state.current_question >= len(quiz):
                    st.session_state.quiz_completed = True
                
                st.rerun()
    
    elif st.session_state.quiz_completed:
        st.balloons()
        st.success("🎉 Quiz Completed!")
        
        score = st.session_state.score
        total = len(st.session_state.quiz_data)
        percentage = (score / total) * 100
        
        st.markdown(f"## Your Score: {score}/{total} ({percentage:.1f}%)")
        
        if percentage >= 80:
            st.success("🌟 Excellent work!")
        elif percentage >= 60:
            st.info("👍 Good job!")
        else:
            st.warning("📚 Keep studying!")
        
        st.markdown("### Review Your Answers:")
        for i, answer in enumerate(st.session_state.answers):
            with st.expander(f"Question {i+1}: {'✅' if answer['is_correct'] else '❌'}"):
                st.markdown(f"**{answer['question']}**")
                quiz_q = st.session_state.quiz_data[i]
                st.write(f"Your answer: {quiz_q['options'][answer['selected']]}")
                st.write(f"Correct answer: {quiz_q['options'][answer['correct']]}")
        
        if st.button("🔄 Take Quiz Again", type="primary"):
            reset_quiz()
            st.rerun()

st.write(css, unsafe_allow_html=True)

with st.sidebar:
    st.subheader("Your documents")
    pdf_docs = st.file_uploader(
        "Upload your PDFs here and click on 'Process'", 
        accept_multiple_files=True
    )
    
    if st.button('Process'):
        if not pdf_docs:
            st.warning("Please upload at least one PDF file.")
        else:
            with st.spinner('Processing'):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vector_store = get_vector_store(text_chunks)
                
                st.session_state.conversation = get_conversation_chain(vector_store)
                st.session_state.chat_history = []
                reset_quiz()
                st.success("Documents processed successfully!")
                st.rerun()