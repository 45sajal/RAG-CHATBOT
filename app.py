import os
from dotenv import load_dotenv


import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SpacyEmbeddings
from langchain.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForCausalLM
import json

load_dotenv()
os.getenv("HUGGINGFACE_HUB_TOKEN")

embeddings = SpacyEmbeddings(model_name="en_core_web_sm")
model_name = "KingNish/Qwen2.5-0.5b-Test-ft"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

pdf_folder = "PDF/"


def pdf_read(pdf_folder):
    text = ""
    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_folder, pdf_file)
        pdf_reader = PdfReader(pdf_path)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks


def vector_store(text_chunks):
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_db")


def search_documents(query, retriever, top_k=2):
    results = retriever.invoke(query)
    results = results[:top_k]
    return results


def generate_similar_queries(query, num_queries=3):
    input_text = f"Generate similar questions to: {query}"
    inputs = tokenizer(
        input_text, return_tensors="pt", truncation=True, padding=True, max_length=1024
    )
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        num_return_sequences=num_queries,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    similar_queries = [
        tokenizer.decode(output, skip_special_tokens=True) for output in outputs
    ]
    # print(similar_queries)
    return similar_queries


def get_conversational_chain(question, context):
    input_text = (
        f"Use the following context to answer the question.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        "Provide a clear and concise answer:"
    )
    inputs = tokenizer(
        input_text, return_tensors="pt", truncation=True, padding=True, max_length=1024
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    response = (
        response.replace(context, "")
        .replace(question, "")
        .replace("Context:", "")
        .replace("Question:", "")
        .strip()
    )

    return response


def user_input(user_question):
    new_db = FAISS.load_local(
        "faiss_db", embeddings, allow_dangerous_deserialization=True
    )
    retriever = new_db.as_retriever()
    if "history" not in st.session_state:
        st.session_state.history = []

    if any(
        keyword in user_question.lower()
        for keyword in ["summarize", "last", "previous"]
    ):
        if not st.session_state.history:
            return "No prior answers to reference."

        past_context = "\n".join(
            [
                f"Q{i+1}: {entry['question']} | A: {entry['answer']}"
                for i, entry in enumerate(st.session_state.history)
            ]
        )

        input_text = (
            f"See your previous outputs and generate a summary for the user."
            f"Here is the conversation history:\n\n{past_context}\n\n"
            f"Question: {user_question}\n"
        )
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=1024,
        )
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        response = response.replace(past_context, "").replace(user_question, "").strip()

        return response

    similar_queries = generate_similar_queries(user_question)
    print("Generated Similar Queries:")
    for idx, query in enumerate(similar_queries, start=1):
        print(f"{idx}. {query}")

    results = search_documents(user_question, retriever)
    context = " ".join([result.page_content for result in results])

    response = get_conversational_chain(user_question, context)

    st.session_state.history.append({"question": user_question, "answer": response})

    with open("conversation_history.json", "w") as f:
        json.dump(st.session_state.history, f, indent=4)

    return response


def display_history():
    st.sidebar.header("Conversation History")
    for i, entry in enumerate(st.session_state.history):
        with st.sidebar.expander(f"Q{i + 1}: {entry['question']}"):
            st.sidebar.write(f"**Q{i + 1}:** {entry['question']}")
            st.sidebar.write(f"**A:** {entry['answer']}")


def main():
    st.set_page_config(
        page_title="RAG-based Chat with PDF", page_icon=":books:", layout="wide"
    )
    st.header("RAG-based Chat with PDF")

    if "history" not in st.session_state:
        st.session_state.history = []

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        response = user_input(user_question)
        st.write(f"Answer: {response}")

    display_history()

    if not os.path.exists("faiss_db"):
        with st.spinner("Initializing..."):
            raw_text = pdf_read(pdf_folder)
            text_chunks = get_chunks(raw_text)
            vector_store(text_chunks)
            st.success("PDFs processed and FAISS index created!")


if __name__ == "__main__":
    main()


# import os
# import streamlit as st
# from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import SpacyEmbeddings
# from langchain.vectorstores import FAISS
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import json

# os.environ["HUGGINGFACE_HUB_TOKEN"] = "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"

# embeddings = SpacyEmbeddings(model_name="en_core_web_sm")
# model_name = "KingNish/Qwen2.5-0.5b-Test-ft"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)

# pdf_folder = "PDF/"


# def pdf_read(pdf_folder):
#     text = ""
#     pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]
#     for pdf_file in pdf_files:
#         pdf_path = os.path.join(pdf_folder, pdf_file)
#         pdf_reader = PdfReader(pdf_path)
#         for page in pdf_reader.pages:
#             text += page.extract_text()
#     return text


# def get_chunks(text):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     chunks = text_splitter.split_text(text)
#     return chunks


# def vector_store(text_chunks):
#     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
#     vector_store.save_local("faiss_db")


# def search_documents(query, retriever, top_k=2):
#     results = retriever.invoke(query)
#     results = results[:top_k]
#     return results


# def get_conversational_chain(question, context):
#     input_text = (
#         f"Use the following context to answer the question.\n\n"
#         f"Context:\n{context}\n\n"
#         f"Question:\n{question}\n\n"
#         "Provide a clear and concise answer:"
#     )
#     inputs = tokenizer(
#         input_text, return_tensors="pt", truncation=True, padding=True, max_length=1024
#     )

#     outputs = model.generate(
#         **inputs,
#         max_new_tokens=300,
#         num_return_sequences=1,
#         eos_token_id=tokenizer.eos_token_id,
#         pad_token_id=tokenizer.pad_token_id,
#     )

#     response = tokenizer.decode(outputs[0], skip_special_tokens=True)

#     response = (
#         response.replace(context, "")
#         .replace(question, "")
#         .replace("Context:", "")
#         .replace("Question:", "")
#         .strip()
#     )

#     return response


# def user_input(user_question):
#     new_db = FAISS.load_local(
#         "faiss_db", embeddings, allow_dangerous_deserialization=True
#     )
#     retriever = new_db.as_retriever()
#     results = search_documents(user_question, retriever)

#     context = " ".join([result.page_content for result in results])

#     response = get_conversational_chain(user_question, context)

#     if "history" not in st.session_state:
#         st.session_state.history = []

#     st.session_state.history.append({"question": user_question, "answer": response})

#     with open("conversation_history.json", "w") as f:
#         json.dump(st.session_state.history, f, indent=4)

#     return response


# def display_history():
#     st.sidebar.header("Conversation History")
#     for i, entry in enumerate(st.session_state.history):
#         with st.sidebar.expander(f"Q{i + 1}: {entry['question']}"):
#             st.sidebar.write(f"**Q{i + 1}:** {entry['question']}")
#             st.sidebar.write(f"**A:** {entry['answer']}")


# def main():
#     st.set_page_config(
#         page_title="RAG-based Chat with PDF", page_icon=":books:", layout="wide"
#     )
#     st.header("RAG-based Chat with PDF")

#     if "history" not in st.session_state:
#         st.session_state.history = []

#     user_question = st.text_input("Ask a Question from the PDF Files")

#     if user_question:
#         response = user_input(user_question)
#         st.write(f"Answer: {response}")

#     display_history()

#     if not os.path.exists("faiss_db"):
#         with st.spinner("Initializing..."):
#             raw_text = pdf_read(pdf_folder)
#             text_chunks = get_chunks(raw_text)
#             vector_store(text_chunks)
#             st.success("PDFs processed and FAISS index created!")


# if __name__ == "__main__":
#     main()
