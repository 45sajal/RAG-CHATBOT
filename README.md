# Mental Health Chatbot

A locally deployed mental health chatbot that allows users to ask questions about mental health, view the conversation history, and generate summaries of the chatbot's responses. The chatbot uses advanced retrieval and language model techniques to provide accurate and context-aware answers.

## Features

- **Interactive Question-Answering**: Ask questions about mental health and receive helpful responses.
- **Conversation History**: View the entire conversation history in a sidebar for context.
- **Response Summaries**: Generate concise summaries of chatbot responses on request.
- **Advanced Retrieval**: Implements RAG Fusion, where multiple queries retrieve multiple chunks of data, and the most relevant chunk is sent to a language model for response generation.

## Technology Stack

- **FAISS**: For efficient embeddings and similarity searches.
- **RAG Fusion**: Retrieval-Augmented Generation combining multiple data chunks.
- **Streamlit**: For a seamless local deployment and interactive user interface.
- **Small Language Model**: Lightweight and efficient for local deployment.

## Installation

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd <repository-folder>

   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt

   ```

3. Run the application:
   ```bash
   streamlit run app.py
   ```
