
# E2E Help Chatbot

## Description
The E2E Help Chatbot is an end-to-end solution designed to answer questions using specific documentation as a knowledge base. It utilizes a combination of language models and vector space search to find relevant information from a set of documents. The chatbot is built using Streamlit, allowing it to be deployed as a web application.

## Features
- Uses the Mistral model from Hugging Face for natural language understanding and generation.
- Employs FAISS for efficient similarity search in large datasets.
- Integrates a web-based loader to fetch and index documents from URLs provided in `links.txt`.
- Features a recursive text splitter to handle large documents.
- Maintains conversational context using a memory buffer.

## Prerequisites
Before you begin, ensure you have met the following requirements:
- Python 3.7 or later
- Streamlit
- Hugging Face Transformers
- PyTorch
- FAISS
- Langchain
- Sentence Transformers

## Installation
To install the required dependencies for the E2E Help Chatbot, run the following command:

```bash
pip install streamlit transformers torch faiss-cpu langchain sentence-transformers
```

Make sure you have a file named `links.txt` in the root directory with URLs to the documents that will populate the vector store.

## Usage
To start the chatbot application, execute:

```bash
streamlit run chatbot.py
```

Then, navigate to the localhost URL (usually `http://localhost:8501`) to interact with the chatbot.

## How It Works
1. **Initialization**: Upon starting the app, it initializes the language model, embeddings model, and vector store.
2. **User Interaction**: Users can enter their questions into the chat interface.
3. **Query Processing**: The chatbot rephrases the question to be standalone and uses it to retrieve relevant documents.
4. **Answer Generation**: The chatbot generates a response based on the retrieved documents and presents it to the user.
5. **Memory Management**: The chatbot saves the context of the conversation for future interactions.

## Code Structure
- `load_llm()`: Loads the language model and tokenizer.
- `embeddings_model()`: Initializes the embeddings model.
- `initialize_vectorstore()`: Reads URLs from `links.txt`, fetches documents, and initializes the vector store.
- `return_chain_elements()`: Sets up the chain of operations for question rephrasing, document retrieval, and answer generation.
- Streamlit Chat Interface: Manages the chat interface and user interactions.

## Contributing
To contribute to the E2E Help Chatbot, follow these steps:
1. Fork the repository.
2. Create a branch: `git checkout -b <branch_name>`.
3. Make your changes and commit them: `git commit -m '<commit_message>'`.
4. Push to the original branch: `git push origin <project_name>/<location>`.
5. Create the pull request.

Alternatively, see the GitHub documentation on [creating a pull request](https://help.github.com/articles/creating-a-pull-request/).

## License
The E2E Help Chatbot is available under the MIT license. See the LICENSE file for more info.
