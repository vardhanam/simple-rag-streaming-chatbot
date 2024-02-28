from transformers import (
AutoTokenizer,
AutoModelForCausalLM,
BitsAndBytesConfig,
pipeline
)

import torch

import streamlit as st

import os

from langchain.llms import HuggingFacePipeline
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, format_document
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string

from operator import itemgetter


@st.cache_resource()
def load_llm():

    #Loading the Mistral Model
    model_name='mistralai/Mistral-7B-Instruct-v0.2'
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )


    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
    )

    # Building a LLM text-generation pipeline
    text_generation_pipeline = pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        temperature=0.2,
        repetition_penalty=1.1,
        return_full_text=True,
        max_new_tokens=1024,
    )

    llm = HuggingFacePipeline(pipeline= text_generation_pipeline)

    return llm

@st.cache_resource()
def embeddings_model():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    return embeddings

@st.cache_resource()
def initialize_vectorstore():

    # Read URLs from the links.txt file and store them in a list
    with open('links.txt', 'r') as file:
        urls_list = [line.strip() for line in file if line.strip()]

    urls_list = list(set(urls_list))

    #Initializing a text_splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )

    #Loading all the content of the urls in docs format
    loader = WebBaseLoader(urls_list)
    docs = loader.load_and_split(text_splitter=text_splitter)

    vectorstore = FAISS.from_documents(
        docs, embedding=hf_embeddings
    )
    retriever = vectorstore.as_retriever(search_kwargs = {'k':10})
    return retriever


@st.cache_resource()
def return_chain_elements():

    #template to get the Standalone question
    _template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

        Chat History:
        {chat_history}
        Follow Up Input: {question}
        Standalone question:
    """
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

    #Function to create the context from retrieved documents
    DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")
    def _combine_documents(
        docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
    ):
        doc_strings = [format_document(doc, document_prompt) for doc in docs]
        return document_separator.join(doc_strings)

    #Creating the template for the final answer
    template = """Answer the question based only on the following context:
        {context}

        Question: {question}
    """
    ANSWER_PROMPT = ChatPromptTemplate.from_template(template)


    # Now we calculate the standalone question
    standalone_question = {
        "standalone_question": {
            "question": lambda x: x["question"],
            "chat_history": lambda x: get_buffer_string(x["chat_history"]),
        }
        | CONDENSE_QUESTION_PROMPT
        | llm
        | StrOutputParser(),
    }

    # Now we retrieve the documents
    retrieved_documents = {
        "docs": itemgetter("standalone_question") | vector_retriever,
        "question": lambda x: x["standalone_question"],
    }

    # Now we construct the inputs for the final prompt
    final_inputs = {
        "context": lambda x: _combine_documents(x["docs"]),
        "question": itemgetter("question"),
    }

    # And finally, we do the part that returns the answers
    answer = {
        "answer": final_inputs | ANSWER_PROMPT | llm,
        "docs": itemgetter("docs"),
    }

    return standalone_question, retrieved_documents, answer


llm = load_llm()

hf_embeddings = embeddings_model()

vector_retriever = initialize_vectorstore()

standalone_question, retrieved_documents, answer = return_chain_elements()

#Title for the streamlit app
st.title('E2E Help Chatbot')

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Creating conversational memory for the chain
if "conversational_memory" not in st.session_state:
    st.session_state.conversational_memory = ConversationBufferMemory(
        return_messages=True, output_key="answer", input_key="question"
    )

# First we add a step to load memory
# This adds a "memory" key to the input object
loaded_memory = RunnablePassthrough.assign(
    chat_history=RunnableLambda(st.session_state.conversational_memory.load_memory_variables) | itemgetter("history"),
)

#Creating a chain for question answering
chain = loaded_memory | standalone_question | retrieved_documents | answer

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("How can I help you with E2E docs?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing Query"):
            inputs = {"question": prompt}
            result = chain.invoke(inputs)
            final_answer = (
                result["answer"] + "\n\n"
                "For further assistance follow the links below:\n"
                f"1. {result['docs'][0].metadata['source']}\n"
                f"2. {result['docs'][1].metadata['source']}\n"
                f"3. {result['docs'][2].metadata['source']}"
            )
            st.markdown(final_answer)

    st.session_state.conversational_memory.save_context(inputs, {"answer": result["answer"]})
    st.session_state.messages.append({"role": "assistant", "content": final_answer})

