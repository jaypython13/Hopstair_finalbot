
#!pip install -q pypdf
#!pip -q install sentence-transformers
#!pip install langchain
#!pip install streamlit
#!pip install chromadb
#!pip install unstructured -q
#!pip install unstructured[local-inference] -q
# !pip install -U langchain-community

from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import Chroma
import logging
import sys
import streamlit as st
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms import LlamaCPP
from llama_index.llms.llama_utils import messages_to_prompt, completion_to_prompt
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.embeddings import LangchainEmbedding
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
import os
import tempfile
global text, documents, index

__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Set up logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

def greet_user(name):
    st.write(f" Hello {name}!,we love having you.I am your AI Buddy.Remember, I am a motivational bot that helps break unconfident or negative thought patterns to help you regain your spark. We do not diagnose or treat mental health symptoms or conditions. We are all about confidence building and boosting.\
    Type quit to exit")

def avoid_words(answer):
    # Define keywords that suggest consulting a professional
    forbidden_keywords = ["counselor","professional counselor", "Based on the context provided", "therapist", "psychiatrist", "career counselor", 'medical help', 'mental health professional', "therapy", 'medical counsellor', "counseling"]
    blob_object= TextBlob(answer)
    answer_words = blob_object.words
    # Check if any forbidden keyword is present in the response
    for keyword in forbidden_keywords:
        if keyword in answer:
            return False

    return True

def get_answer(query):
  similar_docs = get_similar_docs(query)
  chain= load_qa_chain(llm, chain_type="stuff", prompt=prompt)
  answer = chain.run(input_documents = similar_docs, question = query, prompt=prompt)
  avoid_words(answer)
  return answer

def get_similar_docs(query,k=1,score=False,index):
  if score:
    similar_docs = index.similarity_search_with_score(query,k=k)
  else:
    similar_docs = index.similarity_search(query,k=k)
  return similar_docs
    
def configure_llama_model():
    model_url = 'https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf'
    llm = LlamaCPP(
        model_url=model_url,
        temperature=0.1,
        max_new_tokens=256,
        context_window=3900,
        model_kwargs={"n_gpu_layers": -1},
        messages_to_prompt=messages_to_prompt,
        completion_to_prompt=completion_to_prompt,
        verbose=True,
    )
    return llm
def main():
    st.image('hs-logo.png', width = 200)
    st.title("AI Buddy- Your Confidence Catalyst")
    #st.sidebar.title("Hopstair's library data is Processing")
    loader = PyPDFLoader("data/Hopstair_data.pdf")
    documents = loader.load()
    name =  st.text_input("Hope you are well ! Please type your name to begin. ")
    if name:
        greet_user(name)
    query = st.text_input("Type your query here:")
    if query == "quit":
        st.write("Goodbye! Have a great day! Take care, my dear friend. Wishing you all the happiness and success in the world 💪")
    else:
        response = get_answer(query)
        st.text_area(response, height =100)
    # Split the data into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(documents=documents) 
    # Create Embeddings 
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    # load it into Chroma
    index = Chroma.from_documents(text_chunks, embeddings)
    # Configure and initialize components
    llm = configure_llama_model()    

    
    #if query:
        #response = get_answer(query)
        #st.text_area(response, height =100)
    #if query.lower() == "quit":
        #st.write("Goodbye! Have a great day! Take care, my dear friend. Wishing you all the happiness and success in the world 💪")
        
        #break

        #response = get_answer(query)
        #query_engine = index.as_query_engine() 
        #response = query_engine.query(user_input) 
        #t = st.text_area("AI Response:", response, height=100)   
            
if __name__ == "__main__":
    main()

#!streamlit run /usr/local/lib/python3.10/dist-packages/colab_kernel_launcher.py


