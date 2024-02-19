# Delaware-Paramedic-Protocols
Delaware Paramedic Protocols
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.models import ChatOpenAI

directory_path = "/user/ryan/desktop/ALSFinal2022.pdf"

def create_chatbot():
    documents = load_documents(/user/ryan/desktop/ALSFinal2022.pdf)
    split_documents = split_text(documents)
    vectorstore = create_vectorstore(split_documents)
    return create_conversational_chain(vectorstore)

def load_documents(/user/ryan/desktop/ALSFinal2022.pdf):
    document_types = [.pdf] 
    loaders = [DirectoryLoader(directory_path, glob=doc_type) for doc_type in document_types]
    documents = [doc for loader in loaders for doc in loader.load()]
    print(f"Total number of documents: {len(documents)}") 
    return documents

def split_text(documents):
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    split_docs = text_splitter.split_documents(documents)
    return split_docs

def create_vectorstore(documents): 
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(documents, embeddings)
    return vectorstore

def create_conversational_chain(vectorstore):
    llm = ChatOpenAI(temperature=0, model_name="gpt-4")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    conversational_chain = ConversationalRetrievalChain.from_llm(
        llm,
        vectorstore.as_retriever(),
        memory=memory,
        verbose=True
    )
    return conversational_chain

chat_bot = create_chatbot()
