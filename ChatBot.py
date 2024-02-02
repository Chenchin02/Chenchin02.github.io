import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from dotenv import dotenv_values


env_values = dotenv_values('static/.env.txt')
#UTF-8 BOM問題
qdrant_api_key = env_values.get('\ufeffQDRANT_API_KEY') or env_values.get('QDRANT_API_KEY')

class DocumentChatAssistant:
    OPENAI_EMBEDDING_DEPLOYMENT_NAME = "embedding-ada-002"
    OPENAI_EMBEDDING_MODEL_NAME = "text-embedding-ada-002"
    MODEL_NAME = "gpt-4-1106-preview"
    QDRANT_API_URL = 'https://be9d34cf-cd98-4f1f-abb8-62c65a53289b.us-east4-0.gcp.cloud.qdrant.io:6333'
    QDRANT_API_KEY = qdrant_api_key

    def __init__(self, openai_api_key, pdf_filepath="SanreyQA.pdf"):
        self.openai_api_key = openai_api_key
        self.pdf_filepath = pdf_filepath
        self.docs = self.load_and_split_documents()
        self.embeddings = self.get_embeddings()
        self.doc_store = self.get_document_store(self.docs, self.embeddings)
    def load_and_split_documents(self):
        loader = PyPDFLoader(self.pdf_filepath)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        return text_splitter.split_documents(documents)
    def get_embeddings(self):
        return OpenAIEmbeddings(
            openai_api_key=self.openai_api_key,
            deployment=self.OPENAI_EMBEDDING_DEPLOYMENT_NAME,
            model=self.OPENAI_EMBEDDING_MODEL_NAME,
            chunk_size=1
        )
    def get_document_store(self, docs, embeddings):
        return Qdrant.from_documents(
            docs,
            embeddings,
            url=self.QDRANT_API_URL,
            api_key=self.QDRANT_API_KEY,
            collection_name="ruth",
            force_recreate=True
        )
    def get_chat_model(self):
        return ChatOpenAI(
            openai_api_key=self.openai_api_key,
            model_name=self.MODEL_NAME,
            temperature=0
        )
    def ask_question_with_context(self, qa, question, chat_history):
        chat_history_convert = []
        for message in chat_history:
            tmp_tuple = tuple()
            for char in message:
                tmp_tuple += tuple(char)
            chat_history_convert.append(tmp_tuple)

        query = ""
        result = qa.invoke({"question": question, "chat_history": chat_history_convert})
        chat_history = [(query, result["answer"])]
        return chat_history

    def main(self, query, chat_history):
        llm = self.get_chat_model()

        qa = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=self.doc_store.as_retriever(),
            return_source_documents=True,
            verbose=False
        )

        chat_history = self.ask_question_with_context(qa, query, chat_history)
        return chat_history
