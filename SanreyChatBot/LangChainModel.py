from langchain.text_splitter import CharacterTextSplitter    #文本分割工具
# from langchain.document_loaders import UnstructuredFileLoader  #下載非結構化文件
from langchain_community.document_loaders import UnstructuredFileLoader  #下載非結構化文件

from langchain.vectorstores.faiss import FAISS
# from langchain.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.prompts.prompt import PromptTemplate
from langchain.vectorstores.base import VectorStoreRetriever
# from langchain.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
import pickle
import os
from typing import Optional, Tuple
from threading import Lock
# import gradio as gr
os.environ["OPENAI_API_KEY"] = "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"

print("Loading data...")
loader = UnstructuredFileLoader(os.path.dirname(os.path.abspath(__file__)) + "/static/langchainData.txt")
raw_documents = loader.load()

print("Splitting text...")
text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=600,
    chunk_overlap=100,
    length_function=len,
)
documents = text_splitter.split_documents(raw_documents)

print("Creating vectorstore...")
embeddings = OpenAIEmbeddings()  #將文本轉換為向量
vectorstore = FAISS.from_documents(documents, embeddings)
with open("vectorstore.pkl", "wb") as f:
    pickle.dump(vectorstore, f)


_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.
You can assume the question about the most recent state of the union address.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

template = """你是翔睿的公司客服代理人，請根據文檔中的資料回復給顧客，如果顧客問到文檔中未出現的問題，請回答不知道，不要亂回答給顧客，請全力根據顧客給的訊息搜索資料庫.
Question: {question}
=========
{context}
=========
Answer in Markdown:"""
QA_PROMPT = PromptTemplate(template=template, input_variables=[
                           "question", "context"])


def load_retriever():
    with open("vectorstore.pkl", "rb") as f:
        vectorstore = pickle.load(f)
    retriever = VectorStoreRetriever(vectorstore=vectorstore)
    return retriever


def get_basic_qa_chain():
    llm = ChatOpenAI(model_name="gpt-4", temperature=0)
    retriever = load_retriever()
    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True)
    model = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory)
    return model


def get_custom_prompt_qa_chain():
    llm = ChatOpenAI(model_name="gpt-4", temperature=0)
    retriever = load_retriever()
    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True)
    # see: https://github.com/langchain-ai/langchain/issues/6635
    # see: https://github.com/langchain-ai/langchain/issues/1497
    model = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT})
    return model


def get_condense_prompt_qa_chain():
    llm = ChatOpenAI(model_name="gpt-4", temperature=0)
    retriever = load_retriever()
    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True)
    # see: https://github.com/langchain-ai/langchain/issues/5890
    model = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT})
    return model


def get_qa_with_sources_chain():
    llm = ChatOpenAI(model_name="gpt-4", temperature=0)
    retriever = load_retriever()
    history = []
    model = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True)

    def model_func(question):
        # bug: this doesn't work with the built-in memory
        # hacking around it for the tutorial
        # see: https://github.com/langchain-ai/langchain/issues/5630
        new_input = {"question": question['question'], "chat_history": history}
        result = model(new_input)
        history.append((question['question'], result['answer']))
        return result

    return model_func

# 可以根據需求選擇不同chain
chain_options = {
    "basic": get_basic_qa_chain,
    "with_sources": get_qa_with_sources_chain,
    "custom_prompt": get_custom_prompt_qa_chain,
    "condense_prompt": get_condense_prompt_qa_chain
}

def set_openai_api_key(api_key: str):
    """Set the api key and return chain.
    If no api_key, then None is returned.
    """
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        chain = get_basic_qa_chain()
        os.environ["OPENAI_API_KEY"] = ""
        return chain


class ChatWrapper:

    def __init__(self):
        self.lock = Lock()

    def __call__(
        self, api_key: str, inp: str, history: Optional[Tuple[str, str]], chain
    ):
        """Execute the chat functionality."""
        self.lock.acquire()
        try:
            history = history or []
            # If chain is None, that is because no API key was provided.
            if chain is None:
                history.append((inp, "Please paste your OpenAI key to use"))
                return history, history

            # Set OpenAI key
            import openai
            openai.api_key = "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"
            # Run chain and append input.
            output = chain({"question": inp})["answer"]
            history.append((inp, output))
            print(history)
            print(output)
        except Exception as e:
            raise e
        finally:
            self.lock.release()
        return history, history

# test
# 創建 ChatWrapper 實例
# chatbot = ChatWrapper()
#
# # 提供輸入和聊天機器人的處理鏈（chain）
# api_key = "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"
# inp = "噴嘴阻塞"
# history = []  # 可以是之前的對話歷史
# selected_chain = get_basic_qa_chain()  # 可以是聊天機器人的處理鏈
#
# # # 調用 ChatWrapper 函數，獲得更新後的對話歷史和回應
# new_history, response = chatbot(api_key=api_key, inp=inp, history=history, chain=selected_chain)
# # print(type(response))
# # print(response[0][1])
#
# inp = "多久保養一次"
# new_history, response = chatbot(api_key=api_key, inp=inp, history=new_history, chain=selected_chain)