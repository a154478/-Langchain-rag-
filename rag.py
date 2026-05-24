from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from file_history_store import get_history
from vector_store import VectorStoreService
from langchain_community.embeddings import DashScopeEmbeddings
import config_data as config
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory, RunnableLambda


def print_prompt(prompt):
    print("="*20)
    print(prompt.to_string())
    print("="*20)
    return prompt


class RagService(object):
    def __init__(self):
        self.vector_service=VectorStoreService(
            embedding=DashScopeEmbeddings(model=config.embedding_model_name)
        )
        self.prompt_template = ChatPromptTemplate.from_messages(
            [("system","以我提供的参考资料为主"
             "简洁专业的回答用户的问题{context}。"),
             ("system","并且我提供如下的用户对话记录"),
             MessagesPlaceholder("history"),
            ("user","请回答用户问题{input}")
             ]
        )
        self.chat_model =ChatTongyi(model=config.chat_model_name,streaming = True)
        self.chain = self._get_chain()
    def _get_chain(self):
        retriever = self.vector_service.get_retriever()
        def format_document(docs:list[Document]):
           if not docs:
               return "无参考资料。"
           formatted_str=""
           for doc in docs:
               formatted_str+=f"文档片段{doc.page_content}\n,文档元数据：{doc.metadata}\n\n"
           return formatted_str
        def temp1(value):
            return value["input"]
        def temp2(value):

            new_value={}
            new_value["input"]=value["input"]["input"]
            new_value["context"]=value["context"]
            new_value["history"]=value["input"]["histroy"]
            return new_value
        chain=(
             {
            "input":RunnablePassthrough(),
            "context":RunnableLambda(temp1)|retriever|format_document
             }|RunnableLambda(temp2)|self.prompt_template|print_prompt|self.chat_model|StrOutputParser()
       )
        conversation_chain=RunnableWithMessageHistory(
            chain,
            get_history,
            input_messages_key="input",
            history_messages_key="histroy"
        )
        return conversation_chain
if __name__ == '__main__':
    #session_id配置
    session_config = {
        "configurable":{
            "session_id":"user_001",
        }
    }
