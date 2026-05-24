"""知识库"""
import os
import config_data as config
import hashlib
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime


def check_md5(md5_str:str):

    #j检查是不是处理了
    if not os.path.exists(config.md5_path):
        open(config.md5_path,'w',encoding='utf-8').close()
        return False#进入表示文件不存在
    else:
        for line in open(config.md5_path,'r',encoding='utf-8').readlines():
            line = line.strip()
            if line == md5_str:
                return True
        return False
def save_md5(md5_str:str):
    with open(config.md5_path,'a',encoding='utf-8') as f:
        f.write(md5_str+'\n')
def get_string_md5(input_str:str,encoding='utf-8'):
    #将字符串转换为byte数组
    str_bytes=input_str.encode(encoding=encoding)
    md5_obj = hashlib.md5()
    md5_obj.update(str_bytes)
    md5_hex = md5_obj.hexdigest()
    return md5_hex
class KnowledgeBaseService(object):
    def __init__(self):
        os.makedirs(config.persist_directory,exist_ok=True)#不存在就创建，存在就跳过
        self.chroma = Chroma(
            collection_name=config.collection_name,
            embedding_function = DashScopeEmbeddings(model ="text-embedding-v4"),
            persist_directory = config.persist_directory,
        )
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,#分割后文本段最大长度
            chunk_overlap=config.chunk_overlap,#最大重叠字段
            separators=config.separators,#分割符号
            length_function=len,

        )
    def upload_by_str(self,data,filename):
        md5_hex = get_string_md5(data)
        if check_md5(md5_hex):
            return "[跳过]内容已经存在知识库中"
        if len(data) > config.max_spilt_char_number:
            knowledge_chunks:list[str]= self.spliter.split_text(data)
        else:
            knowledge_chunks = [data]
        metadata = {
            "source":filename,
            "create_time":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "operator":"小王",
        }
        self.chroma.add_texts(#加载到数据库
            knowledge_chunks,
            metadatas=[metadata for _ in knowledge_chunks],
        )
        save_md5(md5_hex)
        return "[成功]内容已经载入向量库"
if __name__ == '__main__':
    service = KnowledgeBaseService()
    r= service.upload_by_str("周杰伦","testfile")
    print(r)

