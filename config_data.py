md5_path = "./md5.text"

#Chroma
collection_name="rag"
persist_directory="./chorma_db"

#spilter

chunk_size = 1000
chunk_overlap = 100
separators = ["\n\n","\n",",","，",".","。","?","？"]

max_spilt_char_number = 1000#文本分割阈值

similarity_threshold = 1#检索相关文本的数量
embedding_model_name = "text-embedding-v4"
chat_model_name = "qwen3-max"

session_config = {
        "configurable":{
            "session_id":"user_001",
        }
    }
