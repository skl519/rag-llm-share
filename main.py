import os
import json
import logging
from haystack.dataclasses import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.generators.hugging_face_local import HuggingFaceLocalGenerator
from haystack.utils import ComponentDevice
import PyPDF2

logging.basicConfig(level=logging.INFO)

class RAGSystem:
    def __init__(self):
        logging.info("初始化文档存储...")
        self.store = InMemoryDocumentStore()
        self.doc_embedder = SentenceTransformersDocumentEmbedder(
            model="models-weight/all-mpnet-base-v2",
            device=ComponentDevice.from_str("cuda")
        )
        self.query_embedder = SentenceTransformersTextEmbedder(
            model="models-weight/all-mpnet-base-v2",
            device=ComponentDevice.from_str("cuda")
        )
        self.retriever = InMemoryEmbeddingRetriever(document_store=self.store, top_k=5)
        self.generator = HuggingFaceLocalGenerator(
            model="models-weight/Qwen2.5-0.5B-Instruct",
            task="text-generation",
            device=ComponentDevice.from_str("cuda")
        )
        self._initialize_components()
        logging.info("文档存储初始化完成。")

    def _initialize_components(self):
        self.doc_embedder.warm_up()
        logging.info("文档嵌入器已准备就绪。")
        self.query_embedder.warm_up()
        logging.info("查询嵌入器已准备就绪。")
        self.generator.warm_up()
        logging.info("生成器已准备就绪。")

    def load_or_create_index(self):
        if os.path.exists("indexed_docs_with_embeddings.json"):
            logging.info("已加载本地索引文档 indexed_docs_with_embeddings.json")
            with open("indexed_docs_with_embeddings.json", "r", encoding="utf-8") as f:
                loaded_docs = [Document.from_dict(d) for d in json.load(f)]
            self.store.write_documents(loaded_docs)
            logging.info("文档已写入存储。")
        else:
            logging.info("首次构建 embedding 并保存索引 indexed_docs_with_embeddings.json")
            docs = self._load_documents()
            logging.info("文档加载完成，开始生成嵌入向量...")
            docs_with_emb = self.doc_embedder.run(documents=docs)["documents"]
            self.store.write_documents(docs_with_emb)
            logging.info("嵌入向量生成并写入存储。")
            with open("indexed_docs_with_embeddings.json", "w", encoding="utf-8") as f:
                json.dump([doc.to_dict(flatten=False) for doc in docs_with_emb], f, ensure_ascii=False, indent=2)
            logging.info("索引已保存到 indexed_docs_with_embeddings.json")

    def _load_documents(self):
        logging.info("开始加载文档...")
        docs = []
        for filename in os.listdir('.'):
            if filename.endswith('.txt'):
                with open(filename, "r", encoding="utf-8") as f:
                    docs.append(Document(content=f.read(), meta={"source": filename}))
            elif filename.endswith('.pdf'):
                pdf_reader = PyPDF2.PdfReader(filename)
                pdf_content = "".join([page.extract_text() or "" for page in pdf_reader.pages])
                docs.append(Document(content=pdf_content, meta={"source": filename}))
        logging.info("文档加载完成。")
        return docs

    def answer_query(self, query):
        logging.info(f"用户问题：{query}")
        query_emb = self.query_embedder.run(text=query)["embedding"]
        logging.info("查询嵌入向量生成完成。")
        retrieved_docs = self.retriever.run(query_embedding=query_emb)["documents"]
        logging.info("文档检索完成。")
        context = "\n".join([doc.content for doc in retrieved_docs])
        prompt = f"<|im_start|>system\n你是一个乐于助人的助手。<|im_end|>\n<|im_start|>user\n已知信息：\n{context}\n请回答：{query}<|im_end|>\n<|im_start|>assistant\n"
        logging.info("已生成 prompt，开始生成答案...")
        result = self.generator.run(prompt=prompt, generation_kwargs={"max_new_tokens": 128})
        logging.info("答案：%s", result["replies"][0] if result["replies"] else "无答案")

# 使用示例
rag_system = RAGSystem()
rag_system.load_or_create_index()
rag_system.answer_query("文档的内容是什么？")



"""
git init

git branch -M main
git remote add origin git@github.com:skl519/rag-llm-share.git
git add main.py
git commit -m "first commit"
git push -u origin main

pip install haystack-ai
pip install torch==2.5.1 torchvision==0.20.1
modelscope download --model Qwen/Qwen2.5-0.5B-Instruct --local_dir ./models-weight/Qwen2.5-0.5B-Instruct
"""