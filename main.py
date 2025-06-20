import os
import json
from haystack.dataclasses import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.generators.hugging_face_local import HuggingFaceLocalGenerator
from haystack.utils import ComponentDevice
import PyPDF2

class RAGSystem:
    def __init__(self):
        print("[1] 初始化文档存储...")
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

    def _initialize_components(self):
        self.doc_embedder.warm_up()
        self.query_embedder.warm_up()
        self.generator.warm_up()

    def load_or_create_index(self):
        if os.path.exists("indexed_docs.json"):
            print("[2] 已加载本地索引文档 indexed_docs.json")
            with open("indexed_docs.json", "r", encoding="utf-8") as f:
                loaded_docs = [Document.from_dict(d) for d in json.load(f)]
            self.store.write_documents(loaded_docs)
        else:
            print("[2] 首次构建 embedding 并保存索引 indexed_docs.json")
            docs = self._load_documents()
            docs_with_emb = self.doc_embedder.run(documents=docs)["documents"]
            self.store.write_documents(docs_with_emb)
            with open("indexed_docs.json", "w", encoding="utf-8") as f:
                json.dump([doc.to_dict(flatten=False) for doc in docs_with_emb], f, ensure_ascii=False, indent=2)
            print("[2] 索引已保存到 indexed_docs.json")

    def _load_documents(self):
        docs = []
        for filename in os.listdir('.'):
            if filename.endswith('.txt'):
                with open(filename, "r", encoding="utf-8") as f:
                    docs.append(Document(content=f.read(), meta={"source": filename}))
            elif filename.endswith('.pdf'):
                pdf_reader = PyPDF2.PdfReader(filename)
                pdf_content = "".join([page.extract_text() or "" for page in pdf_reader.pages])
                docs.append(Document(content=pdf_content, meta={"source": filename}))
        return docs

    def answer_query(self, query):
        print(f"[6] 用户问题：{query}")
        query_emb = self.query_embedder.run(text=query)["embedding"]
        retrieved_docs = self.retriever.run(query_embedding=query_emb)["documents"]
        context = "\n".join([doc.content for doc in retrieved_docs])
        prompt = f"<|im_start|>system\n你是一个乐于助人的助手。<|im_end|>\n<|im_start|>user\n已知信息：\n{context}\n请回答：{query}<|im_end|>\n<|im_start|>assistant\n"
        print(prompt)
        print("[6] 已生成 prompt，开始生成答案...")
        result = self.generator.run(prompt=prompt, generation_kwargs={"max_new_tokens": 128})
        print("[6] 答案：", result["replies"][0] if result["replies"] else "无答案")

# 使用示例
rag_system = RAGSystem()
rag_system.load_or_create_index()
rag_system.answer_query("文档的内容是什么？")

"""
git init
git add 
git commit -m "first commit"
git branch -M main
git remote add origin git@github.com:skl519/rag-llm-share.git
git push -u origin main

"""