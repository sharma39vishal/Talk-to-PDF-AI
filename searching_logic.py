from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import os


class PDFQuestionAnswerSystem:
    def __init__(self):
        load_dotenv()
        self.embeddings = OpenAIEmbeddings()
        self.chain = load_qa_chain(OpenAI(), chain_type="stuff")
        self.document_search = None

    def _extract_text_from_pdf(self, pdf_file):
        raw_text = ''
        pdf_reader = PdfReader(pdf_file)
        for page in pdf_reader.pages:
            content = page.extract_text()
            if content:
                raw_text += content
        return raw_text

    def _split_text(self, raw_text):
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=800,
            chunk_overlap=200,
            length_function=len,
        )
        return text_splitter.split_text(raw_text)

    def _index_documents(self, texts):
        self.document_search = FAISS.from_texts(texts, self.embeddings)

    def load_documents_from_pdfs(self, pdf_files):
        all_texts = []
        for pdf_file in pdf_files:
            raw_text = self._extract_text_from_pdf(pdf_file)
            texts = self._split_text(raw_text)
            all_texts.extend(texts)
        self._index_documents(all_texts)

    def ask_question(self, question):
        if self.document_search is None:
            raise ValueError("No documents loaded. Please load documents first.")
        docs = self.document_search.similarity_search(question)
        return self.chain.run(input_documents=docs, question=question)

