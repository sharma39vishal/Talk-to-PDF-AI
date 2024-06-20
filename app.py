from flask import Flask, request, jsonify, render_template
from searching_logic import PDFQuestionAnswerSystem
from flask_cors import CORS
import os
app = Flask(__name__,template_folder=os.path.join('build'), static_folder='build', static_url_path='/')
CORS(app)

@app.route('/ask', methods=['POST'])
def ask_question():
  data = request.get_json()
  question = data.get('question', '')
  pdf_qa_system = PDFQuestionAnswerSystem()
  pdf_files = [ "DSA.pdf"]  # Add more PDF files as needed
  pdf_qa_system.load_documents_from_pdfs(pdf_files)

  answer = pdf_qa_system.ask_question(question)
  return jsonify({"answer": answer})

@app.route('/')
def index():
  return render_template('index.html')  # Ensure 'index.html' is in the 'templates' folder

if __name__ == '__main__':
  app.run(debug=True, host='0.0.0.0')
