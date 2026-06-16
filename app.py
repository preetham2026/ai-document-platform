from flask import Flask, request, jsonify, render_template
import boto3
import json
import os
import PyPDF2
import traceback
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# AWS Credentials
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
BUCKET = os.getenv("S3_BUCKET_NAME")
UPLOAD_FOLDER = "uploads"

# Guardrail config
GUARDRAIL_ID = "k191z5hjoqwb"
GUARDRAIL_VERSION = "DRAFT"

print("=== STARTUP CHECK ===")
print("AWS_REGION:", AWS_REGION)
print("AWS_ACCESS_KEY:", "FOUND" if AWS_ACCESS_KEY else "NOT FOUND")
print("AWS_SECRET_KEY:", "FOUND" if AWS_SECRET_KEY else "NOT FOUND")
print("S3_BUCKET:", BUCKET)
print("=====================")

# AWS Clients
bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY
)

s3_client = boto3.client(
    service_name="s3",
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY
)

print("✅ AWS clients initialized!")

# ================================================
#              HELPER FUNCTIONS
# ================================================

def extract_text_from_pdf(filepath):
    text = ""
    with open(filepath, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text()
    return text

def extract_text_from_txt(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()

def ask_bedrock(question, context):
    print(f"Asking Bedrock: {question[:50]}...")

    prompt = f"""You are an intelligent document assistant.
Use ONLY the following document content to answer the question.
If the answer is not in the document, say "I could not find that information in the document."

Document Content:
{context[:3000]}

Question: {question}

Answer:"""

    body = json.dumps({
        "prompt": prompt,
        "max_gen_len": 512,
        "temperature": 0.3,
    })

    try:
        print("Calling Bedrock API...")
        response = bedrock_client.invoke_model(
            modelId="meta.llama3-8b-instruct-v1:0",
            body=body,
            contentType="application/json",
            accept="application/json",
            guardrailIdentifier=GUARDRAIL_ID,
            guardrailVersion=GUARDRAIL_VERSION,
        )
        print("Bedrock response received!")

        response_body = json.loads(response["body"].read())
        print("Response body:", str(response_body)[:200])

        answer = response_body.get("generation", "")

        if not answer or answer.strip() == "":
            return "🛡️ This response was blocked by Bedrock Guardrails!"

        return answer.strip()

    except Exception as e:
        print("BEDROCK ERROR:", traceback.format_exc())
        raise e

def upload_to_s3(filepath, filename):
    try:
        s3_client.upload_file(filepath, BUCKET, f"documents/{filename}")
        return f"s3://{BUCKET}/documents/{filename}"
    except Exception as e:
        print("S3 ERROR:", str(e))
        return f"S3 upload failed: {str(e)}"

# ================================================
#                   ROUTES
# ================================================

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_document():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    print(f"File saved: {filepath}")

    try:
        if file.filename.endswith(".pdf"):
            text = extract_text_from_pdf(filepath)
        elif file.filename.endswith(".txt"):
            text = extract_text_from_txt(filepath)
        else:
            return jsonify({"error": "Only PDF and TXT files supported"}), 400
    except Exception as e:
        print("FILE READ ERROR:", traceback.format_exc())
        return jsonify({"error": f"Could not read file: {str(e)}"}), 500

    s3_path = upload_to_s3(filepath, file.filename)

    return jsonify({
        "message": "✅ Document uploaded successfully!",
        "filename": file.filename,
        "s3_path": s3_path,
        "text_preview": text[:200] + "...",
        "total_chars": len(text)
    })

@app.route("/ask", methods=["POST"])
def ask_question():
    data = request.get_json()
    question = data.get("question", "")
    filename = data.get("filename", "")

    print(f"Question: {question}")
    print(f"Filename: {filename}")

    if not question:
        return jsonify({"error": "No question provided"}), 400
    if not filename:
        return jsonify({"error": "No document selected"}), 400

    filepath = os.path.join(UPLOAD_FOLDER, filename)
    print(f"Looking for file: {filepath}")
    print(f"File exists: {os.path.exists(filepath)}")

    if not os.path.exists(filepath):
        return jsonify({"error": f"Document not found: {filepath}"}), 404

    try:
        if filename.endswith(".pdf"):
            context = extract_text_from_pdf(filepath)
        else:
            context = extract_text_from_txt(filepath)
        print(f"Context length: {len(context)}")
    except Exception as e:
        print("CONTEXT ERROR:", traceback.format_exc())
        return jsonify({"error": f"Could not read file: {str(e)}"}), 500

    try:
        answer = ask_bedrock(question, context)
        return jsonify({
            "question": question,
            "answer": answer,
            "document": filename,
            "model": "Llama 3 via AWS Bedrock"
        })
    except Exception as e:
        print("FULL ERROR:", traceback.format_exc())
        return jsonify({"error": f"AI error: {str(e)}"}), 500

@app.route("/documents", methods=["GET"])
def list_documents():
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    files = os.listdir(UPLOAD_FOLDER)
    docs = [f for f in files if f.endswith((".pdf", ".txt"))]
    return jsonify({"total": len(docs), "documents": docs})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "service": "AI Document Intelligence Platform",
        "model": "Llama 3 via AWS Bedrock",
        "storage": f"S3 bucket: {BUCKET}"
    })

# ================================================
#                    RUN
# ================================================
if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    print("🚀 AI Document Intelligence Platform Starting...")
    print("📡 Open browser: http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)