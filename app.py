from flask import Flask, request, jsonify, render_template
import boto3
import json
import os
import PyPDF2
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# AWS Clients
bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name=os.getenv("AWS_REGION", "us-east-1")
)

GUARDRAIL_ID = "k191z5hjoqwb"
GUARDRAIL_VERSION = "DRAFT"

s3 = boto3.client(
    service_name="s3",
    region_name=os.getenv("AWS_REGION", "us-east-1")
)

BUCKET = os.getenv("S3_BUCKET_NAME")
UPLOAD_FOLDER = "uploads"

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
        response = bedrock.invoke_model(
            modelId="meta.llama3-8b-instruct-v1:0",
            body=body,
            contentType="application/json",
            accept="application/json",
            guardrailIdentifier=GUARDRAIL_ID,
            guardrailVersion=GUARDRAIL_VERSION,
        )

        response_body = json.loads(response["body"].read())

        # Check if guardrail blocked it
        if response_body.get("amazon-bedrock-guardrailAction") == "BLOCKED":
            return "🛡️ Guardrail blocked this request — harmful content detected!"

        # Check generation field
        answer = response_body.get("generation", "")

        if not answer or answer.strip() == "":
            return "🛡️ This response was blocked by Bedrock Guardrails for safety reasons!"

        return answer.strip()

    except bedrock.exceptions.ClientError as e:
        error_code = e.response["Error"]["Code"]
        if error_code == "ValidationException":
            return "🛡️ Guardrail blocked this request — content policy violation!"
        raise e

    

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

    # Save file locally
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # Extract text
    try:
        if file.filename.endswith(".pdf"):
            text = extract_text_from_pdf(filepath)
        elif file.filename.endswith(".txt"):
            text = extract_text_from_txt(filepath)
        else:
            return jsonify({"error": "Only PDF and TXT files supported"}), 400
    except Exception as e:
        return jsonify({"error": f"Could not read file: {str(e)}"}), 500

    # Upload to S3
    try:
        s3_path = upload_to_s3(filepath, file.filename)
    except Exception as e:
        s3_path = "S3 upload failed — using local file"

    return jsonify({
        "message": f"✅ Document uploaded successfully!",
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

    if not question:
        return jsonify({"error": "No question provided"}), 400
    if not filename:
        return jsonify({"error": "No document selected"}), 400

    # Load document text
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(filepath):
        return jsonify({"error": "Document not found"}), 404

    try:
        if filename.endswith(".pdf"):
            context = extract_text_from_pdf(filepath)
        else:
            context = extract_text_from_txt(filepath)
    except Exception as e:
        return jsonify({"error": f"Could not read file: {str(e)}"}), 500

    # Ask Bedrock
    try:
        answer = ask_bedrock(question, context)
        return jsonify({
            "question": question,
            "answer": answer,
            "document": filename,
            "model": "Llama 3 via AWS Bedrock"
        })
    except Exception as e:
        return jsonify({"error": f"AI error: {str(e)}"}), 500

@app.route("/documents", methods=["GET"])
def list_documents():
    files = os.listdir(UPLOAD_FOLDER)
    docs = [f for f in files if f.endswith((".pdf", ".txt"))]
    return jsonify({
        "total": len(docs),
        "documents": docs
    })

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