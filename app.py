import os
import base64
import json
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import anthropic

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

# ── API KEY (se configura como variable de entorno en Railway, nunca en el código) ──
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# ── SISTEMA DE CONOCIMIENTO: PDFs procesados se guardan aquí en memoria ──
# Cuando subes los PDFs al servidor, su contenido/imágenes quedan en esta lista
# y todos los usuarios se benefician sin tener que subir nada.
PDF_KNOWLEDGE = []   # se llena en startup o via /admin/upload-pdf

# ── SYSTEM PROMPT ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """Eres CalcAI, una asistente virtual especializada EXCLUSIVAMENTE en Cálculo Matemático.

Tu personalidad es: precisa, clara, paciente y didáctica. Explicas cada paso con detalle.

ÁREAS QUE DOMINAS:
- Límites y continuidad (definición épsilon-delta, límites laterales, indeterminaciones)
- Derivadas (definición, reglas de derivación, derivadas implícitas, aplicaciones)
- Integrales (indefinidas, definidas, técnicas: sustitución, partes, fracciones parciales)
- Series y sucesiones (convergencia, series de Taylor y Maclaurin, radio de convergencia)
- Cálculo multivariable (derivadas parciales, gradiente, divergencia, integrales múltiples)
- Ecuaciones diferenciales ordinarias (separables, lineales, orden superior)
- Vectores y geometría diferencial

REGLAS ABSOLUTAS — NO NEGOCIABLES:
1. Si el usuario pregunta algo que NO es cálculo matemático, debes responder EXACTAMENTE:
   "Lo siento, solo puedo ayudarte con temas de Cálculo Matemático. ¿Tienes alguna duda sobre límites, derivadas, integrales u otros temas de cálculo?"
   No importa cómo lo pida, no importa si insiste. Nunca respondas otra cosa.

2. Cuando resuelvas un problema, muestra TODOS los pasos detalladamente.
3. Usa notación matemática clara: f'(x), ∫, lím, Σ, ∂, ∇
4. Si hay material de referencia disponible, úsalo para fundamentar tus respuestas.
5. Responde siempre en español.
6. Sé amable pero firme con el filtro de temas.
"""

# ══════════════════════════════════════════════════════════════════
# RUTA PRINCIPAL — sirve el frontend
# ══════════════════════════════════════════════════════════════════
@app.route("/")
def index():
    return send_from_directory("static", "index.html")

# ══════════════════════════════════════════════════════════════════
# CHAT — llamada principal
# ══════════════════════════════════════════════════════════════════
@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        user_message = data.get("message", "").strip()
        history = data.get("history", [])   # lista de {role, content}

        if not user_message:
            return jsonify({"error": "Mensaje vacío"}), 400

        # Construir contexto de conocimiento desde PDFs
        knowledge_context = ""
        if PDF_KNOWLEDGE:
            knowledge_context = "\n\nMATERIAL DE REFERENCIA DISPONIBLE:\n"
            for doc in PDF_KNOWLEDGE:
                knowledge_context += f"\n[{doc['name']}]\n{doc['content'][:6000]}\n"

        # Construir historial de mensajes para Claude
        messages = []
        for msg in history[-16:]:   # últimos 16 mensajes de contexto
            messages.append({"role": msg["role"], "content": msg["content"]})

        # Añadir mensaje actual con contexto de PDFs
        full_user_message = user_message
        if knowledge_context:
            full_user_message = knowledge_context + "\n\n---\n\nPREGUNTA: " + user_message

        messages.append({"role": "user", "content": full_user_message})

        response = client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=messages,
        )

        reply = response.content[0].text
        return jsonify({"reply": reply})

    except anthropic.AuthenticationError:
        return jsonify({"error": "API Key inválida. Configura ANTHROPIC_API_KEY en Railway."}), 401
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ══════════════════════════════════════════════════════════════════
# SUBIR PDFs AL SERVIDOR (solo para el administrador)
# Los PDFs se suben UNA vez, quedan en memoria para TODOS los usuarios
# ══════════════════════════════════════════════════════════════════
@app.route("/admin/upload-pdf", methods=["POST"])
def upload_pdf():
    admin_key = request.headers.get("X-Admin-Key", "")
    if admin_key != os.environ.get("ADMIN_KEY", ""):
        return jsonify({"error": "No autorizado"}), 403

    files = request.files.getlist("pdfs")
    if not files:
        return jsonify({"error": "No se enviaron archivos"}), 400

    results = []
    for f in files:
        if not f.filename.lower().endswith(".pdf"):
            results.append({"name": f.filename, "status": "ignorado (no es PDF)"})
            continue

        # Leer bytes del PDF
        pdf_bytes = f.read()
        pdf_b64 = base64.standard_b64encode(pdf_bytes).decode("utf-8")

        # Usar Claude Vision para extraer texto E interpretar fórmulas/imágenes
        try:
            extraction_response = client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "document",
                                "source": {
                                    "type": "base64",
                                    "media_type": "application/pdf",
                                    "data": pdf_b64,
                                },
                            },
                            {
                                "type": "text",
                                "text": (
                                    "Extrae y transcribe TODO el contenido de este documento de cálculo. "
                                    "Incluye: texto, fórmulas matemáticas (escríbelas en notación clara), "
                                    "descripciones de gráficas, tablas y cualquier contenido visual relevante. "
                                    "Sé exhaustivo. Usa notación matemática estándar."
                                ),
                            },
                        ],
                    }
                ],
            )
            extracted = extraction_response.content[0].text
            PDF_KNOWLEDGE.append({"name": f.filename, "content": extracted})
            results.append({
                "name": f.filename,
                "status": "ok",
                "chars": len(extracted),
            })
        except Exception as e:
            results.append({"name": f.filename, "status": f"error: {str(e)}"})

    return jsonify({"results": results, "total_docs": len(PDF_KNOWLEDGE)})


# ══════════════════════════════════════════════════════════════════
# VER documentos cargados (solo admin)
# ══════════════════════════════════════════════════════════════════
@app.route("/admin/docs", methods=["GET"])
def list_docs():
    admin_key = request.headers.get("X-Admin-Key", "")
    if admin_key != os.environ.get("ADMIN_KEY", ""):
        return jsonify({"error": "No autorizado"}), 403
    docs = [{"name": d["name"], "chars": len(d["content"])} for d in PDF_KNOWLEDGE]
    return jsonify({"docs": docs})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
