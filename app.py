from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import os
import json
import time
import re
import threading
import asyncio
from datetime import datetime
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema import Document
from dotenv import load_dotenv

app = Flask(__name__)
CORS(app)

# Variáveis globais para armazenar o estado do chatbot
chatbot = None
db = None
initialization_error = None
conversation_stats = {
    'total_conversations': 0,
    'total_documents': 0,
    'participants': [],
    'is_loaded': False,
    'last_updated': None,
    'error_message': None
}

# ============== CONFIGURAÇÃO DA API KEY ==============

load_dotenv()

def configure_api_key():
    """Verifica se a chave de API está configurada."""
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if api_key:
        print("✅ Chave de API do Google configurada.")
        return True
    else:
        print("❌ Configure GOOGLE_API_KEY no arquivo .env")
        return False

def preprocess_whatsapp_data(json_path):
    """
    Pré-processa os dados do WhatsApp para melhor estruturação.
    """
    print(f"Carregando e pré-processando dados de '{json_path}'...")
    
    try:
        with open(json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        documents = []
        participants = set()
        
        # Processamento simplificado para evitar problemas de memória
        conversation_window = []
        conversation_id = 0
        
        for i, item in enumerate(data):
            author = item.get('author', 'Desconhecido')
            message = item.get('message', '').strip()
            timestamp = item.get('timestamp', '')
            
            if author != 'Desconhecido':
                participants.add(author)
            
            if len(message) < 3:
                continue
            
            # Remove mensagens de sistema
            system_messages = [
                'mensagem apagada', 'media omitted', 'audio omitted', 
                'image omitted', 'document omitted', 'video omitted'
            ]
            if any(sys_msg in message.lower() for sys_msg in system_messages):
                continue
            
            conversation_window.append({
                'author': author,
                'message': message,
                'timestamp': timestamp,
                'index': i
            })
            
            # Cria documento a cada 10 mensagens para evitar documentos muito grandes
            if len(conversation_window) >= 10:
                conversation_text = create_conversation_context(conversation_window, conversation_id)
                participants_list = list(set([msg['author'] for msg in conversation_window]))
                participants_str = ', '.join(participants_list)
                
                documents.append(Document(
                    page_content=conversation_text,
                    metadata={
                        'conversation_id': str(conversation_id),
                        'start_time': str(conversation_window[0]['timestamp']),
                        'end_time': str(conversation_window[-1]['timestamp']),
                        'message_count': len(conversation_window),
                        'participants': participants_str
                    }
                ))
                conversation_id += 1
                conversation_window = []
        
        # Adiciona últimas mensagens se sobraram
        if len(conversation_window) >= 3:
            conversation_text = create_conversation_context(conversation_window, conversation_id)
            participants_list = list(set([msg['author'] for msg in conversation_window]))
            participants_str = ', '.join(participants_list)
            
            documents.append(Document(
                page_content=conversation_text,
                metadata={
                    'conversation_id': str(conversation_id),
                    'start_time': str(conversation_window[0]['timestamp']),
                    'end_time': str(conversation_window[-1]['timestamp']),
                    'message_count': len(conversation_window),
                    'participants': participants_str
                }
            ))
        
        print(f"Dados processados: {len(documents)} conversas contextualizadas criadas.")
        return documents, list(participants)
        
    except Exception as e:
        print(f"Erro no pré-processamento: {e}")
        raise e

def create_conversation_context(messages, conv_id):
    """Cria um contexto rico para uma conversa."""
    participants = list(set([msg['author'] for msg in messages]))
    start_time = messages[0]['timestamp']
    
    context_lines = [
        f"=== CONVERSA #{conv_id} ===",
        f"Participantes: {', '.join(participants)}",
        f"Início: {start_time}",
        f"Número de mensagens: {len(messages)}",
        ""
    ]
    
    for msg in messages:
        author = msg['author']
        message = msg['message']
        timestamp = msg['timestamp']
        
        cleaned_message = clean_message(message)
        if cleaned_message:
            context_lines.append(f"[{timestamp}] {author}: {cleaned_message}")
    
    return "\n".join(context_lines)

def clean_message(message):
    """Limpa e normaliza mensagens."""
    message = re.sub(r'[^\w\s\.,!?;:()\-áàâãéèêíïóôõöúçñ]', ' ', message)
    message = re.sub(r'\s+', ' ', message)
    return message.strip()

def format_response(raw_response):
    """Formata a resposta da IA para melhor apresentação no chat."""
    if not raw_response:
        return "Não encontrei informações sobre isso nas conversas."
    
    response = raw_response.replace('*', '').replace('**', '')
    
    technical_phrases = [
        "Baseado nas conversas analisadas,",
        "De acordo com o contexto fornecido,",
        "Analisando as informações disponíveis,",
        "Com base nos dados apresentados,",
        "Segundo as conversas examinadas,"
    ]
    
    for phrase in technical_phrases:
        response = response.replace(phrase, "")
    
    response = re.sub(r'\s+', ' ', response).strip()
    
    if response and response[0].islower():
        response = response[0].upper() + response[1:]
    
    sentences = re.split(r'(?<=\.)\s+(?=[A-Z])', response)
    
    paragraphs = []
    current_paragraph = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        current_paragraph.append(sentence)
        
        if len(current_paragraph) >= 2 and (
            sentence.endswith('.') or sentence.endswith('!') or sentence.endswith('?')
        ):
            paragraphs.append(' '.join(current_paragraph))
            current_paragraph = []
    
    if current_paragraph:
        paragraphs.append(' '.join(current_paragraph))
    
    formatted_response = '\n\n'.join(paragraphs)
    formatted_response = re.sub(r'\n{3,}', '\n\n', formatted_response)
    
    return formatted_response.strip()

def create_vector_database_with_logging(documents, embedding_model, db_path="./chroma_db"):
    """Cria e armazena os embeddings em um banco de dados vetorial Chroma."""
    print("Iniciando a criação do banco de dados vetorial...")
    
    total_docs = len(documents)
    batch_size = 5  # Reduzido para evitar problemas de memória no Render
    start_time = time.time()
    
    print(f"Processando lote 1 de {(total_docs // batch_size) + 1}...")
    vector_db = Chroma.from_documents(
        documents=documents[:batch_size],
        embedding=embedding_model,
        persist_directory=db_path
    )
    
    for i in range(batch_size, total_docs, batch_size):
        progress = (i / total_docs) * 100
        print(f"Processando lote {(i // batch_size) + 1}... ({progress:.2f}% concluído)")
        try:
            vector_db.add_documents(documents[i:i+batch_size])
        except Exception as e:
            print(f"Erro ao adicionar lote {i//batch_size + 1}: {e}")
            # Continua com o próximo lote
    
    total_time = time.time() - start_time
    print(f"Banco de dados vetorial criado com sucesso em {total_time:.2f} segundos.")
    return vector_db

def create_qa_chain_compatible(vector_db):
    """Cria a cadeia de Pergunta e Resposta (QA Chain)."""
    print("Configurando o chatbot...")
    
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.3,
            max_output_tokens=800
        )
    except Exception as e:
        print(f"Erro ao configurar LLM: {e}")
        # Fallback para modelo mais simples
        llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            temperature=0.3,
            max_output_tokens=800
        )
    
    prompt_template = """Você é um assistente que analisa conversas de relacionamento de forma carinhosa e natural.

Responda de forma conversacional, como se fosse um amigo próximo comentando sobre as conversas. Seja direto e natural, sem usar linguagem técnica ou formatação especial.

IMPORTANTE:
- Responda de forma natural e fluida
- Não use asteriscos, formatação especial ou linguagem técnica
- Se não encontrar informações, diga simplesmente que não encontrou
- Seja empático mas direto
- Organize sua resposta em parágrafos quando necessário

Conversas analisadas:
{context}

Pergunta: {question}

Resposta:"""

    QA_PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    try:
        retriever = vector_db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}  # Reduzido para melhor performance
        )
    except Exception as e:
        print(f"Erro na configuração do retriever: {e}")
        retriever = vector_db.as_retriever(search_kwargs={"k": 4})
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_PROMPT}
    )
    
    print("Chatbot pronto para conversar!")
    return qa_chain

def initialize_chatbot_sync():
    """Versão síncrona da inicialização do chatbot para o Render."""
    global chatbot, db, conversation_stats, initialization_error
    
    print("Inicializando chatbot (modo síncrono)...")
    
    try:
        if not configure_api_key():
            initialization_error = "API Key não configurada"
            conversation_stats['error_message'] = initialization_error
            return False
        
        json_file_path = "conversa_formatada.json"
        db_directory = "./chroma_db"
        
        if not os.path.exists(json_file_path):
            error_msg = f"Arquivo '{json_file_path}' não encontrado!"
            print(f"❌ ERRO: {error_msg}")
            initialization_error = error_msg
            conversation_stats['error_message'] = error_msg
            return False
        
        try:
            embedding_model = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                task_type="retrieval_document"
            )
        except Exception as e:
            print(f"Erro ao configurar embedding model: {e}")
            initialization_error = f"Erro no embedding model: {e}"
            conversation_stats['error_message'] = initialization_error
            return False
        
        # Verifica se o banco de dados já existe
        if os.path.exists(db_directory) and os.path.exists(os.path.join(db_directory, "chroma.sqlite3")):
            print("Banco de dados vetorial existente encontrado.")
            try:
                start_load_time = time.time()
                db = Chroma(persist_directory=db_directory, embedding_function=embedding_model)
                
                # Teste simples para verificar se funciona
                test_query = db.similarity_search("teste", k=1)
                load_time = time.time() - start_load_time
                print(f"Banco existente carregado com sucesso em {load_time:.2f}s.")
                
                # Atualiza estatísticas
                try:
                    doc_count = db._collection.count()
                    conversation_stats['total_documents'] = doc_count
                    conversation_stats['is_loaded'] = True
                    conversation_stats['last_updated'] = datetime.now().isoformat()
                    conversation_stats['error_message'] = None
                    print(f"✅ {doc_count} documentos carregados do banco existente")
                except Exception as e:
                    print(f"Erro ao obter contagem: {e}")
                    # Continua mesmo com erro na contagem
                    conversation_stats['is_loaded'] = True
                    conversation_stats['last_updated'] = datetime.now().isoformat()
                    conversation_stats['error_message'] = None
                    
            except Exception as e:
                print(f"❌ Erro ao carregar banco existente: {e}")
                print("Criando novo banco...")
                
                try:
                    docs, participants = preprocess_whatsapp_data(json_file_path)
                    
                    if not docs:
                        error_msg = "Nenhuma conversa foi processada."
                        print(f"ERRO: {error_msg}")
                        initialization_error = error_msg
                        conversation_stats['error_message'] = error_msg
                        return False
                    
                    db = create_vector_database_with_logging(docs, embedding_model, db_directory)
                    conversation_stats['total_conversations'] = len(docs)
                    conversation_stats['participants'] = participants
                    conversation_stats['total_documents'] = len(docs)
                    conversation_stats['is_loaded'] = True
                    conversation_stats['last_updated'] = datetime.now().isoformat()
                    conversation_stats['error_message'] = None
                    
                except Exception as create_error:
                    error_msg = f"Erro ao criar novo banco: {create_error}"
                    print(f"❌ {error_msg}")
                    initialization_error = error_msg
                    conversation_stats['error_message'] = error_msg
                    return False
        else:
            print("Nenhum banco de dados encontrado. Criando novo...")
            try:
                docs, participants = preprocess_whatsapp_data(json_file_path)
                
                if not docs:
                    error_msg = "Nenhuma conversa foi processada."
                    print(f"ERRO: {error_msg}")
                    initialization_error = error_msg
                    conversation_stats['error_message'] = error_msg
                    return False
                
                db = create_vector_database_with_logging(docs, embedding_model, db_directory)
                conversation_stats['total_conversations'] = len(docs)
                conversation_stats['participants'] = participants
                conversation_stats['total_documents'] = len(docs)
                conversation_stats['is_loaded'] = True
                conversation_stats['last_updated'] = datetime.now().isoformat()
                conversation_stats['error_message'] = None
                
            except Exception as create_error:
                error_msg = f"Erro ao criar banco: {create_error}"
                print(f"❌ {error_msg}")
                initialization_error = error_msg
                conversation_stats['error_message'] = error_msg
                return False
        
        # Cria o chatbot
        try:
            chatbot = create_qa_chain_compatible(db)
            print("✅ Chatbot inicializado com sucesso!")
            return True
        except Exception as chatbot_error:
            error_msg = f"Erro ao criar chatbot: {chatbot_error}"
            print(f"❌ {error_msg}")
            initialization_error = error_msg
            conversation_stats['error_message'] = error_msg
            return False
        
    except Exception as e:
        error_msg = f"Erro geral durante inicialização: {e}"
        print(f"❌ {error_msg}")
        initialization_error = error_msg
        conversation_stats['error_message'] = error_msg
        return False

# ============== ROTAS DA API ==============

@app.route('/')
def index():
    """Serve a interface web diretamente."""
    return render_template_string("""
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Memórias com meu Grande Amor</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Georgia', serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #2c3e50;
            overflow-x: hidden;
        }

        .floating-hearts {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: 1;
        }

        .heart {
            position: absolute;
            font-size: 20px;
            color: rgba(255, 182, 193, 0.6);
            animation: float 6s ease-in-out infinite;
        }

        @keyframes float {
            0%, 100% { transform: translateY(0px) rotate(0deg); }
            50% { transform: translateY(-20px) rotate(180deg); }
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            position: relative;
            z-index: 2;
        }

        .header {
            text-align: center;
            margin-bottom: 30px;
            background: rgba(255, 255, 255, 0.95);
            padding: 30px;
            border-radius: 20px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
            backdrop-filter: blur(10px);
        }

        .header h1 {
            font-size: 2.5em;
            color: #8e44ad;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
        }

        .header p {
            font-size: 1.2em;
            color: #6c5ce7;
            font-style: italic;
        }

        .main-content {
            display: grid;
            grid-template-columns: 1fr 2fr;
            gap: 20px;
            margin-bottom: 20px;
        }

        .sidebar {
            background: rgba(255, 255, 255, 0.95);
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
            height: fit-content;
        }

        .sidebar h3 {
            color: #e74c3c;
            margin-bottom: 15px;
            font-size: 1.3em;
            border-bottom: 2px solid #f8d7da;
            padding-bottom: 8px;
        }

        .suggestions {
            list-style: none;
        }

        .suggestions li {
            background: linear-gradient(45deg, #ffeaa7, #fab1a0);
            margin: 8px 0;
            padding: 12px;
            border-radius: 10px;
            cursor: pointer;
            transition: all 0.3s ease;
            font-size: 0.9em;
            border: 1px solid transparent;
        }

        .suggestions li:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
            border-color: #e84393;
        }

        .chat-section {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
            display: flex;
            flex-direction: column;
            height: 600px;
        }

        .chat-header {
            background: linear-gradient(45deg, #fd79a8, #fdcb6e);
            color: white;
            padding: 20px;
            border-radius: 15px 15px 0 0;
            text-align: center;
            font-size: 1.1em;
            font-weight: bold;
        }

        .chat-messages {
            flex: 1;
            padding: 20px;
            overflow-y: auto;
            background-color: #f8f9fa;
        }

        .message {
            margin-bottom: 15px;
            animation: fadeIn 0.5s ease-in;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .user-message {
            background: linear-gradient(45deg, #74b9ff, #0984e3);
            color: white;
            padding: 12px 18px;
            border-radius: 20px 20px 5px 20px;
            margin-left: auto;
            max-width: 80%;
            width: fit-content;
            margin-right: 0;
        }

        .bot-message {
            background: linear-gradient(45deg, #fd79a8, #e84393);
            color: white;
            padding: 15px 20px;
            border-radius: 20px 20px 20px 5px;
            max-width: 85%;
            line-height: 1.6;
            white-space: pre-wrap;
        }

        .chat-input-section {
            padding: 20px;
            background: #f1f3f4;
            border-radius: 0 0 15px 15px;
        }

        .chat-input {
            display: flex;
            gap: 10px;
        }

        .chat-input input {
            flex: 1;
            padding: 15px;
            border: 2px solid #ddd;
            border-radius: 25px;
            font-size: 16px;
            transition: border-color 0.3s;
        }

        .chat-input input:focus {
            outline: none;
            border-color: #e84393;
            box-shadow: 0 0 10px rgba(232, 67, 147, 0.3);
        }

        .send-btn {
            background: linear-gradient(45deg, #fd79a8, #e84393);
            color: white;
            border: none;
            padding: 15px 20px;
            border-radius: 50%;
            cursor: pointer;
            transition: all 0.3s ease;
            font-size: 18px;
        }

        .send-btn:hover {
            transform: scale(1.1);
            box-shadow: 0 5px 15px rgba(232, 67, 147, 0.4);
        }

        .stats-section {
            background: rgba(255, 255, 255, 0.95);
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
            margin-top: 20px;
            text-align: center;
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }

        .stat-card {
            background: linear-gradient(45deg, #a29bfe, #6c5ce7);
            color: white;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
        }

        .stat-number {
            font-size: 2em;
            font-weight: bold;
        }

        .loading {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid #f3f3f3;
            border-top: 3px solid #e84393;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .welcome-message {
            background: linear-gradient(45deg, #81ecec, #74b9ff);
            color: white;
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            margin-bottom: 15px;
            font-style: italic;
        }

        .error-message {
            background: linear-gradient(45deg, #ff6b6b, #ee5a24);
            color: white;
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            margin-bottom: 15px;
        }

        @media (max-width: 768px) {
            .main-content {
                grid-template-columns: 1fr;
            }
            
            .header h1 {
                font-size: 2em;
            }
            
            .chat-section {
                height: 500px;
            }
        }

        .chat-messages::-webkit-scrollbar {
            width: 6px;
        }

        .chat-messages::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 10px;
        }

        .chat-messages::-webkit-scrollbar-thumb {
            background: linear-gradient(45deg, #fd79a8, #e84393);
            border-radius: 10px;
        }
    </style>
</head>
<body>
    <div class="floating-hearts"></div>

    <div class="container">
        <div class="header">
            <h1><i class="fas fa-heart"></i> Memórias com Grande Amor <i class="fas fa-heart"></i></h1>
            <p>Uma pequena homenagem a tudo que já passamos juntos</p>
        </div>

        <div class="main-content">
            <div class="sidebar">
                <h3><i class="fas fa-lightbulb"></i> Perguntas Sugeridas</h3>
                <ul class="suggestions">
                    <li onclick="askQuestion(this.textContent)">Quem é urubacu?</li>
                    <li onclick="askQuestion(this.textContent)">E a Taylor Swift de Limoeiro?</li>
                    <li onclick="askQuestion(this.textContent)">Como demonstrávamos carinho um pelo outro?</li>
                    <li onclick="askQuestion(this.textContent)">Quais planos futuros fizemos juntos?</li>
                    <li onclick="askQuestion(this.textContent)">Quem começou com bobbie goods?</li>
                    <li onclick="askQuestion(this.textContent)">Que falou "Grande Amor" pela primeira vez?</li>
                    <li onclick="askQuestion(this.textContent)">Quais as nossas brigas mais bestas?</li>
                </ul>

                <div style="margin-top: 20px; padding: 15px; background: linear-gradient(45deg, #ffeaa7, #fab1a0); border-radius: 10px;">
                    <h4 style="color: #d63031; margin-bottom: 10px;"><i class="fas fa-info-circle"></i> Como usar:</h4>
                    <p style="font-size: 0.85em; color: #2d3436;">Manda bala no que quer saber, o que nosso guru do amor não souber ele inventa.</p>
                </div>
            </div>

            <div class="chat-section">
                <div class="chat-header">
                    <i class="fas fa-comments"></i> Pode perguntar, Big Love
                </div>
                <div class="chat-messages" id="chatMessages">
                    <div class="welcome-message">
                        <i class="fas fa-magic"></i> Bora loves, vem lembrar de alguns dos nossos momentos juntos <i class="fas fa-heart"></i>
                    </div>
                </div>
                <div class="chat-input-section">
                    <div class="chat-input">
                        <input 
                            type="text" 
                            id="messageInput" 
                            placeholder="O que quer saber, mini boi..."
                            onkeypress="if(event.key==='Enter') sendMessage()"
                        >
                        <button class="send-btn" onclick="sendMessage()">
                            <i class="fas fa-paper-plane"></i>
                        </button>
                    </div>
                </div>
            </div>
        </div>

        <div class="stats-section">
            <h3 style="color: #6c5ce7; margin-bottom: 15px;"><i class="fas fa-chart-heart"></i> Estatísticas das Conversas porque seu homem é rato de computação</h3>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-number" id="totalMessages">-</div>
                    <div>Conversas Analisadas</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number" id="totalParticipants">2</div>
                    <div>Participantes</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number" id="avgResponseTime">-</div>
                    <div>Status do Sistema</div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Configuração da API - detecta automaticamente o ambiente
        const API_BASE_URL = window.location.origin + '/api';
        
        let conversationData = {
            totalConversations: 0,
            participants: ["Você", "Parceiro(a)"],
            isLoaded: false,
            initializationInProgress: false
        };

        // Cria corações flutuantes
        function createFloatingHearts() {
            const heartsContainer = document.querySelector('.floating-hearts');
            for (let i = 0; i < 15; i++) {
                const heart = document.createElement('div');
                heart.className = 'heart';
                heart.innerHTML = '♥';
                heart.style.left = Math.random() * 100 + '%';
                heart.style.top = Math.random() * 100 + '%';
                heart.style.animationDelay = Math.random() * 6 + 's';
                heartsContainer.appendChild(heart);
            }
        }

        // Carrega dados reais do backend com retry
        async function loadConversationStats(retryCount = 0) {
            const maxRetries = 10; // Máximo de tentativas
            const retryDelay = 5000; // 5 segundos entre tentativas
            
            if (retryCount === 0) {
                addMessage('sistema', 'Conectando com o nosso amor astral... <div class="loading"></div>');
            }
            
            try {
                // Verifica status do sistema
                const statusResponse = await fetch(`${API_BASE_URL}/status`);
                
                if (!statusResponse.ok) {
                    throw new Error(`Status HTTP: ${statusResponse.status}`);
                }
                
                const status = await statusResponse.json();
                
                if (!status.chatbot_ready) {
                    if (retryCount >= maxRetries) {
                        addMessage('sistema', '❌ Timeout na inicialização do sistema. Recarregue a página.');
                        document.getElementById('avgResponseTime').textContent = 'Erro';
                        return;
                    }
                    
                    if (retryCount === 0) {
                        addMessage('sistema', 'Inicializando guru do amor... <div class="loading"></div>');
                    }
                    
                    // Tentativa de forçar inicialização
                    try {
                        await fetch(`${API_BASE_URL}/initialize`, { method: 'POST' });
                    } catch (e) {
                        console.log('Endpoint de inicialização não disponível');
                    }
                    
                    // Retry após delay
                    setTimeout(() => loadConversationStats(retryCount + 1), retryDelay);
                    return;
                }
                
                // Sistema pronto - carrega estatísticas
                const statsResponse = await fetch(`${API_BASE_URL}/stats`);
                const stats = await statsResponse.json();
                
                conversationData = {
                    totalConversations: stats.total_conversations || stats.total_documents || 0,
                    participants: stats.participants || ["Você", "Parceiro(a)"],
                    isLoaded: true,
                    initializationInProgress: false
                };
                
                updateStats();
                
                // Remove mensagens de carregamento
                const messages = document.getElementById('chatMessages');
                const loadingMessages = messages.querySelectorAll('.message');
                loadingMessages.forEach(msg => {
                    if (msg.textContent.includes('Conectando') || 
                        msg.textContent.includes('Aguardando') ||
                        msg.textContent.includes('Inicializando')) {
                        msg.remove();
                    }
                });
                
                // Mensagem de sucesso
                addMessage('sistema', `✨ Deu bom falar com o guru do amor, só perguntar agora`);
                
            } catch (error) {
                console.error('Erro ao conectar com o backend:', error);
                
                if (retryCount >= maxRetries) {
                    addMessage('sistema', `❌ Erro persistente ao conectar: ${error.message}`);
                    document.getElementById('avgResponseTime').textContent = 'Erro';
                    
                    // Fallback para modo demonstração
                    setTimeout(() => {
                        addMessage('sistema', '💡 Executando em modo demonstração.');
                        conversationData.totalConversations = 150;
                        conversationData.isLoaded = true;
                        updateStats();
                    }, 2000);
                } else {
                    // Retry após delay
                    setTimeout(() => loadConversationStats(retryCount + 1), retryDelay);
                }
            }
        }

        function updateStats() {
            document.getElementById('totalMessages').textContent = conversationData.totalConversations;
            document.getElementById('avgResponseTime').textContent = conversationData.isLoaded ? 'Online' : 'Carregando...';
        }

        function addMessage(type, content, messageId = null) {
            const messagesContainer = document.getElementById('chatMessages');
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message';
            
            if (messageId) {
                messageDiv.dataset.messageId = messageId;
            }

            if (type === 'user') {
                messageDiv.innerHTML = `<div class="user-message">${content}</div>`;
            } else if (type === 'sistema') {
                messageDiv.innerHTML = `<div class="bot-message"><i class="fas fa-robot"></i> ${content}</div>`;
            } else {
                messageDiv.innerHTML = `<div class="bot-message"><i class="fas fa-heart"></i> ${content}</div>`;
            }

            messagesContainer.appendChild(messageDiv);
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }

        function askQuestion(question) {
            document.getElementById('messageInput').value = question;
            sendMessage();
        }

        async function sendMessage() {
            const input = document.getElementById('messageInput');
            const message = input.value.trim();

            if (!message) return;

            // Adiciona mensagem do usuário
            addMessage('user', message);
            input.value = '';

            // Verifica se está conectado ao backend
            if (!conversationData.isLoaded) {
                addMessage('sistema', 'Aguarde a inicialização do sistema para fazer perguntas...');
                return;
            }

            // Mostra carregamento
            const loadingId = Date.now();
            addMessage('sistema', `Analisando suas conversas sobre "${message.toLowerCase()}"... <div class="loading"></div>`, loadingId);

            try {
                const startTime = Date.now();
                
                // Faz requisição real para a API
                const response = await fetch(`${API_BASE_URL}/chat`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ question: message }),
                    timeout: 30000 // Timeout de 30 segundos
                });

                if (!response.ok) {
                    throw new Error(`Erro ${response.status}: ${response.statusText}`);
                }

                const data = await response.json();
                
                if (data.error) {
                    throw new Error(data.error);
                }

                // Remove mensagem de loading
                removeMessageById(loadingId);

                // Adiciona resposta da IA
                addMessage('bot', data.answer);

            } catch (error) {
                console.error('Erro na requisição:', error);
                
                // Remove mensagem de loading
                removeMessageById(loadingId);
                
                let errorMessage = '❌ Erro ao processar sua pergunta. ';
                
                if (error.message.includes('fetch') || error.message.includes('NetworkError')) {
                    errorMessage += 'Problema de conexão com o servidor.';
                } else if (error.message.includes('500')) {
                    errorMessage += 'Erro interno do servidor.';
                } else if (error.message.includes('timeout')) {
                    errorMessage += 'Timeout na resposta do servidor.';
                } else {
                    errorMessage += error.message;
                }
                
                addMessage('sistema', errorMessage);
                
                // Fallback para demonstração em caso de erro persistente
                setTimeout(() => {
                    const demoResponses = [
                        `Baseando-me na análise de suas conversas, encontrei padrões interessantes sobre "${message.toLowerCase()}". Vocês demonstravam muito carinho através de pequenos gestos e palavras de apoio mútuo. ❤️`,
                        `Após analisar suas conversas relacionadas, percebo que vocês tinham uma comunicação muito afetuosa e compartilhavam muitos sonhos juntos. 💕`,
                        `Nas conversas analisadas, identifiquei que vocês se apoiavam muito nos momentos difíceis e celebravam juntos as vitórias. 🌟`
                    ];
                    const randomResponse = demoResponses[Math.floor(Math.random() * demoResponses.length)];
                    addMessage('bot', randomResponse);
                }, 1500);
            }
        }

        function removeMessageById(id) {
            const messages = document.getElementById('chatMessages');
            const messageElements = messages.querySelectorAll('.message');
            messageElements.forEach(msg => {
                if (msg.dataset.messageId == id) {
                    msg.remove();
                }
            });
        }

        // Inicialização
        document.addEventListener('DOMContentLoaded', function() {
            createFloatingHearts();
            loadConversationStats();
        });
    </script>
</body>
</html>
    """)

@app.route('/api/chat', methods=['POST'])
def chat():
    """Processa perguntas do usuário."""
    global chatbot, initialization_error
    
    if not chatbot:
        error_msg = initialization_error or 'Chatbot não inicializado. Aguarde a inicialização ou verifique os logs.'
        return jsonify({
            'error': error_msg
        }), 500
    
    try:
        data = request.json
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({'error': 'Pergunta não fornecida'}), 400
        
        print(f"Processando pergunta: {question}")
        start_time = time.time()
        
        result = chatbot.invoke({"query": question})
        response_time = time.time() - start_time
        
        # Formata a resposta para melhor apresentação
        formatted_answer = format_response(result["result"])
        
        # Processa documentos fonte (simplificado)
        source_docs = []
        if result.get("source_documents"):
            for i, doc in enumerate(result["source_documents"][:3], 1):
                source_info = {
                    'id': i,
                    'content_preview': doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
                }
                if hasattr(doc, 'metadata') and doc.metadata:
                    source_info['metadata'] = doc.metadata
                source_docs.append(source_info)
        
        return jsonify({
            'answer': formatted_answer,
            'response_time': round(response_time, 2),
            'sources_count': len(source_docs),
            'sources': source_docs,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"Erro no chat: {e}")
        return jsonify({
            'error': f'Erro ao processar pergunta: {str(e)}'
        }), 500

@app.route('/api/stats')
def get_stats():
    """Retorna estatísticas das conversas."""
    global conversation_stats
    return jsonify(conversation_stats)

@app.route('/api/status')
def get_status():
    """Retorna status do sistema."""
    global chatbot, db, initialization_error
    return jsonify({
        'chatbot_ready': chatbot is not None,
        'database_loaded': db is not None,
        'initialization_error': initialization_error,
        'stats': conversation_stats,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/initialize', methods=['POST'])
def force_initialize():
    """Força a inicialização do sistema."""
    global chatbot, db, initialization_error
    
    if chatbot is not None:
        return jsonify({
            'message': 'Sistema já inicializado',
            'chatbot_ready': True
        })
    
    try:
        print("Forçando inicialização...")
        success = initialize_chatbot_sync()
        
        if success:
            return jsonify({
                'message': 'Sistema inicializado com sucesso',
                'chatbot_ready': True
            })
        else:
            return jsonify({
                'error': initialization_error or 'Erro desconhecido na inicialização',
                'chatbot_ready': False
            }), 500
            
    except Exception as e:
        error_msg = f'Erro durante inicialização forçada: {e}'
        print(error_msg)
        return jsonify({
            'error': error_msg,
            'chatbot_ready': False
        }), 500

@app.route('/api/suggestions')
def get_suggestions():
    """Retorna sugestões de perguntas."""
    suggestions = [
        "Quem é urubacu?",
        "E a Taylor Swift de Limoeiro?",
        "Como demonstrávamos carinho um pelo outro?",
        "Quais planos futuros fizemos juntos?",
        "Quais atividades mais gostávamos de fazer?",
        "Quem começou com bobbie goods?",
        "Quem falou 'Grande Amor' pela primeira vez?",
        "Quais características mais admirávamos um no outro?",
        "Quais as nossas brigas mais bestas?",
        "Quais memórias mais especiais criamos juntos?"
    ]
    
    return jsonify({'suggestions': suggestions})

@app.route('/api/debug')
def debug_info():
    """Endpoint de debug para diagnosticar problemas."""
    global chatbot, db, conversation_stats, initialization_error
    
    import platform
    
    # Informações do sistema
    debug_data = {
        'system_info': {
            'platform': platform.platform(),
            'python_version': platform.python_version_tuple(),
            'working_directory': os.getcwd(),
            'environment_vars': {
                key: value for key, value in os.environ.items() 
                if not key.startswith('GOOGLE_API') and not 'SECRET' in key and not 'KEY' in key
            }
        },
        'file_system': {
            'current_directory': os.getcwd(),
            'contents': []
        },
        'chatbot_status': {
            'chatbot_ready': chatbot is not None,
        },
        'database_status': {
            'db_loaded': db is not None,
            'initialization_error': initialization_error,
            'stats': conversation_stats
        }
    }
    
    # Lista arquivos no diretório atual
    try:
        for item in os.listdir('.'):
            item_path = os.path.join('.', item)
            is_dir = os.path.isdir(item_path)
            size = 0 if is_dir else os.path.getsize(item_path)
            
            file_info = {
                'name': item,
                'is_directory': is_dir,
                'size': size
            }
            
            # Se for o diretório chroma_db, lista conteúdo
            if item == 'chroma_db' and is_dir:
                try:
                    file_info['contents'] = os.listdir(item_path)
                except Exception as e:
                    file_info['contents'] = f"Erro ao listar: {e}"
            
            debug_data['file_system']['contents'].append(file_info)
    except Exception as e:
        debug_data['file_system']['error'] = str(e)
    
    return jsonify(debug_data)

if __name__ == '__main__':
    print("=== INICIANDO SERVIDOR DO CHATBOT (RENDER MODE) ===")
    
    # Para o Render, inicializa de forma síncrona
    print("Inicializando chatbot de forma síncrona...")
    
    try:
        # Inicialização síncrona para o Render
        success = initialize_chatbot_sync()
        if success:
            print("✅ Chatbot inicializado com sucesso!")
        else:
            print(f"❌ Falha na inicialização: {initialization_error}")
    except Exception as e:
        print(f"❌ Erro crítico na inicialização: {e}")
        initialization_error = str(e)
    
    print("Servidor Flask iniciando...")
    print("Interface disponível na URL do Render")
    print("API disponível em: /api/")
    
    # Usa a porta fornecida pelo Render
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port, threaded=True)