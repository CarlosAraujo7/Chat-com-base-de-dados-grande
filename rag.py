import os
import json
import time
import re
from datetime import datetime
from getpass import getpass

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema import Document
from dotenv import load_dotenv
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
    Adiciona contexto temporal e agrupa mensagens relacionadas.
    """
    print(f"Carregando e pré-processando dados de '{json_path}'...")
    
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    documents = []
    
    # Agrupa mensagens em conversas (por proximidade temporal)
    conversation_window = []
    last_timestamp = None
    conversation_id = 0
    
    for i, item in enumerate(data):
        # Extrai informações básicas
        author = item.get('author', 'Desconhecido')
        message = item.get('message', '').strip()
        timestamp = item.get('timestamp', '')
        
        # Pula mensagens vazias ou muito curtas
        if len(message) < 3:
            continue
            
        # Remove mensagens de sistema comum do WhatsApp
        system_messages = [
            'mensagem apagada',
            'media omitted',
            'audio omitted',
            'image omitted',
            'document omitted',
            'video omitted'
        ]
        
        if any(sys_msg in message.lower() for sys_msg in system_messages):
            continue
        
        # Determina se inicia nova conversa (gap de mais de 1 hora ou mudança significativa de tópico)
        current_time = timestamp
        if (last_timestamp and 
            should_start_new_conversation(last_timestamp, current_time, len(conversation_window))):
            
            # Salva a conversa anterior se tiver conteúdo suficiente
            if len(conversation_window) >= 3:  # Mínimo de 3 mensagens para formar contexto
                conversation_text = create_conversation_context(conversation_window, conversation_id)
                
                # Converte listas em strings para compatibilidade com ChromaDB
                participants = list(set([msg['author'] for msg in conversation_window]))
                participants_str = ', '.join(participants)  # Converte lista em string
                
                documents.append(Document(
                    page_content=conversation_text,
                    metadata={
                        'conversation_id': str(conversation_id),  # Converte para string
                        'start_time': str(conversation_window[0]['timestamp']),
                        'end_time': str(conversation_window[-1]['timestamp']),
                        'message_count': len(conversation_window),  # Mantém como int
                        'participants': participants_str  # Agora é string
                    }
                ))
                conversation_id += 1
            
            conversation_window = []
        
        conversation_window.append({
            'author': author,
            'message': message,
            'timestamp': timestamp,
            'index': i
        })
        
        last_timestamp = current_time
    
    # Adiciona a última conversa
    if len(conversation_window) >= 3:
        conversation_text = create_conversation_context(conversation_window, conversation_id)
        
        # Converte listas em strings para compatibilidade com ChromaDB
        participants = list(set([msg['author'] for msg in conversation_window]))
        participants_str = ', '.join(participants)  # Converte lista em string
        
        documents.append(Document(
            page_content=conversation_text,
            metadata={
                'conversation_id': str(conversation_id),  # Converte para string
                'start_time': str(conversation_window[0]['timestamp']),
                'end_time': str(conversation_window[-1]['timestamp']),
                'message_count': len(conversation_window),  # Mantém como int
                'participants': participants_str  # Agora é string
            }
        ))
    
    print(f"Dados processados: {len(documents)} conversas contextualizadas criadas.")
    return documents

def should_start_new_conversation(last_time, current_time, window_size):
    """Determina se deve iniciar uma nova conversa baseado em critérios temporais e de tamanho."""
    # Se a janela já tem muitas mensagens (mais de 20), inicia nova conversa
    if window_size > 20:
        return True
    
    # Adicione aqui lógica de comparação temporal se os timestamps forem processáveis
    # Por simplicidade, vamos usar apenas o tamanho da janela
    return False

def create_conversation_context(messages, conv_id):
    """Cria um contexto rico para uma conversa."""
    participants = list(set([msg['author'] for msg in messages]))
    start_time = messages[0]['timestamp']
    
    # Cria um resumo da conversa
    context_lines = [
        f"=== CONVERSA #{conv_id} ===",
        f"Participantes: {', '.join(participants)}",
        f"Início: {start_time}",
        f"Número de mensagens: {len(messages)}",
        ""
    ]
    
    # Adiciona as mensagens com melhor formatação
    for msg in messages:
        author = msg['author']
        message = msg['message']
        timestamp = msg['timestamp']
        
        # Limpa e formata a mensagem
        cleaned_message = clean_message(message)
        if cleaned_message:
            context_lines.append(f"[{timestamp}] {author}: {cleaned_message}")
    
    return "\n".join(context_lines)

def clean_message(message):
    """Limpa e normaliza mensagens."""
    # Remove caracteres especiais excessivos
    message = re.sub(r'[^\w\s\.,!?;:()\-áàâãéèêíïóôõöúçñ]', ' ', message)
    # Remove espaços múltiplos
    message = re.sub(r'\s+', ' ', message)
    return message.strip()

def create_vector_database_with_logging(documents, embedding_model, db_path="./chroma_db"):
    """Cria e armazena os embeddings em um banco de dados vetorial Chroma com logs de progresso."""
    print("Iniciando a criação do banco de dados vetorial...")
    
    total_docs = len(documents)
    batch_size = 10  # Reduzido para conversas mais longas
    
    start_time = time.time()
    
    print(f"Processando lote 1 de {(total_docs // batch_size) + 1}...")
    
    # CORREÇÃO: Remove a chamada .persist() pois não existe mais na versão atual
    vector_db = Chroma.from_documents(
        documents=documents[:batch_size],
        embedding=embedding_model,
        persist_directory=db_path
    )
    
    for i in range(batch_size, total_docs, batch_size):
        start_batch_time = time.time()
        
        progress = (i / total_docs) * 100
        elapsed_time = time.time() - start_time
        avg_time_per_doc = elapsed_time / i if i > 0 else 0
        remaining_docs = total_docs - i
        eta = avg_time_per_doc * remaining_docs
        
        print(f"Processando lote {(i // batch_size) + 1} de {(total_docs // batch_size) + 1}... "
              f"({progress:.2f}% concluído, ETA: {eta:.2f}s)")
        
        vector_db.add_documents(documents[i:i+batch_size])
        
        batch_time = time.time() - start_batch_time
        print(f"Lote processado em {batch_time:.2f}s.")

    print("Finalizando e salvando o banco de dados...")
    # CORREÇÃO: O Chroma agora persiste automaticamente quando persist_directory é especificado
    # Não é necessário chamar .persist() explicitamente
    
    total_time = time.time() - start_time
    print(f"Banco de dados vetorial criado com sucesso em {total_time:.2f} segundos.")
    print(f"Dados salvos em: {db_path}")
    return vector_db

def create_qa_chain_compatible(vector_db, is_old_structure=False):
    """Cria a cadeia de Pergunta e Resposta (QA Chain) com compatibilidade para estruturas antigas."""
    print("Configurando o chatbot...")
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        temperature=0.1,
        max_output_tokens=1000
    )
    
    # Prompt adaptado para estruturas diferentes
    if is_old_structure:
        prompt_template = """Você é um assistente especializado em analisar conversas de relacionamento.

INSTRUÇÕES IMPORTANTES:
1. Use APENAS as informações das mensagens fornecidas no contexto
2. As mensagens podem estar fragmentadas - tente conectar informações relacionadas
3. Se não encontrar informações relevantes, diga: "Não encontrei informações específicas sobre isso nas conversas"
4. Seja empático e respeitoso ao tratar de temas pessoais
5. Tente identificar padrões e dinâmicas do relacionamento baseado nas mensagens disponíveis

MENSAGENS DAS CONVERSAS:
{context}

PERGUNTA: {question}

RESPOSTA DETALHADA:"""
    else:
        prompt_template = """Você é um assistente especializado em analisar conversas de relacionamento.

INSTRUÇÕES IMPORTANTES:
1. Use APENAS as informações das conversas fornecidas no contexto
2. Se não encontrar informações relevantes, diga: "Não encontrei informações específicas sobre isso nas conversas analisadas"
3. Cite trechos específicos quando possível
4. Seja empático e respeitoso ao tratar de temas pessoais
5. Identifique padrões, momentos importantes e dinâmicas do relacionamento
6. Se a pergunta for sobre sentimentos ou interpretações, baseie-se apenas no que foi explicitamente dito

CONTEXTO DAS CONVERSAS:
{context}

PERGUNTA: {question}

RESPOSTA DETALHADA:"""

    QA_PROMPT = PromptTemplate(
        template=prompt_template, 
        input_variables=["context", "question"]
    )
    
    # Configuração do retriever mais simples para compatibilidade
    try:
        retriever = vector_db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 8}  # Mais documentos para compensar mensagens fragmentadas
        )
    except Exception as e:
        print(f"Erro na configuração avançada do retriever: {e}")
        # Fallback para configuração mais básica
        retriever = vector_db.as_retriever(search_kwargs={"k": 8})

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_PROMPT}
    )
    
    print("Chatbot pronto para conversar!")
    return qa_chain

def display_conversation_stats(db):
    """Exibe estatísticas sobre as conversas carregadas."""
    try:
        # CORREÇÃO: Método mais robusto para obter contagem
        collection_count = db._collection.count()
        print(f"\n=== ESTATÍSTICAS DAS CONVERSAS ===")
        print(f"Total de documentos indexados: {collection_count}")
        
        # Tenta obter algumas amostras
        try:
            # CORREÇÃO: Usa peek() para obter amostras de forma mais segura
            try:
                sample_docs = db._collection.peek(limit=3)
            except:
                # Fallback para método alternativo
                sample_docs = db.get(limit=3)
                
            if sample_docs and 'documents' in sample_docs:
                print(f"Exemplos de documentos encontrados: {len(sample_docs['documents'])}")
                
                # Mostra um exemplo de conteúdo
                if sample_docs['documents']:
                    first_doc = sample_docs['documents'][0]
                    preview = first_doc[:200] + "..." if len(first_doc) > 200 else first_doc
                    print(f"Exemplo de conteúdo: {preview}")
                    
                # Verifica metadados se existirem
                if sample_docs.get('metadatas'):
                    metadata_example = sample_docs['metadatas'][0] if sample_docs['metadatas'] and sample_docs['metadatas'][0] else {}
                    if metadata_example:
                        print(f"Metadados disponíveis: {list(metadata_example.keys())}")
                        
                        # Mostra participantes se disponível
                        if 'participants' in metadata_example:
                            print(f"Exemplo de participantes: {metadata_example['participants']}")
                        
        except Exception as e:
            print(f"Não foi possível acessar amostras: {e}")
            
    except Exception as e:
        print(f"Erro ao obter estatísticas: {e}")
    
    print("=====================================\n")

# Função para sugerir perguntas interessantes
def suggest_questions():
    """Sugere perguntas interessantes para explorar o relacionamento."""
    suggestions = [
        "Quais foram os momentos mais felizes que vocês compartilharam?",
        "Sobre o que vocês mais conversaram?",
        "Quais foram os principais desafios ou conflitos?",
        "Como vocês demonstraram carinho um pelo outro?",
        "Quais planos futuros vocês fizeram juntos?",
        "Quais foram as principais mudanças no relacionamento ao longo do tempo?",
        "Quais atividades vocês mais gostavam de fazer juntos?",
        "Como vocês se apoiaram nos momentos difíceis?",
        "Quais foram as conversas mais profundas que tiveram?",
        "Quais características vocês mais admiravam um no outro?"
    ]
    
    print("\n=== SUGESTÕES DE PERGUNTAS ===")
    for i, question in enumerate(suggestions, 1):
        print(f"{i}. {question}")
    print("===============================\n")

def check_file_exists(file_path):
    """Verifica se o arquivo existe e fornece orientações caso não exista."""
    if not os.path.exists(file_path):
        print(f"❌ ERRO: Arquivo '{file_path}' não encontrado!")
        print("\n💡 COMO RESOLVER:")
        print("1. Certifique-se de que o arquivo está na mesma pasta do script")
        print("2. Verifique se o nome do arquivo está correto")
        print("3. O arquivo deve ser um JSON no formato:")
        print('[{"author": "Nome", "message": "Mensagem", "timestamp": "Data"}, ...]')
        return False
    return True

if __name__ == "__main__":
    configure_api_key()
    
    json_file_path = "conversa_formatada.json"
    db_directory = "./chroma_db"
    
    # CORREÇÃO: Verifica se o arquivo existe antes de prosseguir
    if not check_file_exists(json_file_path):
        exit()
    
    embedding_model = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        task_type="retrieval_document"  # Otimizado para recuperação
    )
    
    # Verifica se o banco de dados já existe
    recreate_db = False
    db_directory_new = "./chroma_db_optimized"
    
    if os.path.exists(db_directory):
        print("Banco de dados vetorial existente encontrado.")
        print("ATENÇÃO: Detectado um banco muito grande (provavelmente da versão antiga).")
        print(f"Total de documentos: {len(os.listdir(db_directory)) if os.path.exists(db_directory) else 'Desconhecido'}")
        
        choice = input("Deseja criar um (N)ovo banco otimizado, (U)sar o existente ou (T)entar usar ambos? [N/U/T]: ").upper()
        
        if choice == 'N':
            recreate_db = True
            db_directory = db_directory_new
            print(f"Criando novo banco otimizado em: {db_directory}")
        elif choice == 'T':
            # Tenta usar o banco existente primeiro
            try:
                print("Testando banco existente...")
                start_load_time = time.time()
                db = Chroma(persist_directory=db_directory, embedding_function=embedding_model)
                
                # Teste simples para ver se funciona
                test_query = db.similarity_search("teste", k=1)
                load_time = time.time() - start_load_time
                print(f"Banco existente carregado com sucesso em {load_time:.2f}s.")
                
                # Se chegou até aqui, o banco funciona
                print("✅ Banco existente está funcionando!")
                
            except Exception as e:
                print(f"❌ Erro ao usar banco existente: {e}")
                print("Criando novo banco otimizado...")
                recreate_db = True
                db_directory = db_directory_new
        else:
            print("Carregando banco existente...")
            try:
                start_load_time = time.time()
                db = Chroma(persist_directory=db_directory, embedding_function=embedding_model)
                load_time = time.time() - start_load_time
                print(f"Banco de dados carregado em {load_time:.2f}s.")
            except Exception as e:
                print(f"Erro ao carregar banco existente: {e}")
                print("Criando novo banco...")
                recreate_db = True
                db_directory = db_directory_new
    else:
        recreate_db = True
        print("Nenhum banco de dados encontrado. Criando novo...")
    
    if recreate_db:
        print("Processando dados do WhatsApp...")
        docs = preprocess_whatsapp_data(json_file_path)
        
        if not docs:
            print("ERRO: Nenhuma conversa foi processada.")
            print("Verifique se o arquivo 'conversa_formatada.json' existe e tem o formato correto:")
            print('Formato esperado: [{"author": "Nome", "message": "Mensagem", "timestamp": "Data"}]')
            exit()
        
        print(f"✅ {len(docs)} conversas contextualizadas criadas!")
        db = create_vector_database_with_logging(docs, embedding_model, db_directory)
    
    # Exibe estatísticas
    display_conversation_stats(db)
    
    # Determina se é estrutura antiga baseado no número de documentos
    try:
        doc_count = db._collection.count()
        is_old_structure = doc_count > 10000
        if is_old_structure:
            print("🔄 Detectada estrutura de banco antiga (muitos documentos pequenos)")
            print("   Usando configuração compatível...")
    except Exception as e:
        print(f"Não foi possível determinar estrutura do banco: {e}")
        is_old_structure = False
    
    # Cria o chatbot
    chatbot = create_qa_chain_compatible(db, is_old_structure)

    # Sugere perguntas
    suggest_questions()

    print("\n--- Chat com Memórias do Relacionamento ---")
    print("Digite 'sair' para terminar, ou 'sugestões' para ver mais ideias de perguntas.")

    while True:
        query = input("\nFaça uma pergunta sobre o relacionamento: ")
        
        if query.lower() == 'sair':
            break
        elif query.lower() == 'sugestões':
            suggest_questions()
            continue

        try:
            start_time = time.time()
            result = chatbot.invoke({"query": query})
            response_time = time.time() - start_time

            print(f"\n{'='*50}")
            print("RESPOSTA:")
            print(f"{'='*50}")
            print(result["result"])
            print(f"\n⏱️ Tempo de resposta: {response_time:.2f}s")

            # Mostra fontes de forma mais organizada
            if result.get("source_documents"):
                print(f"\n{'='*50}")
                print("CONVERSAS CONSULTADAS:")
                print(f"{'='*50}")
                for i, doc in enumerate(result["source_documents"], 1):
                    print(f"\n--- Conversa {i} ---")
                    
                    # Verifica se tem metadados
                    if hasattr(doc, 'metadata') and doc.metadata:
                        metadata = doc.metadata
                        if 'conversation_id' in metadata:
                            print(f"ID: {metadata['conversation_id']}")
                        if 'start_time' in metadata:
                            print(f"Data: {metadata['start_time']}")
                        if 'message_count' in metadata:
                            print(f"Mensagens: {metadata['message_count']}")
                        if 'participants' in metadata:
                            print(f"Participantes: {metadata['participants']}")  # Já é string agora
                    
                    # Mostra um trecho da conversa
                    content = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
                    print(f"Trecho: {content}")
            else:
                print("\n🔍 Nenhuma conversa específica foi consultada para esta resposta.")
                    
        except Exception as e:
            error_msg = str(e)
            print(f"\n❌ Erro ao processar pergunta: {error_msg}")
            
            if "score_threshold" in error_msg:
                print("💡 Problema detectado: Incompatibilidade de versão do Chroma.")
                print("   Recomendo recriar o banco de dados com a nova estrutura.")
            elif "API" in error_msg.upper():
                print("💡 Problema de conexão com a API do Google. Verifique sua chave e conexão.")
            else:
                print("💡 Tente reformular sua pergunta de forma mais específica.")
                
            print("   Digite 'sair' para encerrar ou tente uma nova pergunta.")

    print("\n👋 Obrigado por usar o Chat com Memórias! Até logo!")