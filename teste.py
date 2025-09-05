import os
import json
import time
import re
from datetime import datetime, timedelta
from getpass import getpass
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
import hashlib
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema import Document

class WhatsAppProcessor:
    def __init__(self):
        self.system_messages = [
            'mensagem apagada', 'media omitted', 'audio omitted', 'image omitted',
            'document omitted', 'video omitted', 'sticker omitted', 'gif omitted',
            'contact card omitted', 'location omitted', 'você foi adicionado',
            'saiu', 'entrou no grupo', 'mudou o assunto', 'mudou a descrição',
            'criou o grupo', 'removeu', 'adicionado', 'chamada de voz',
            'chamada de vídeo', 'perdida', 'ocupado', 'rejeitada'
        ]
        
        self.emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map
            "\U0001F1E0-\U0001F1FF"  # flags
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE
        )

    def parse_timestamp(self, timestamp_str: str) -> Optional[datetime]:
        """Parse robusto de timestamps do WhatsApp"""
        if not timestamp_str:
            return None
            
        # Formatos comuns do WhatsApp
        formats = [
            '%d/%m/%Y, %H:%M',
            '%d/%m/%y, %H:%M',
            '%m/%d/%Y, %H:%M',
            '%Y-%m-%d %H:%M:%S',
            '%d-%m-%Y %H:%M',
            '%d/%m/%Y %H:%M:%S',
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%dT%H:%M:%S.%f'
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(timestamp_str.strip(), fmt)
            except ValueError:
                continue
                
        # Tentativa com regex para extrair data/hora
        date_match = re.search(r'(\d{1,2})[/\-](\d{1,2})[/\-](\d{2,4})', timestamp_str)
        time_match = re.search(r'(\d{1,2}):(\d{2})', timestamp_str)
        
        if date_match and time_match:
            try:
                day, month, year = map(int, date_match.groups())
                hour, minute = map(int, time_match.groups())
                
                if year < 100:
                    year += 2000
                    
                return datetime(year, month, day, hour, minute)
            except ValueError:
                pass
                
        return None

    def extract_keywords(self, text: str) -> List[str]:
        """Extrai palavras-chave relevantes do texto"""
        # Remove pontuação e divide em palavras
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Remove stopwords comuns em português
        stopwords = {
            'de', 'a', 'o', 'que', 'e', 'do', 'da', 'em', 'um', 'para', 'é', 'com', 'não',
            'uma', 'os', 'no', 'se', 'na', 'por', 'mais', 'as', 'dos', 'como', 'mas', 'foi',
            'ao', 'ele', 'das', 'tem', 'à', 'seu', 'sua', 'ou', 'ser', 'quando', 'muito',
            'nos', 'já', 'eu', 'também', 'só', 'pelo', 'pela', 'até', 'isso', 'ela',
            'entre', 'era', 'depois', 'sem', 'mesmo', 'aos', 'ter', 'seus', 'suas',
            'você', 'vocês', 'meu', 'minha', 'nosso', 'nossa', 'dele', 'dela',
            'vou', 'vai', 'vamos', 'então', 'aqui', 'lá', 'onde', 'porque', 'ainda',
            'bem', 'só', 'sim', 'não', 'né', 'ok', 'kkk', 'kkkk', 'rs', 'rsrs'
        }
        
        # Conta palavras, ignorando stopwords e palavras muito curtas
        word_count = defaultdict(int)
        for word in words:
            if len(word) > 2 and word not in stopwords:
                word_count[word] += 1
                
        # Retorna as palavras mais frequentes
        return sorted(word_count.keys(), key=word_count.get, reverse=True)

    def detect_conversation_boundaries_conservative(self, messages: List[Dict]) -> List[List[Dict]]:
        """Detecta limites de conversas de forma mais conservadora para preservar contexto"""
        if not messages:
            return []
            
        conversations = []
        current_conversation = []
        
        print("🔍 Analisando limites de conversas (modo conservador)...")
        
        for i, message in enumerate(messages):
            if i % 1000 == 0 and i > 0:
                print(f"   Processando mensagem {i}/{len(messages)} ({(i/len(messages)*100):.1f}%)")
                
            if not current_conversation:
                current_conversation.append(message)
                continue
                
            last_msg = current_conversation[-1]
            current_time = self.parse_timestamp(message['timestamp'])
            last_time = self.parse_timestamp(last_msg['timestamp'])
            
            # Critérios mais conservadores para nova conversa
            should_split = False
            
            # 1. Gap temporal muito grande (mais de 24 horas)
            if current_time and last_time:
                time_gap = (current_time - last_time).total_seconds() / 3600
                if time_gap > 24:  # Aumentado de 4h para 24h
                    should_split = True
            
            # 2. Conversa muito longa (mais de 100 mensagens) - aumentado de 50
            if len(current_conversation) > 100:
                should_split = True
            
            # 3. Mudança drástica de participantes (só se for um grupo)
            recent_authors = set([msg['author'] for msg in current_conversation[-20:]])
            if len(recent_authors) > 2 and message['author'] not in recent_authors:
                # Só divide se for claramente um grupo diferente
                should_split = True
            
            if should_split and len(current_conversation) >= 5:  # Mínimo aumentado para 5
                conversations.append(current_conversation)
                current_conversation = [message]
            else:
                current_conversation.append(message)
        
        # Adiciona a última conversa
        if len(current_conversation) >= 5:  # Mínimo aumentado para 5
            conversations.append(current_conversation)
            
        print(f"✅ Análise concluída: {len(conversations)} conversas detectadas (modo conservador)")
        return conversations

    def create_overlapping_chunks(self, messages: List[Dict], chunk_size: int = 30, overlap: int = 10) -> List[List[Dict]]:
        """Cria chunks sobrepostos para garantir que nenhuma informação importante seja perdida"""
        if not messages:
            return []
            
        chunks = []
        
        print(f"📦 Criando chunks sobrepostos (tamanho: {chunk_size}, sobreposição: {overlap})...")
        
        for i in range(0, len(messages), chunk_size - overlap):
            chunk = messages[i:i + chunk_size]
            if len(chunk) >= 5:  # Apenas chunks com pelo menos 5 mensagens
                chunks.append(chunk)
                
        print(f"✅ {len(chunks)} chunks criados")
        return chunks

    def clean_message(self, message: str) -> str:
        """Limpa mensagens preservando mais conteúdo"""
        if not message:
            return ""
            
        # Remove apenas caracteres de controle realmente problemáticos
        message = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', message)
        
        # Normaliza espaços em branco
        message = re.sub(r'\s+', ' ', message)
        
        # Preserva URLs (apenas marca URLs muito longas)
        if len(message) > 200 and 'http' in message:
            message = re.sub(r'http[s]?://[^\s]{100,}', '[link_longo]', message)
        
        return message.strip()

    def is_system_message(self, message: str) -> bool:
        """Detecta mensagens do sistema de forma mais conservadora"""
        if not message or len(message.strip()) < 2:
            return True
            
        message_lower = message.lower().strip()
        
        # Verifica apenas mensagens claramente do sistema
        for sys_msg in self.system_messages:
            if message_lower == sys_msg or message_lower.startswith(sys_msg):
                return True
                
        # Padrões muito específicos
        system_patterns = [
            r'^\s*\[.*\]\s*$',  # [mensagem do sistema]
            r'^chamada (perdida|rejeitada)$',
            r'^você foi adicionado$',
            r'^.* saiu$',
            r'^.* mudou .*para.*$'
        ]
        
        for pattern in system_patterns:
            if re.match(pattern, message_lower):
                return True
                
        return False

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

def create_whatsapp_database(json_path: str, db_path: str = "./chroma_db_optimized") -> Chroma:
    """Função principal para criar o banco de dados do WhatsApp"""
    processor = WhatsAppProcessor()
    
    print(f"\n{'='*60}")
    print(f"🚀 CRIANDO BANCO DE DADOS WHATSAPP OTIMIZADO")
    print(f"{'='*60}")
    
    # Etapa 1: Carregamento
    print(f"📂 Carregando dados de '{json_path}'...")
    start_time = time.time()
    
    try:
        with open(json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        load_time = time.time() - start_time
        print(f"✅ Arquivo carregado em {load_time:.2f}s")
        print(f"📊 Total de mensagens brutas: {len(data):,}")
    except Exception as e:
        print(f"❌ Erro ao carregar arquivo: {e}")
        return None
    
    # Etapa 2: Filtragem conservadora
    print(f"\n🔍 FILTRANDO MENSAGENS (modo conservador)...")
    filter_start = time.time()
    
    valid_messages = []
    system_count = 0
    empty_count = 0
    
    for i, item in enumerate(data):
        if i % 5000 == 0 and i > 0:
            print(f"   Processando item {i:,}/{len(data):,} ({(i/len(data)*100):.1f}%)")
            
        author = item.get('author', 'Desconhecido')
        message = item.get('message', '').strip()
        timestamp = item.get('timestamp', '')
        
        if len(message) < 1:  # Menos restritivo
            empty_count += 1
            continue
            
        if processor.is_system_message(message):
            system_count += 1
            continue
            
        # Limpa a mensagem preservando mais conteúdo
        cleaned_message = processor.clean_message(message)
        
        valid_messages.append({
            'author': author,
            'message': cleaned_message,
            'original_message': message,
            'timestamp': timestamp,
            'index': i
        })
    
    filter_time = time.time() - filter_start
    print(f"✅ Filtragem concluída em {filter_time:.2f}s")
    print(f"📈 Mensagens válidas: {len(valid_messages):,}")
    print(f"🗑️ Mensagens de sistema removidas: {system_count:,}")
    print(f"🗑️ Mensagens vazias removidas: {empty_count:,}")
    print(f"📊 Taxa de aproveitamento: {(len(valid_messages)/len(data)*100):.1f}%")
    
    # Etapa 3: Estratégia dupla - conversas + chunks
    print(f"\n🧩 APLICANDO ESTRATÉGIA DUPLA...")
    
    # Estratégia 1: Conversas conservadoras
    conversations = processor.detect_conversation_boundaries_conservative(valid_messages)
    
    # Estratégia 2: Chunks sobrepostos para garantir cobertura
    chunks = processor.create_overlapping_chunks(valid_messages, chunk_size=40, overlap=15)
    
    print(f"📊 Estratégia dupla:")
    print(f"   • Conversas naturais: {len(conversations)}")
    print(f"   • Chunks sobrepostos: {len(chunks)}")
    
    # Etapa 4: Criação de documentos
    print(f"\n📝 CRIANDO DOCUMENTOS...")
    doc_start = time.time()
    
    documents = []
    doc_id = 0
    
    # Adiciona conversas naturais
    print(f"📝 Processando conversas naturais...")
    for i, conversation in enumerate(conversations):
        if i % 10 == 0 and i > 0:
            print(f"   Conversa {i}/{len(conversations)} ({(i/len(conversations)*100):.1f}%)")
            
        if len(conversation) < 5:
            continue
            
        # Cria contexto rico
        participants = list(set([msg['author'] for msg in conversation]))
        timestamps = [processor.parse_timestamp(msg['timestamp']) for msg in conversation]
        valid_timestamps = [ts for ts in timestamps if ts]
        
        # Extrai palavras-chave do conteúdo
        all_text = ' '.join([msg['message'] for msg in conversation])
        keywords = processor.extract_keywords(all_text)
        
        context_lines = [
            f"=== CONVERSA NATURAL #{doc_id} ===",
            f"Participantes: {', '.join(participants)}",
            f"Mensagens: {len(conversation)}",
            f"Palavras-chave: {', '.join(keywords[:8])}",
            ""
        ]
        
        # Adiciona todas as mensagens com contexto mínimo
        for msg in conversation:
            timestamp = msg['timestamp']
            author = msg['author']
            content = msg['message']
            
            context_lines.append(f"[{timestamp}] {author}: {content}")
        
        # Metadados simples
        metadata = {
            'type': 'conversation',
            'doc_id': str(doc_id),
            'participants': ', '.join(participants),
            'message_count': len(conversation),
            'keywords': ', '.join(keywords[:10]),
            'start_time': str(valid_timestamps[0]) if valid_timestamps else '',
            'end_time': str(valid_timestamps[-1]) if valid_timestamps else ''
        }
        
        documents.append(Document(
            page_content="\n".join(context_lines),
            metadata=metadata
        ))
        doc_id += 1
    
    # Adiciona chunks sobrepostos
    print(f"📝 Processando chunks sobrepostos...")
    for i, chunk in enumerate(chunks):
        if i % 20 == 0 and i > 0:
            print(f"   Chunk {i}/{len(chunks)} ({(i/len(chunks)*100):.1f}%)")
            
        participants = list(set([msg['author'] for msg in chunk]))
        all_text = ' '.join([msg['message'] for msg in chunk])
        keywords = processor.extract_keywords(all_text)
        
        context_lines = [
            f"=== CHUNK #{doc_id} ===",
            f"Participantes: {', '.join(participants)}",
            f"Mensagens: {len(chunk)}",
            f"Palavras-chave: {', '.join(keywords[:5])}",
            ""
        ]
        
        # Adiciona mensagens do chunk
        for msg in chunk:
            timestamp = msg['timestamp']
            author = msg['author']
            content = msg['message']
            
            context_lines.append(f"[{timestamp}] {author}: {content}")
        
        metadata = {
            'type': 'chunk',
            'doc_id': str(doc_id),
            'participants': ', '.join(participants),
            'message_count': len(chunk),
            'keywords': ', '.join(keywords[:10])
        }
        
        documents.append(Document(
            page_content="\n".join(context_lines),
            metadata=metadata
        ))
        doc_id += 1
    
    doc_time = time.time() - doc_start
    print(f"✅ Documentos criados em {doc_time:.2f}s")
    print(f"📄 Total de documentos: {len(documents):,}")
    
    # Etapa 5: Criação do banco vetorial
    print(f"\n🗄️ CRIANDO BANCO VETORIAL...")
    
    embedding_model = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        task_type="retrieval_document"
    )
    
    # Criação em batches com logs
    batch_size = 10
    total_batches = (len(documents) + batch_size - 1) // batch_size
    
    print(f"📊 Configuração do banco:")
    print(f"   • Total de documentos: {len(documents):,}")
    print(f"   • Tamanho do batch: {batch_size}")
    print(f"   • Total de batches: {total_batches}")
    
    db_start = time.time()
    
    # Primeiro batch
    print(f"🚀 Criando banco com primeiro batch...")
    try:
        vector_db = Chroma.from_documents(
            documents=documents[:batch_size],
            embedding=embedding_model,
            persist_directory=db_path
        )
        print(f"✅ Primeiro batch criado!")
    except Exception as e:
        print(f"❌ Erro ao criar banco: {e}")
        return None
    
    # Batches restantes
    for batch_num in range(1, total_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(documents))
        
        print(f"📦 Batch {batch_num + 1}/{total_batches} (docs {start_idx}-{end_idx-1})")
        
        try:
            vector_db.add_documents(documents[start_idx:end_idx])
            progress = ((batch_num + 1) / total_batches) * 100
            print(f"✅ Progresso: {progress:.1f}%")
        except Exception as e:
            print(f"⚠️ Erro no batch {batch_num + 1}: {e}")
    
    db_time = time.time() - db_start
    total_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"🎉 BANCO CRIADO COM SUCESSO!")
    print(f"{'='*60}")
    print(f"⏱️ Tempo total: {total_time:.2f}s ({total_time/60:.1f} min)")
    print(f"🗄️ Tempo do banco: {db_time:.2f}s")
    print(f"📍 Localização: {db_path}")
    
    # Teste de busca
    print(f"\n🧪 Testando busca...")
    try:
        # Teste com uma palavra que deveria estar no banco
        test_query = "Taylor"
        results = vector_db.similarity_search(test_query, k=3)
        print(f"✅ Teste '{test_query}': {len(results)} resultado(s) encontrado(s)")
        
        if results:
            for i, result in enumerate(results[:2]):
                preview = result.page_content[:150] + "..."
                print(f"   Resultado {i+1}: {preview}")
        
    except Exception as e:
        print(f"⚠️ Erro no teste: {e}")
    
    return vector_db

if __name__ == "__main__":
    print("🚀 WhatsApp Database Creator - Versão Otimizada")
    print("=" * 60)
    
    configure_api_key()
    
    json_file_path = "conversa_formatada.json"
    db_directory = "./chroma_db_optimized"
    
    if not os.path.exists(json_file_path):
        print(f"❌ Arquivo '{json_file_path}' não encontrado!")
        exit()
    
    # Cria o banco
    db = create_whatsapp_database(json_file_path, db_directory)
    
    if db:
        print(f"\n✅ Banco de dados criado com sucesso!")
        print(f"📂 Use este caminho em outros scripts: {db_directory}")
        
        # Teste adicional
        print(f"\n🔍 Teste final de busca...")
        test_terms = ["Taylor", "primeira", "Limoeiro", "Swift"]
        
        for term in test_terms:
            try:
                results = db.similarity_search(term, k=2)
                print(f"   '{term}': {len(results)} resultado(s)")
            except:
                print(f"   '{term}': erro na busca")
    else:
        print(f"❌ Falha ao criar banco de dados!")