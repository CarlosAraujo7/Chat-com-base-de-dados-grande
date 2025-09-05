FROM python:3.11-slim

WORKDIR /app

# Instala dependências do sistema
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copia arquivos de dependências
COPY requirements.txt .

# Instala dependências Python
RUN pip install --no-cache-dir -r requirements.txt

# Copia código da aplicação
COPY . .

# Cria diretório para ChromaDB se não existir
RUN mkdir -p ./chroma_db

# Expõe a porta
EXPOSE 10000

# Comando para iniciar a aplicação
CMD ["python", "app.py"]