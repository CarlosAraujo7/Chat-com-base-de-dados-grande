#!/bin/bash

# Script de inicialização para o Render
echo "=== INICIANDO LLM DO AMOR ==="

# Verifica se os arquivos necessários existem
if [ ! -f "conversa_formatada.json" ]; then
    echo "❌ ERRO: conversa_formatada.json não encontrado!"
    exit 1
fi

# Verifica se a API key está configurada
if [ -z "$GOOGLE_API_KEY" ]; then
    echo "❌ ERRO: GOOGLE_API_KEY não configurada!"
    exit 1
fi

echo "✅ Arquivos necessários encontrados"
echo "✅ Variáveis de ambiente configuradas"

# Inicia a aplicação
echo "🚀 Iniciando aplicação..."
python app.py