#!/bin/bash
# Script de limpeza para Git LFS
# Gerado automaticamente

echo "🧹 Limpando cache do Git..."

# Remove arquivos do cache
git rm --cached "chroma_db\chroma.sqlite3" 2>/dev/null
git rm --cached "chroma_db\ce3fd1c7-a054-474d-a1df-0ff19551e11c\data_level0.bin" 2>/dev/null

echo "✅ Cache limpo!"

echo "📋 Adicionando .gitattributes..."
git add .gitattributes
git commit -m "Configure Git LFS for ChromaDB"

echo "📦 Adicionando arquivos com LFS..."
git add chroma_db/
git add *.json
git commit -m "Add ChromaDB files with Git LFS"

echo "🚀 Fazendo push..."
git push origin main

echo "✅ Deploy concluído!"
