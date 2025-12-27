#!/usr/bin/env python3
"""
Script de démarrage avec patch lzma
"""
# Patch critique pour lzma - DOIT être fait avant tout import
import sys
import types
import io as _io

# Créer un faux module lzma complet
fake_lzma = types.ModuleType('lzma')

# Classe LZMAFile factice qui hérite de io.BufferedIOBase
class FakeLZMAFile(_io.BufferedIOBase):
    def __init__(self, *args, **kwargs):
        pass
    def read(self, size=-1):
        return b''
    def write(self, data):
        return 0

fake_lzma.LZMAFile = FakeLZMAFile
fake_lzma.open = lambda *args, **kwargs: FakeLZMAFile()
fake_lzma.compress = lambda data: data
fake_lzma.decompress = lambda data: data

sys.modules['lzma'] = fake_lzma
sys.modules['_lzma'] = fake_lzma

# Maintenant on peut importer l'application
from app_flask import app

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🌌 Spatial Galaxy Detector - Interface Web")
    print("="*60)
    print("\nOuvrez votre navigateur à: http://localhost:5000")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
