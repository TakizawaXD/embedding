# Importar las bibliotecas necesarias
from transformers import DistilBertTokenizer, DistilBertModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

documents = []

def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy()
    return embedding

def create_document(text):
    embedding = get_embedding(text)
    document = {
        'text': text,
        'embedding': embedding.tolist()
    }
    documents.append(document)
    print("Documento creado con éxito.")

def read_documents():
    if not documents:
        print("No hay documentos disponibles.")
        return
    for doc in documents:
        print(f"Texto: {doc['text']}")
def update_document(doc_index, new_text):
    if 0 <= doc_index < len(documents):
        new_embedding = get_embedding(new_text)
        documents[doc_index] = {'text': new_text, 'embedding': new_embedding.tolist()}
        print("Documento actualizado con éxito.")
    else:
        print("Índice de documento no válido.")

def delete_document(doc_index):
    if 0 <= doc_index < len(documents):
        del documents[doc_index]
        print("Documento eliminado con éxito.")
    else:
        print("Índice de documento no válido.")

def search_similar_documents(query_text):
    query_embedding = get_embedding(query_text).flatten()
    similarities = []
    for idx, doc in enumerate(documents):
        doc_embedding = np.array(doc['embedding']).flatten()
        similarity = cosine_similarity([query_embedding], [doc_embedding])
        similarities.append((idx, doc['text'], similarity[0][0]))
    similarities.sort(key=lambda x: x[2], reverse=True)
    
    print("Documentos más similares:")
    for doc_index, text, sim in similarities[:5]:  
        print(f"Índice: {doc_index}, Texto: {text}, Similitud: {sim:.4f}")
def menu():
    while True:
        print("\nOperacion CRUD con Hugging Face by Andres Montalvo")
        print("1. Crear documento")
        print("2. Leer documentos")
        print("3. Actualizar documento")
        print("4. Eliminar documento")
        print("5. Buscar documentos por similitud")
        print("6. Salir")
        try:
            option = int(input("Seleccione una opción (1/2/3/4/5/6): "))
            
            if option == 1:
                text = input("Ingrese el texto para crear: ")
                create_document(text)
            
            elif option == 2:
                read_documents()
            
            elif option == 3:
                doc_index = int(input("Ingrese el índice del documento a actualizar: "))
                new_text = input("Ingrese el nuevo texto: ")
                update_document(doc_index, new_text)
            
            elif option == 4:
                doc_index = int(input("Ingrese el índice del documento a eliminar: "))
                delete_document(doc_index)
            
            elif option == 5:
                query_text = input("Ingrese el texto para buscar documentos similares: ")
                search_similar_documents(query_text)
            
            elif option == 6:
                print("Saliendo...")
                break
            
            else:
                print("Opción no válida. Por favor, seleccione una opción válida.")
        
        except Exception as e:
            print(f"Error: {e}")
menu()
