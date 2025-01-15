# CRUD con Hugging Face y DistilBERT

Este proyecto implementa operaciones CRUD (Crear, Leer, Actualizar, Eliminar) sobre documentos utilizando el modelo DistilBERT de Hugging Face para obtener representaciones de texto (embeddings). Además, permite la búsqueda de documentos similares mediante la similitud de coseno entre los embeddings.

## Requisitos

Asegúrate de tener Python 3.x instalado, además de las siguientes bibliotecas:

- `torch`
- `transformers`
- `numpy`
- `scikit-learn`

Puedes instalar estas dependencias utilizando `pip`:

```bash
pip install torch transformers numpy scikit-learn
```

## Descripción del proyecto
el sistema permite crear, leer, actualizar y eliminar documentos, así como buscar documentos similares utilizando la similitud de coseno entre los embeddings. La interfaz de usuario se encuentra en la función `menu()`, que se encarga de recibir las operaciones del usuario y realizar las correspondientes acciones.

La función `create_document()` recibe un texto y lo almacena en la lista `documents`. La función `read_documents()` imprime los documentos almacenados en la lista, si no hay documentos, muestra un mensaje indicando que no hay documentos disponibles. La función `update_document()` recibe el índice del documento a actualizar y el nuevo texto, y actualiza el documento almacenado en la lista. La función `delete_document()` recibe el índice del documento a eliminar y elimina el documento almacenado en la lista.

## Funcionamiento del sistema

tokenizador y embeddings: utiliza el modelo DistilBERT de Hugging Face para obtener representaciones de texto (embeddings) y la función `tokenize()` de `transformers` para tokenizar el texto.

similaridad de coseno: utiliza la función `cosine_similarity()` de `sklearn.metrics.pairwise` para calcular la similitud de coseno entre los embeddings de los documentos.

crear documento: recibe un texto y lo almacena en la lista `documents`.

leer documentos: imprime los documentos almacenados en la lista, si no hay documentos, muestra un mensaje indicando que no hay documentos disponibles.

actualizar documento: recibe el índice del documento a actualizar y el nuevo texto, y actualiza el documento almacenado en la lista.

eliminar documento: recibe el índice del documento a eliminar y elimina el documento almacenado en la lista.

buscar documentos similares: recibe el texto a buscar y devuelve una lista de documentos similares, utilizando la similitud de coseno entre los embeddings de los documentos.

## Ejecución del proyecto

Para ejecutar el proyecto, sigue los siguientes pasos:

1. Abre una terminal o línea de comandos.
2. Ejecuta el siguiente comando para instalar las dependencias necesarias:

```bash
pip install torch transformers numpy scikit-learn
```

3. Ejecuta el siguiente comando para ejecutar el proyecto:

```bash
python crud.py
```

ejemplo de ejecución:

```bash
$ python crud_huggingface.py
Operaciones CRUD con Hugging Face
1. Crear documento
2. Leer documentos
3. Actualizar documento
4. Eliminar documento
5. Buscar documentos por similitud
6. Salir
Seleccione una opción (1/2/3/4/5/6): 1
Ingrese el texto para crear: Este es un ejemplo de documento.
Documento creado con éxito.

Seleccione una opción (1/2/3/4/5/6): 2
Texto: Este es un ejemplo de documento.
```

# Detalles tecnicos 

distbert: utiliza el modelo DistilBERT de Hugging Face para obtener representaciones de texto (embeddings) y la función `tokenize()` de `transformers` para tokenizar el texto.

cosine_similarity: utiliza la función `cosine_similarity()` de `sklearn.metrics.pairwise` para calcular la similitud de coseno entre los embeddings de los documentos.

# paso a paso 

1. instalar las dependencias 

```bash
 from transformers import DistilBertTokenizer, DistilBertModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
```



# configuracion la base de datos local

Se utiliza el sistema de bases de datos SQalquemist para crear la base de datos vectorial. Para ello, se debe crear una base de datos local en la ruta `/data  ` y se debe ejecutar el comando `python3 create_db.py` para crear la base de datos.


## instalacion de la bases de datos local 

La base de datos se instala utilizando el siguiente comando:

```bash
pip install -r requirements.txt

```

```python

## indexar los documentos

para indexar los documentos, se utiliza la función `add()` de la base de datos vectorial.

# funcionamiento de los embeddings

¿Qué son los embeddings?

Los embeddings son representaciones numéricas de los documentos que se utilizan para realizar operaciones de búsqueda y indexación en la base de datos vectorial. Los embeddings se utilizan para almacenar los documentos y para realizar operaciones de similitud de coseno entre ellos.

¿Cómo se obtienen los embeddings?

Los embeddings se obtienen utilizando el modelo DistilBERT de Hugging Face. El modelo DistilBERT es un modelo de lenguaje pre-entrenado que se utiliza para tareas de texto, como la clasificación de sentimientos, la generación de texto, la traducción automática, entre otras.

Para obtener los embeddings, se utiliza la función `encode()` de `transformers`. Esta función toma un texto y devuelve un objeto `torch.Tensor` que contiene los embeddings del texto.

# funcionamiento de la similitud de coseno

¿Qué es la similitud de coseno?

La similitud de coseno es una medida de la similitud entre dos vectores de alta dimensión. En este caso, los vectores son los embeddings de los documentos. La similitud de coseno se calcula utilizando la función `cosine_similarity()` de `sklearn.metrics.pairwise`.

ejemplo de código:

```python
from sklearn.metrics.pairwise import cosine_similarity

query_embedding = get_embedding(query_text).flatten()
doc_embedding = np.array(doc['embedding']).flatten()
similarity = cosine_similarity([query_embedding], [doc_embedding])
```

# paso a paso 

1. instalar las dependencias 

```
bash
 from transformers import DistilBertTokenizer, DistilBertModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
```

# ejemplos de uso

crear documento: recibe un texto y lo almacena en la lista `documents`.

```bash

ingrese el texto para crear: este es un ejemplo de documento.
documento creado con éxito.

textos disponibles:
1. este es un ejemplo de documento.
```

[Andres Montalvo](https://github.com/TakizawaXD)
