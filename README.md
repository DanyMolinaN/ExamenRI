# Sistema de Recuperación de Información: NFCorpus con TF-IDF y BM25

**Asignatura:** ICCD753 — Recuperación de Información  
**Institución:** Escuela Politécnica Nacional  
**Estudiante:** Dany Molina  
**Fecha:** 18 de noviembre de 2026

---

## 1. Descripción del proyecto

Este proyecto implementa y compara dos modelos clásicos de recuperación de información sobre el corpus **NFCorpus**:

- **TF-IDF con similitud coseno**, usando `scikit-learn`.
- **BM25 (Okapi)**, usando `rank_bm25`.

El sistema carga el corpus y los juicios de relevancia desde Hugging Face, procesa el texto en inglés, construye los índices de búsqueda y evalúa ambos modelos mediante `Precision@10`, `Recall@10` y `NDCG@10`.

También incluye una API local desarrollada con Flask y la opción de publicar el servicio mediante ngrok. Las rutas implementadas son:

- `GET /health`
- `POST /search`

---

## 2. Objetivos

- Cargar y explorar el corpus NFCorpus.
- Aplicar un flujo de preprocesamiento reproducible.
- Construir un índice invertido y un vocabulario.
- Implementar recuperación con TF-IDF y BM25.
- Comparar ambos modelos con consultas y qrels de evaluación.
- Visualizar comparativamente las métricas obtenidas.
- Exponer los modelos mediante una API web.

---

## 3. Tecnologías utilizadas

| Tecnología | Uso |
|---|---|
| Python | Lenguaje principal |
| Jupyter Notebook | Desarrollo y presentación experimental |
| Hugging Face `datasets` | Descarga de corpus, consultas y qrels |
| NLTK | Tokenización, stopwords y stemming |
| `scikit-learn` | TF-IDF y similitud coseno |
| `rank_bm25` | Implementación de BM25 |
| NumPy | Operaciones numéricas |
| Pandas | Manipulación auxiliar de datos |
| Matplotlib | Visualización de métricas |
| Flask | Servicio web REST |
| pyngrok / ngrok | Publicación opcional del servicio |

---

## 4. Corpus utilizado

Se utiliza el corpus [`BeIR/nfcorpus`](https://huggingface.co/datasets/BeIR/nfcorpus), disponible a través de Hugging Face.

| Componente | Cantidad |
|---|---:|
| Documentos | 3.633 |
| Consultas totales | 3.237 |
| Qrels de evaluación | 12.334 |
| Consultas de evaluación con qrels | 323 |
| Vocabulario | 27.869 términos |
| Longitud promedio de BM25 | 154,1 tokens |

La estructura principal de los datos es:

- **Corpus:** `_id`, `title`, `text`
- **Consultas:** `_id`, `title`, `text`
- **Qrels:** `query-id`, `corpus-id`, `score`

---

## 5. Flujo de procesamiento

### 5.1 Preprocesamiento

Para documentos y consultas se realizan los siguientes pasos:

1. Conversión a minúsculas.
2. Tokenización con `nltk.word_tokenize`.
3. Eliminación de caracteres no alfanuméricos.
4. Eliminación de palabras vacías en inglés.
5. Aplicación de stemming con `PorterStemmer`.

Ejemplo:

```text
Entrada:
Dietary fat intake and risk of cardiovascular diseases.

Resultado:
['dietari', 'fat', 'intak', 'risk', 'cardiovascular', 'diseas']
```

### 5.2 Índice invertido

Se construye una lista invertida que relaciona cada término con los documentos en los que aparece y su frecuencia dentro de cada documento.

Ejemplo simplificado:

```text
diet -> [(7, 1), (11, 1), (16, 1), (20, 6), (24, 1)]
```

---

## 6. Modelos implementados

### 6.1 TF-IDF

TF-IDF pondera los términos según su frecuencia en el documento y su rareza en la colección. La matriz generada tiene la forma:

```text
(3633 documentos, 27869 términos)
Densidad: 0.3457 %
```

La recuperación se realiza con:

```python
cosine_similarity(query_vector, tfidf_matrix)
```

**Ventajas:** recuperación rápida y adecuada para coincidencias exactas de términos.  
**Limitación:** no normaliza la influencia por longitud del documento.

### 6.2 BM25

BM25 se implementó con:

```python
BM25Okapi(tokenized, k1=1.5, b=0.75)
```

- `k1 = 1.5`: controla la saturación de la frecuencia de términos.
- `b = 0.75`: controla la normalización según la longitud del documento.

**Ventaja:** ajusta mejor la frecuencia de términos y la longitud variable de los documentos biomédicos.

---

## 7. Estructura del proyecto

```text
Examen1bimDanyMolina/
├── Examen1bimDanyMolina.ipynb
├── examen1bim.ipynb
├── metricas_comparacion.png
├── README.md
└── Repaso/
    ├── README.md
    ├── Repaso 3.ipynb
    ├── RepasoDany.ipynb
    ├── Respas2.ipynb
    ├── examen1bim.ipynb
    ├── requirements.txt
    └── datasets/
```

El notebook principal es `Examen1bimDanyMolina.ipynb`.

---

## 8. Requisitos

Se recomienda Python 3.9 o superior. El entorno donde se obtuvo la salida adjunta utilizó Python 3.13.

Las dependencias principales son:

```text
datasets
nltk
scikit-learn
rank_bm25
numpy
pandas
matplotlib
flask
flask-ngrok
pyngrok
jupyter
gunicorn
```

---

## 9. Instalación

Desde la terminal de Windows:

```powershell
cd "C:\Users\HP VICTUS\Documents\Recuperacion de la Informacion\Examen1bimDanyMolina"

python -m venv .venv
.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install datasets nltk scikit-learn rank_bm25 numpy pandas matplotlib flask flask-ngrok pyngrok jupyter
```

Si la política de PowerShell impide activar el entorno, puede utilizar:

```powershell
.venv\Scripts\activate.bat
```

---

## 10. Ejecución del notebook

Inicie Jupyter desde la raíz del proyecto:

```powershell
jupyter notebook
```

Abra `Examen1bimDanyMolina.ipynb` y ejecute las celdas en orden.

En la primera ejecución, el proceso puede tardar al descargar el corpus. NLTK descarga además los recursos `punkt`, `punkt_tab` y `stopwords`.

---

## 11. API local con Flask

El notebook define una aplicación Flask con dos endpoints.

### 11.1 Estado del servicio

```http
GET /health
```

Respuesta esperada:

```json
{
  "status": "ok",
  "corpus": "NFCorpus",
  "docs": 3633,
  "models": ["tfidf", "bm25"]
}
```

### 11.2 Recuperación de documentos

```http
POST /search
Content-Type: application/json
```

Ejemplo:

```json
{
  "query": "dietary fat cardiovascular",
  "model": "bm25",
  "top_k": 10
}
```

Campos admitidos:

| Campo | Tipo | Valor predeterminado | Restricciones |
|---|---|---|---|
| `query` | string | obligatorio | No puede estar vacío |
| `model` | string | `bm25` | `bm25` o `tfidf` |
| `top_k` | entero | `10` | Entre 1 y 1000 |

---

## 12. Ejemplos de uso

### Comprobación local

```powershell
curl http://127.0.0.1:5000/health
```

### Búsqueda con BM25

```powershell
curl -X POST http://127.0.0.1:5000/search `
  -H "Content-Type: application/json" `
  -d '{"query":"dietary fat cardiovascular","model":"bm25","top_k":10}'
```

### Búsqueda con TF-IDF

```powershell
curl -X POST http://127.0.0.1:5000/search `
  -H "Content-Type: application/json" `
  -d '{"query":"cancer treatment vitamin","model":"tfidf","top_k":5}'
```

### Publicación con ngrok

La celda de ejecución del notebook requiere un token válido de ngrok:

```python
NGROK_TOKEN = "YOUR_NGROK_TOKEN"
```

Configure allí un token obtenido desde [ngrok Dashboard](https://dashboard.ngrok.com/get-started/your-authtoken). Nunca publique el token real dentro de repositorios o capturas.

## 12.1 Despliegue en Render

El servicio web para Render está separado del notebook en `app.py`. Render instalará
las dependencias indicadas en `requirements.txt`, descargará NFCorpus y construirá
los índices TF-IDF y BM25 al iniciar el servicio. Por eso, el primer arranque puede
tardar más que los siguientes.

1. Sube el proyecto a un repositorio de GitHub.
2. En Render, selecciona **New > Blueprint** y conecta ese repositorio.
3. Render detectará `render.yaml`; confirma la creación del servicio
   `nfcorpus-search-api`.
4. Cuando el despliegue termine, prueba `https://<tu-servicio>.onrender.com/health`
   y envía las consultas a `https://<tu-servicio>.onrender.com/search`.

No configures un token de ngrok para Render. El servicio usa el puerto asignado por
Render y el endpoint `/health` como comprobación de disponibilidad.

---

## 13. Evaluación experimental

Las métricas se calcularon sobre las 323 consultas test que tienen qrels y promedian los resultados por consulta.

| Métrica | TF-IDF | BM25 |
|---|---:|---:|
| `Precision@10` | 0.2232 | 0.2319 |
| `Recall@10` | 0.1418 | 0.1429 |
| `NDCG@10` | 0.2949 | 0.3165 |

![Comparación TF-IDF frente a BM25](metricas_comparacion.png)

BM25 obtuvo mejores resultados en las tres métricas. La diferencia fue más visible en `NDCG@10`, donde alcanzó `0.3165` frente a `0.2949` de TF-IDF.

---

## 14. Interpretación de las métricas

- **Precision@10:** proporción de documentos relevantes entre los diez primeros resultados.
- **Recall@10:** proporción de documentos relevantes encontrados dentro de los diez primeros resultados.
- **NDCG@10:** calidad del ranking considerando la posición y el nivel de relevancia de cada documento.

Los documentos recuperados se consideran relevantes cuando su score de qrel es mayor o igual a 1.

---

## 15. Resultados observados

Para la consulta:

```text
Do Cholesterol Statin Drugs Cause Breast Cancer?
```

Ambos modelos recuperaron documentos altamente relevantes relacionados con estatinas y cáncer de mama. TF-IDF obtuvo coincidencias exactas por ponderación vectorial, mientras que BM25 destacó documentos relevantes adicionales gracias a su normalización por longitud y saturación de frecuencia.

La ejecución adjunta confirmó 3.633 documentos indexados, 27.869 términos y una longitud promedio BM25 de 154,1 tokens.

---

## 16. Solución de problemas

### El servicio de ngrok muestra autenticación fallida

Si aparece `ERR_NGROK_105`, comprueba que:

1. El token configurado sea válido y vigente.
2. El token haya sido creado en tu cuenta de ngrok.
3. No hayas dejado el valor literal `YOUR_NGROK_TOKEN`.
4. No hayas compartido públicamente el token real.

El servicio Flask local permanece disponible en `http://127.0.0.1:5000` aunque falle el túnel externo.

### No se encuentra el corpus

Verifica tu conexión a internet y que las dependencias de `datasets` estén instaladas.

### Faltan recursos de NLTK

Ejecuta dentro del notebook:

```python
import nltk
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
```

### El servidor Flask no responde

Comprueba que se haya ejecutado la celda que inicia Flask y que el puerto `5000` no esté ocupado.

---

## 17. Conclusiones

1. El corpus NFCorpus contiene 3.633 documentos adecuados para una comparación experimental a escala académica.
2. El preprocesamiento reduce variantes mediante normalización y stemming.
3. TF-IDF ofrece una representación vectorial eficiente basada en pesos y similitud coseno.
4. BM25 supera ligeramente a TF-IDF en las tres métricas evaluadas.
5. El servicio web permite consultar ambos modelos de manera uniforme mediante JSON.
6. BM25 es el modelo predeterminado de la API por su mejor desempeño en esta evaluación.

---

## 18. Referencias

- BeIR. *NFCorpus Dataset*. Hugging Face: `BeIR/nfcorpus`.
- BeIR. *NFCorpus Qrels*. Hugging Face: `BeIR/nfcorpus-qrels`.
- Manning, C. D., Raghavan, P., & Schütze, H. *Introduction to Information Retrieval*. Cambridge University Press.
- Robertson, S. & Zaragoza, H. *The Probabilistic Relevance Framework: BM25 and Beyond*. Foundations and Trends in Information Retrieval.
