Planetary Protection Pioneers

Overview

Building a Retrieval-Augmented Generation (RAG) bot can significantly enhance the capabilities of a language model by incorporating external knowledge to generate more accurate and contextually relevant responses. This guide will walk you through creating a RAG bot specifically focused on planetary protection practices, leveraging Gradio and the Hugging Face APIs.

How RAG Enhances LLM Performance
RAG improves the performance of language models by augmenting them with external documents. This method retrieves relevant documents based on the user query and combines them with the original prompt before passing them to the language model for response generation. This approach ensures that the language model can access up-to-date and domain-specific information without the need for extensive retraining.

Steps in RAG
Input: The user inputs a query.
Indexing: Documents are indexed by chunking, generating embeddings, and storing them in a vector database.
Retrieval: Relevant documents are retrieved by comparing the query against the indexed vectors.
Generation: Retrieved documents are combined with the original prompt, and the combined text is passed to the model for response generation.
In the example provided, using the model directly fails to respond to the question due to a lack of knowledge of current events. On the other hand, when using RAG, the system can pull the relevant information needed for the model to answer the question appropriately.

Building the Planetary Protection Pioneers Chatbot
We will use the Zephyr LLM model and the all-MiniLM-L6-v2 sentence transformer model. This sentence-transformer model maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for tasks like clustering or semantic search.

Requirements
A PDF as your knowledge base: In this case, Planetary_Protection_Pioneers.pdf.
A requirements.txt file: To list all dependencies.
An app.py file: The main application code.
An account on Hugging Face: To access and use their models.
