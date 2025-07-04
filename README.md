## MEDCON Chatbot

An AI-powered medical chatbot designed to provide accurate responses to health-related queries as part of the MEDCON healthcare system. Built with **LangChain**, **Hugging Face (Mistral-7B-Instruct)**, **FAISS**, and **Streamlit**, it leverages retrieval-augmented generation (RAG) for efficient context retrieval from medical documents.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Overview
MEDCON Chatbot is a component of the MEDCON healthcare system, aimed at delivering reliable answers to medical queries. It uses a RAG pipeline to retrieve relevant information from a FAISS vector store (populated with medical documents) and generates responses using Hugging Face’s Mistral-7B-Instruct model. The interactive Streamlit interface ensures a user-friendly experience with real-time query processing and chat history persistence.

## Features
- **Retrieval-Augmented Generation (RAG)**: Combines FAISS vector store with Sentence Transformers (`all-MiniLM-L6-v2`) for efficient document retrieval.
- **AI-Powered Responses**: Integrates Hugging Face’s Mistral-7B-Instruct model with custom prompt engineering for accurate medical answers.
- **Interactive UI**: Streamlit-based web interface for real-time query input and chat history display.
- **Robust Error Handling**: Manages API token issues and vector store loading errors for reliable performance.
- **Scalable Design**: Optimized for integration with other MEDCON modules (e.g., heart disease detection).

## Project Structure
MEDCON-CHAT_BOT/
├── medibot.py              # Main Streamlit application
├── connect_memory_with_LLM.py  # Script for LangChain and FAISS integration
├── Pipfile                 # Dependency management
├── Pipfile.lock            # Locked dependencies
├── README.md               # Project documentation
└── requirements.txt        # Python dependencies

text


**Note**: The FAISS vector store (`vectorstore/`) and `.env` file (containing `HF_TOKEN`) are excluded from version control for security and size reasons.

## Setup Instructions
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/usmancynosure/MEDCON-CHAT_BOT.git
   cd MEDCON-CHAT_BOT
Set Up a Virtual Environment:
bash

python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
Install Dependencies:
bash

pip install -r requirements.txt
Or, if using Pipenv:
bash

pip install pipenv
pipenv install
Set Up Environment Variables: Create a .env file in the project root:
text

HF_TOKEN=your_huggingface_api_token
Obtain a Hugging Face API token from Hugging Face.
Prepare FAISS Vector Store:
The FAISS database (vectorstore/) is required for document retrieval. If unavailable, recreate it using the script that generated it (not included in this repository).
Ensure the vector store uses Sentence Transformers (all-MiniLM-L6-v2) embeddings.
Run the Application:
bash

streamlit run medibot.py
Usage
Open the Streamlit app in your browser (typically at http://localhost:8501).
Enter a medical query (e.g., “How to cure cancer?”) in the chat input.
View the AI-generated response and source documents retrieved from the FAISS vector store.
Chat history is preserved during the session for continuous interaction.
Technologies Used
Python: Core programming language.
LangChain: For building the RAG pipeline and managing LLM interactions.
Hugging Face: Mistral-7B-Instruct model for response generation.
FAISS: Vector store for efficient document retrieval.
Streamlit: Web interface for user interaction.
Sentence Transformers: all-MiniLM-L6-v2 for document embeddings.
Pipenv: Dependency management.
Contributing
Contributions are welcome! To contribute:

## Fork the repository.
Create a new branch (git checkout -b feature/your-feature).
Make changes and commit (git commit -m "Add your feature").
Push to your branch (git push origin feature/your-feature).
Open a pull request on GitHub.
Please ensure code follows PEP 8 standards and includes relevant documentation.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Contact
Author: Usman Waris
Email: imosmanwaris.tech@gmail.com
GitHub: usmancynosure
LinkedIn: usman-waris-0a9b8c7d
