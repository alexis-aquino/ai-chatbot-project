# AI Chatbot Project

## About This Project
This is my very first full coding project where I am learning how to build a simple **AI chatbot** step by step.  
The goal is to understand the basics of machine learning, natural language processing (NLP), and software development practices (GitHub, project structure, etc.).  

I’m treating this like a college learning project: professional enough to practice real dev workflows, but still at a beginner stage.



## Week 1 Progress
During the first week, I managed to:

- Set up the project folder structure (`src/`, `data/`, `notebooks/`)
- Learn Git & GitHub basics (init, commit, push)
- Write and organize JSON data for chatbot intents
- Build a preprocessing pipeline (tokenization, stopword removal, stemming)
- Train my first machine learning model (Logistic Regression classifier)
- Integrate everything into a **working chatbot MVP**

This means I can already type something like *“hi”* and get a chatbot response.

## Example Chat
You: hi
Bot: Hello! How can I help you?

You: thanks
Bot: You're welcome!

## Next Steps (Week 2 Goals)
- Learn Bag of Words vs TF-IDF vectorization
- Improve classification accuracy using TF-IDF
- Start experimenting with more advanced models

## Reflections
As a student, this is my first time working on a project that combines **coding + machine learning + GitHub workflow**.  
It feels rewarding to already have a working chatbot, even though it’s very simple.  
I can see how professional developers structure their code and why version control is so important.  

This project will grow as I continue learning each week.

Week 2 Progress – Classic ML Pipeline

Expanded dataset (~100+ patterns).

Added 10+ intents with multiple patterns & responses.

Built modular files:

dataset_loader.py → loads JSON intents.

preprocessing.py → text cleaning.

vectorizer.py → Bag-of-Words + TF-IDF.

label_encoder.py → intent label encoding.

Trained a scikit-learn pipeline (Logistic Regression).

Improved intent classification accuracy with TF-IDF.

First working chatbot integration tested.

Example (Week 2 ML version):

You: hello
Bot: Hi there!
You: bye
Bot: See you soon!

Week 3 Progress – Neural Network + Context

Installed TensorFlow/Keras.

Tokenized & padded input sequences with Keras Tokenizer.

Trained first Neural Network model for intent classification.

Saved models in multiple formats (.keras, .h5).

Implemented confidence threshold → fallback response for unclear inputs.

Added context memory with context_manager.py:

Bot can remember last intent and adjust response.

Makes conversations feel more natural.

Updated chatbot.py to prefer best_chatbot_model.keras.

Example (Week 3 Neural Net + Context):

You: can you help me?
Bot: Sure! Do you need help with coding?
You: yes
Bot: Awesome! What programming language are you learning?

Next Steps (Week 4 Plan)

Improve context handling (multi-turn conversations).

Add more intents & responses for richer chat.

Experiment with embeddings (Word2Vec, GloVe).

Deploy chatbot (maybe Flask or Streamlit).