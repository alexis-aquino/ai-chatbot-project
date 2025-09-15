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

Project Structure
ai-chatbot-project/
│
├── data/
│ └── intents.json # Training data (intents, patterns, responses)
│
├── src/
│ ├── preprocessing.py # Text cleaning and preprocessing
│ ├── train_intent_classifier.py # ML model training
│ └── chatbot.py # Chatbot integration (MVP)
│
├── model.pkl # Saved model
├── README.md # Project documentation
└── requirements.txt # (to be added later)

## Example Chat
You: hi
Bot: Hello! How can I help you?

You: thanks
Bot: You're welcome!


---

## Next Steps (Week 2 Goals)
- Learn Bag of Words vs TF-IDF vectorization
- Improve classification accuracy using TF-IDF
- Start experimenting with more advanced models

---

## Reflections
As a student, this is my first time working on a project that combines **coding + machine learning + GitHub workflow**.  
It feels rewarding to already have a working chatbot, even though it’s very simple.  
I can see how professional developers structure their code and why version control is so important.  

This project will grow as I continue learning each week.

