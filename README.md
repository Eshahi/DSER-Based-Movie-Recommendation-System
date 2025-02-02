# 🎬 DSER-Based Movie Recommendation System 🎯  

A **Deep-Sequential Embedding for Single-Domain Recommendation (DSER)-based** movie recommender system. The project is inspired by the research paper **["DSER: Deep-Sequential Embedding for Single Domain Recommendation"](https://www.sciencedirect.com/science/article/pii/S0957417422013306)** and implements a **hybrid recommendation model** leveraging **deep learning, Doc2Vec, collaborative filtering, and hybrid recommendation techniques** to provide personalized movie recommendations.  

## 🚀 Key Features  

- **Doc2Vec-Based User and Movie Embeddings**: Converts sequential user interactions into dense feature representations.  
- **Hybrid Recommendation Model**: Combines **Generalized Matrix Factorization (GMF)** and **Multi-Layer Perceptron (MLP)** to capture **linear and nonlinear** user–movie relationships.  
- **Large-Scale Movie Data Processing**: Utilizes **MovieLens** metadata and user ratings for training.  
- **Implicit Feedback Learning**: Transforms explicit user ratings into **implicit feedback** for better recommendation accuracy.  
- **Deep Learning Implementation**: Built using **TensorFlow/Keras**, **Gensim**, and **NumPy**.  

## 🛠️ Technologies Used  

- **Python**  
- **TensorFlow / Keras**  
- **Gensim (Doc2Vec)**  
- **NumPy & Pandas**  
- **Scikit-Learn**

## 📊 Dataset  

The system is trained using **MovieLens** dataset, which includes:  

- **Movies metadata** (titles, genres, keywords, cast, crew).  
- **User ratings** (implicit feedback from interactions).  
- **Links between movie IDs and external databases** (TMDB & IMDB).  

## 📖 Reference  

This project is based on the research paper:  

> Minsung Hong, Chulmo Koo, Namho Chung, "DSER: Deep-Sequential Embedding for Single Domain Recommendation,"  
> _Expert Systems With Applications_, Volume 208, 2022.  
> [DOI: 10.1016/j.eswa.2022.118156](https://www.sciencedirect.com/science/article/pii/S0957417422013306)  

## 🔥 Future Enhancements  

- **Implementing Attention Mechanisms** for improved sequence modeling.  
- **Extending to Multi-Domain Recommendations** (e.g., books, music).  
- **Deploying the Model as a Web API** for real-time recommendations.  
