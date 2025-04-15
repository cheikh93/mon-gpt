# 🤖 GPT Mini - From Scratch to Kubernetes 🚀

Ce projet montre comment créer un modèle GPT minimal, l'entraîner, l'exposer via une API Flask, le conteneuriser avec Docker et le déployer dans un cluster Kubernetes.

---

## 🔧 Technologies utilisées

- Python 3.11
- PyTorch
- Flask
- Docker
- Kubernetes (via Docker Desktop ou Minikube)
- PowerShell / curl

---

## 📁 Structure du projet

mon-gpt/ ├── app/ # API Flask ├── data/ # Corpus d'entraînement ├── model/ # Modèle GPT ├── scripts/ # Tokenizer et dataset ├── Dockerfile # Image Docker ├── deployment.yaml # Déploiement Kubernetes ├── service.yaml # Service Kubernetes ├── train.py # Entraînement ├── test_api.py # Test d'API ├── README.md

---

## 🧠 Fonctionnement

1. 📚 Un corpus est lu depuis `data/corpus.txt`
2. 🔡 Il est tokenisé via un `SimpleTokenizer`
3. 🧠 Un mini modèle GPT est implémenté avec PyTorch
4. 🔁 Il est entraîné à prédire le mot suivant
5. 🌐 Il est exposé via une API `/generate` avec Flask
6. 🐳 Le projet est Dockerisé
7. ☸️ Il est déployé dans Kubernetes
8. 📩 Testé avec `curl` ou `Invoke-RestMethod`

---

## 🧪 Exemple d’appel à l’API

### Sous PowerShell :
```powershell
Invoke-RestMethod -Uri "http://localhost:5000/generate" -Method POST -Body '{"prompt":"Le chat"}' -ContentType "application/json"
## Déploiement :
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml

##
Accès à l’API :
http://localhost:30007/generate
## 
👤 Auteur
Cheikh Mbacke Diouf 👨‍💻
Projet personnel pour maîtriser les bases d’un LLM (Large Language Model) et les outils modernes de déploiement (Docker, Kubernetes).
