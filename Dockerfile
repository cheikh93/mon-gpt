# Image de base avec Python
FROM python:3.11-slim

# Crée le dossier de travail dans le conteneur
WORKDIR /app

# Copie les fichiers de dépendances
COPY requirements.txt .

# Installe les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copie le reste des fichiers du projet
COPY . .

# Port exposé (Flask écoute sur 5000)
EXPOSE 5000

# Commande pour démarrer Flask
CMD ["python", "-m", "app.app"]
