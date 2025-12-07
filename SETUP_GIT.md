# Setting Up Git and GitHub Repository

## Step 1: Install Git (if not already installed)

If Git is not installed on your system:

1. Download Git for Windows from: https://git-scm.com/download/win
2. Run the installer with default settings
3. Restart your terminal/PowerShell after installation

## Step 2: Configure Git (first time only)

```powershell
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

## Step 3: Initialize Git Repository

Navigate to your project directory and run:

```powershell
git init
git add .
git commit -m "Initial commit: AI Answer Aggregator"
```

## Step 4: Create GitHub Repository

### Option A: Using GitHub Website
1. Go to https://github.com/new
2. Repository name: `ai-answer-aggregator` (or your preferred name)
3. Description: "Python program that queries multiple AI models and combines their responses"
4. Choose Public or Private
5. **DO NOT** initialize with README, .gitignore, or license (we already have these)
6. Click "Create repository"

### Option B: Using GitHub CLI (if installed)
```powershell
gh repo create ai-answer-aggregator --public --source=. --remote=origin --push
```

## Step 5: Connect and Push

After creating the repository on GitHub, run:

```powershell
git remote add origin https://github.com/YOUR_USERNAME/ai-answer-aggregator.git
git branch -M main
git push -u origin main
```

Replace `YOUR_USERNAME` with your actual GitHub username.

## Alternative: Quick Setup Script

If you have Git installed, you can run these commands in sequence:

```powershell
git init
git add .
git commit -m "Initial commit: AI Answer Aggregator"
# Then create repo on GitHub and run:
# git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git
# git push -u origin main
```

