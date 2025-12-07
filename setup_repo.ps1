# PowerShell script to set up and push to GitHub
# Make sure Git is installed first!

Write-Host "=== GitHub Repository Setup ===" -ForegroundColor Cyan
Write-Host ""

# Check if git is available
try {
    $gitVersion = git --version
    Write-Host "✓ Git found: $gitVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Git is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Git from: https://git-scm.com/download/win" -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "Step 1: Initializing Git repository..." -ForegroundColor Cyan
git init

Write-Host ""
Write-Host "Step 2: Adding files..." -ForegroundColor Cyan
git add .

Write-Host ""
Write-Host "Step 3: Creating initial commit..." -ForegroundColor Cyan
git commit -m "Initial commit: AI Answer Aggregator"

Write-Host ""
Write-Host "✓ Local repository initialized!" -ForegroundColor Green
Write-Host ""

Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Go to https://github.com/new and create a new repository" -ForegroundColor White
Write-Host "2. Name it (e.g., 'ai-answer-aggregator')" -ForegroundColor White
Write-Host "3. DO NOT initialize with README, .gitignore, or license" -ForegroundColor White
Write-Host "4. Copy the repository URL" -ForegroundColor White
Write-Host ""
Write-Host "Then run these commands (replace YOUR_USERNAME and REPO_NAME):" -ForegroundColor Yellow
Write-Host "  git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git" -ForegroundColor Cyan
Write-Host "  git branch -M main" -ForegroundColor Cyan
Write-Host "  git push -u origin main" -ForegroundColor Cyan
Write-Host ""

# Check if GitHub CLI is available
try {
    $ghVersion = gh --version
    Write-Host "GitHub CLI detected! Would you like to create the repo automatically? (y/n)" -ForegroundColor Green
    $response = Read-Host
    if ($response -eq 'y' -or $response -eq 'Y') {
        Write-Host ""
        Write-Host "Enter repository name (e.g., ai-answer-aggregator):" -ForegroundColor Cyan
        $repoName = Read-Host
        Write-Host "Enter repository description (optional):" -ForegroundColor Cyan
        $repoDesc = Read-Host
        Write-Host "Make it public? (y/n):" -ForegroundColor Cyan
        $isPublic = Read-Host
        
        $visibility = if ($isPublic -eq 'y' -or $isPublic -eq 'Y') { "--public" } else { "--private" }
        $descParam = if ($repoDesc) { "--description `"$repoDesc`"" } else { "" }
        
        Write-Host ""
        Write-Host "Creating GitHub repository..." -ForegroundColor Cyan
        gh repo create $repoName $visibility $descParam --source=. --remote=origin --push
        
        Write-Host ""
        Write-Host "✓ Repository created and pushed to GitHub!" -ForegroundColor Green
    }
} catch {
    Write-Host "GitHub CLI not found. Use manual steps above." -ForegroundColor Yellow
}

