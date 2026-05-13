# One-command bootstrap for LiveStrat on Windows.
#
# What it does, in order:
#   1. Checks that Python and Docker are installed.
#   2. Copies .env.example to .env if you have not made a .env yet.
#   3. Starts the Postgres container that docker compose defines.
#   4. Installs the pinned Python dependencies for both the Flask app
#      and the analytics pipeline.
#   5. Creates the four user tables in Postgres (flask init-db).
#   6. Runs the unittest suite so you can see green ticks before
#      launching the app.
#
# Run it from the project root:
#   powershell -ExecutionPolicy Bypass -File .\setup.ps1

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSCommandPath
Set-Location $ProjectRoot

function Write-Step($message) {
    Write-Host ""
    Write-Host "[setup] $message" -ForegroundColor Cyan
}

function Require-Command($name, $hint) {
    $cmd = Get-Command $name -ErrorAction SilentlyContinue
    if (-not $cmd) {
        Write-Host "[setup] Missing required command: $name" -ForegroundColor Red
        Write-Host "        $hint" -ForegroundColor Yellow
        exit 1
    }
}

# Confirm the two tools the rest of the script depends on are on the PATH.
Write-Step "Verifying prerequisites"
Require-Command "python" "Install Python 3.10+ from https://www.python.org/downloads/"
Require-Command "docker" "Install Docker Desktop from https://www.docker.com/products/docker-desktop/"
$pythonVersion = (& python --version) 2>&1
Write-Host "[setup] Python found: $pythonVersion"

# If the user has not made their own .env yet, seed one from the example file.
Write-Step "Bootstrapping .env"
if (-not (Test-Path ".env")) {
    if (Test-Path ".env.example") {
        Copy-Item ".env.example" ".env"
        Write-Host "[setup] Copied .env.example to .env. Edit secrets before running in production."
    } else {
        Write-Host "[setup] No .env.example found, skipping." -ForegroundColor Yellow
    }
} else {
    Write-Host "[setup] .env already present, leaving untouched."
}

# Start the Postgres container in the background. The sleep gives it a few
# seconds to be ready for connections before flask init-db runs.
Write-Step "Starting PostgreSQL via docker compose"
docker compose up -d postgres
Start-Sleep -Seconds 4

# Install the pinned Python packages. Both requirements files matter:
# requirements.txt for the Flask app, the pipeline one for everything else.
Write-Step "Installing pinned Python dependencies"
python -m pip install --upgrade pip --quiet
python -m pip install -r requirements.txt --quiet
python -m pip install -r analytics_pipeline\requirements.txt --quiet

# Create the four user tables (users, alert_preferences, saved_strategy_profiles,
# notification_events). Safe to run again if the tables already exist.
Write-Step "Initialising database tables"
$env:FLASK_APP = "app.py"
python -m flask init-db

# Run the test suite. All 25 tests should pass.
Write-Step "Running test suite"
python -m unittest discover -s tests -v

# Print the handful of commands you are most likely to want next.
Write-Step "Setup complete"
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Green
Write-Host "  python app.py               # start Flask"
Write-Host "  `$env:PYTHONPATH = 'analytics_pipeline'"
Write-Host "  python -m src.reports.build_report_figures   # regenerate the 7 report figures"
Write-Host "  python -m src.sentiment.validate_finbert     # confirm FinBERT inference path"
Write-Host ""
