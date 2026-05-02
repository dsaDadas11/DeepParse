param(
    [string]$UserId = "u_student_finance_demo",
    [string]$ContainerName = "deepparse_api",
    [switch]$SkipGeneration,
    [switch]$NoBuild,
    [switch]$ForceRebuild,
    [switch]$CheckOnly,
    [switch]$NoPause
)

$ErrorActionPreference = "Stop"
$scriptExitCode = 0

try {
    $projectRoot = $PSScriptRoot
    $backendDir = Join-Path $projectRoot "backend"
    $frontendDir = Join-Path $projectRoot "frontend"
    $backendEnvPath = Join-Path $backendDir ".env"
    $backendEnvExamplePath = Join-Path $backendDir ".env.example"
    $frontendEnvPath = Join-Path $frontendDir ".env"
    $frontendEnvExamplePath = Join-Path $frontendDir ".env.example"

    $hostRetrievalReportPath = Join-Path $backendDir "app\eval\results\student_retrieval_compare.json"
    $hostGenerationReportPath = Join-Path $backendDir "app\eval\results\student_generation_eval.json"
    $hostCorpusStatePath = Join-Path $backendDir "app\eval\results\student_corpus_state.json"

    $containerCorpusPath = "/app/sample_data/pdfs"
    $containerRetrievalCasesPath = "/app/eval/resume_retrieval_benchmark_manual_v2.json"
    $containerGenerationCasesPath = "/app/eval/resume_generation_benchmark_manual_v2.json"
    $containerRetrievalOutputPath = "/app/eval/results/student_retrieval_compare.json"
    $containerGenerationOutputPath = "/app/eval/results/student_generation_eval.json"
    $baselineMode = "generic_static_hybrid_tuned"

    function Format-Rate([double]$value) {
        return "{0:P1}" -f $value
    }

    function Format-Points([double]$value) {
        return "{0:+0.0;-0.0;0.0}pp" -f ($value * 100.0)
    }

    function Format-Score([double]$value) {
        return "{0:0.00}" -f $value
    }

    function Format-PercentPoint([double]$value) {
        return "{0:0.0}" -f ($value * 100.0)
    }

    function Copy-IfMissing([string]$sourcePath, [string]$targetPath) {
        if (Test-Path -LiteralPath $targetPath) {
            return
        }
        if (-not (Test-Path -LiteralPath $sourcePath)) {
            Write-Host "WARNING: Source file not found: $sourcePath" -ForegroundColor Yellow
            return
        }
        Copy-Item -LiteralPath $sourcePath -Destination $targetPath -Force
        Write-Host "Created: $targetPath" -ForegroundColor Green
    }

    function Get-EnvValue([string]$path, [string]$name) {
        if (-not (Test-Path -LiteralPath $path)) {
            return ""
        }

        $pattern = "^\s*" + [regex]::Escape($name) + "\s*=\s*(.*)\s*$"
        foreach ($line in Get-Content -LiteralPath $path) {
            if ($line -match $pattern) {
                return $matches[1].Trim().Trim("'`"")
            }
        }

        return ""
    }

    function Assert-Configured([string]$path, [string[]]$requiredKeys) {
        $missing = @()
        foreach ($key in $requiredKeys) {
            $value = Get-EnvValue -path $path -name $key
            if ([string]::IsNullOrWhiteSpace($value) -or $value -like "your_*") {
                $missing += $key
            }
        }

        if ($missing.Count -gt 0) {
            throw "Please update $path before running. Missing values: $($missing -join ', ')"
        }
    }

    function Ensure-FrontendEnv([string]$targetPath, [string]$apiBase, [string]$viewerId) {
        $content = @(
            "VITE_TITLE=DeepParse"
            "VITE_API_BASE=$apiBase"
            "VITE_FORCE_VIEWER_ID=$viewerId"
        )
        Set-Content -LiteralPath $targetPath -Value $content -Encoding UTF8
    }

    function Test-DockerRunning() {
        try {
            $null = docker info 2>$null
            return $LASTEXITCODE -eq 0
        }
        catch {
            return $false
        }
    }

    function Test-ContainerRunning([string]$name) {
        $result = docker ps --filter "name=$name" --filter "status=running" --format "{{.Names}}" 2>$null
        return ($result -eq $name)
    }

    function Test-ContainerExists([string]$name) {
        $result = docker ps -a --filter "name=$name" --format "{{.Names}}" 2>$null
        return ($result -eq $name)
    }

    function Wait-For-Container([string]$name, [int]$maxAttempts = 60) {
        for ($attempt = 1; $attempt -le $maxAttempts; $attempt++) {
            docker exec $name python -c "print('ready')" *> $null
            if ($LASTEXITCODE -eq 0) {
                return
            }
            Write-Host "Waiting for container $name... ($attempt/$maxAttempts)" -ForegroundColor Yellow
            Start-Sleep -Seconds 5
        }
        throw "Container $name did not become ready in time."
    }

    function Initialize-Database([string]$name) {
        docker exec $name python -c "from utils.database import init_db; init_db()"
        if ($LASTEXITCODE -ne 0) {
            throw "Database initialization failed."
        }
    }

    function Run-Command([string]$label, [scriptblock]$action) {
        Write-Host ""
        Write-Host $label -ForegroundColor Cyan
        & $action
        if ($LASTEXITCODE -ne 0) {
            throw "$label failed with exit code $LASTEXITCODE"
        }
    }

    function Invoke-ComposeUp([string]$backendDir, [bool]$noBuild) {
        $existingImage = docker images "backend-deepparse_api" --format "{{.Repository}}" 2>$null
        $hasLocalImage = $existingImage -eq "backend-deepparse_api"

        Push-Location $backendDir
        try {
            if ($hasLocalImage -and -not $noBuild) {
                Write-Host "Found existing local image 'backend-deepparse_api'. Using it without rebuild..." -ForegroundColor Green
                Write-Host "To force rebuild, use -NoBuild parameter." -ForegroundColor Yellow
                $composeArgs = @("compose", "up", "-d", "--no-build")
            } else {
                $composeArgs = @("compose", "up", "-d")
                if (-not $noBuild) {
                    $composeArgs += "--build"
                }
            }
            docker @composeArgs
            if ($LASTEXITCODE -ne 0) {
                throw "docker compose up failed with exit code $LASTEXITCODE"
            }
        }
        finally {
            Pop-Location
        }
    }

    function Invoke-CorpusBuild([string]$label, [scriptblock]$action, [string]$name, [string]$userId, [string]$sourceDir) {
        Write-Host ""
        Write-Host $label -ForegroundColor Cyan
        & $action

        $exitCode = $LASTEXITCODE
        if ($exitCode -eq 0) {
            return
        }

        Write-Host "WARNING: Build command returned exit code $exitCode. Re-checking corpus state..." -ForegroundColor Yellow
        $postState = Get-CorpusState -name $name -userId $userId -sourceDir $sourceDir
        Save-CorpusState -state $postState -path $hostCorpusStatePath
        Show-CorpusState -state $postState

        if ($postState.action -eq "skip") {
            Write-Host "Corpus is complete despite the Docker exec error. Continuing to evaluation." -ForegroundColor Yellow
            return
        }

        throw "$label failed with exit code $exitCode"
    }

    function Save-CorpusState([object]$state, [string]$path) {
        $parentDir = Split-Path -Path $path -Parent
        if (-not (Test-Path -LiteralPath $parentDir)) {
            New-Item -ItemType Directory -Force -Path $parentDir | Out-Null
        }

        $payload = [ordered]@{
            checked_at            = (Get-Date).ToString("o")
            user_id               = [string]$state.user_id
            source_dir            = [string]$state.source_dir
            source_file_count     = [int]$state.source_file_count
            completed_file_count  = [int]$state.completed_file_count
            missing_file_count    = [int]$state.missing_file_count
            extra_db_file_count   = [int]$state.extra_db_file_count
            index_exists          = [bool]$state.index_exists
            doc_count             = [int]$state.doc_count
            action                = [string]$state.action
            reason                = [string]$state.reason
            missing_files         = @($state.missing_files)
            extra_db_files        = @($state.extra_db_files)
        }

        $payload | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $path -Encoding UTF8
    }

    function Get-CorpusState([string]$name, [string]$userId, [string]$sourceDir) {
        $maxAttempts = 3

        for ($attempt = 1; $attempt -le $maxAttempts; $attempt++) {
            $rawOutput = docker exec $name python /app/eval/inspect_corpus_state.py --user-id $userId --source-dir $sourceDir 2>&1
            $exitCode = $LASTEXITCODE
            $textOutput = ($rawOutput | Out-String).Trim()

            if ($exitCode -eq 0) {
                $jsonLine = $textOutput
                if ($textOutput -match "\r?\n") {
                    $jsonLine = (
                        $textOutput -split "\r?\n" |
                        Where-Object { -not [string]::IsNullOrWhiteSpace($_) } |
                        Select-Object -Last 1
                    )
                }

                try {
                    return $jsonLine | ConvertFrom-Json
                }
                catch {
                    throw "Failed to parse corpus state JSON for user '$userId'. Raw output: $textOutput"
                }
            }

            if ($attempt -lt $maxAttempts) {
                Write-Host "Corpus state inspection failed on attempt $attempt/$maxAttempts. Retrying in 5 seconds..." -ForegroundColor Yellow
                Start-Sleep -Seconds 5
                continue
            }

            $details = if ([string]::IsNullOrWhiteSpace($textOutput)) {
                "No stderr/stdout was returned."
            } else {
                $textOutput
            }
            throw "Failed to inspect corpus state for user '$userId'. Details: $details"
        }
    }

    function Show-CorpusState([object]$state) {
        Write-Host ""
        Write-Host "========================================" -ForegroundColor Green
        Write-Host "Corpus Status" -ForegroundColor Green
        Write-Host "========================================" -ForegroundColor Green
        Write-Host ("  user_id             : {0}" -f $state.user_id)
        Write-Host ("  source_files        : {0}" -f $state.source_file_count)
        Write-Host ("  completed_files     : {0}" -f $state.completed_file_count)
        Write-Host ("  missing_files       : {0}" -f $state.missing_file_count)
        Write-Host ("  extra_db_files      : {0}" -f $state.extra_db_file_count)
        Write-Host ("  es_index_exists     : {0}" -f $state.index_exists)
        Write-Host ("  es_doc_count        : {0}" -f $state.doc_count)
        Write-Host ("  next_action         : {0}" -f $state.action)
        Write-Host ("  reason              : {0}" -f $state.reason)
        Write-Host ("  state_report        : {0}" -f $hostCorpusStatePath)
    }

    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "DeepParse Smart Benchmark Runner" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""

    $dockerVersion = docker --version 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "Docker is not installed or not in PATH. Please install Docker first."
    }
    Write-Host "Docker found: $dockerVersion" -ForegroundColor Green
    Write-Host ""

    Write-Host "Checking Docker daemon status..." -ForegroundColor Cyan
    if (-not (Test-DockerRunning)) {
        Write-Host ""
        Write-Host "========================================" -ForegroundColor Red
        Write-Host "DOCKER DAEMON IS NOT RUNNING!" -ForegroundColor Red
        Write-Host "========================================" -ForegroundColor Red
        Write-Host ""
        Write-Host "Please start Docker Desktop first:" -ForegroundColor Yellow
        Write-Host "  1. Open Docker Desktop from Start Menu" -ForegroundColor Yellow
        Write-Host "  2. Wait for the Docker icon to show 'Running' status" -ForegroundColor Yellow
        Write-Host "  3. Then run this script again" -ForegroundColor Yellow
        Write-Host ""
        throw "Docker daemon is not running. Please start Docker Desktop and try again."
    }
    Write-Host "Docker daemon is running!" -ForegroundColor Green
    Write-Host ""

    Copy-IfMissing -sourcePath $backendEnvExamplePath -targetPath $backendEnvPath
    Copy-IfMissing -sourcePath $frontendEnvExamplePath -targetPath $frontendEnvPath

    Write-Host "Checking configuration..." -ForegroundColor Cyan
    Assert-Configured -path $backendEnvPath -requiredKeys @(
        "GENERATION_API_KEY",
        "GENERATION_BASE_URL",
        "EMBEDDING_API_KEY",
        "EMBEDDING_BASE_URL"
    )
    Write-Host "Configuration OK" -ForegroundColor Green

    Ensure-FrontendEnv -targetPath $frontendEnvPath -apiBase "http://localhost:8000" -viewerId $UserId

    Write-Host ""
    Write-Host "Checking Docker container status..." -ForegroundColor Cyan
    $containerRunning = Test-ContainerRunning -name $ContainerName
    $containerExists = Test-ContainerExists -name $ContainerName

    if ($containerRunning) {
        Write-Host "Container '$ContainerName' is already running. Skipping docker compose up." -ForegroundColor Green
    } elseif ($containerExists) {
        Write-Host "Container '$ContainerName' exists but is not running. Starting it..." -ForegroundColor Yellow
        docker start $ContainerName
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Existing container failed to start. Recreating it from the current project directory..." -ForegroundColor Yellow
            docker rm -f $ContainerName *> $null
            if ($LASTEXITCODE -ne 0) {
                throw "Failed to remove stale container $ContainerName"
            }
            Invoke-ComposeUp -backendDir $backendDir -noBuild ([bool]$NoBuild)
        }
    } else {
        Write-Host "Container '$ContainerName' does not exist. Starting Docker services..." -ForegroundColor Yellow
        Invoke-ComposeUp -backendDir $backendDir -noBuild ([bool]$NoBuild)
    }

    Write-Host ""
    Write-Host "Waiting for container to be ready..." -ForegroundColor Cyan
    Wait-For-Container -name $ContainerName
    Write-Host "Container is ready!" -ForegroundColor Green

    Write-Host ""
    Write-Host "Initializing database..." -ForegroundColor Cyan
    Initialize-Database -name $ContainerName
    Write-Host "Database initialized!" -ForegroundColor Green

    $corpusState = Get-CorpusState -name $ContainerName -userId $UserId -sourceDir $containerCorpusPath
    if ($ForceRebuild) {
        $corpusState.action = "reset"
        $corpusState.reason = "ForceRebuild was requested."
    }
    Save-CorpusState -state $corpusState -path $hostCorpusStatePath
    Show-CorpusState -state $corpusState

    if ($CheckOnly) {
        Write-Host ""
        Write-Host "CheckOnly mode enabled. Exiting without rebuild or evaluation." -ForegroundColor Yellow
        return
    }

    switch ($corpusState.action) {
        "skip" {
            Write-Host ""
            Write-Host "STEP 1/3: Corpus already prepared. Skipping rebuild." -ForegroundColor Green
        }
        "resume" {
            Write-Host ""
            Write-Host "========================================" -ForegroundColor Yellow
            Write-Host "STEP 1/3: Resuming demo corpus rebuild..." -ForegroundColor Yellow
            Write-Host "Detected partial progress. Only missing files will be processed." -ForegroundColor Yellow
            Write-Host "========================================" -ForegroundColor Yellow
            Invoke-CorpusBuild -label "Resuming PDF files..." -action {
                docker exec $ContainerName python /app/rebuild_user_corpus.py `
                    --user-id $UserId `
                    --source-dir $containerCorpusPath `
                    --resume `
                    --single-file-timeout-seconds 900
            } -name $ContainerName -userId $UserId -sourceDir $containerCorpusPath
        }
        "rebuild" {
            Write-Host ""
            Write-Host "========================================" -ForegroundColor Yellow
            Write-Host "STEP 1/3: Rebuilding demo corpus..." -ForegroundColor Yellow
            Write-Host "No completed rebuild detected. Processing all source PDF files." -ForegroundColor Yellow
            Write-Host "========================================" -ForegroundColor Yellow
            Invoke-CorpusBuild -label "Processing PDF files..." -action {
                docker exec $ContainerName python /app/rebuild_user_corpus.py `
                    --user-id $UserId `
                    --source-dir $containerCorpusPath `
                    --reset-first `
                    --single-file-timeout-seconds 900
            } -name $ContainerName -userId $UserId -sourceDir $containerCorpusPath
        }
        "reset" {
            Write-Host ""
            Write-Host "========================================" -ForegroundColor Yellow
            Write-Host "STEP 1/3: Rebuilding demo corpus from scratch..." -ForegroundColor Yellow
            Write-Host "Detected inconsistent state. Existing progress will be replaced." -ForegroundColor Yellow
            Write-Host "========================================" -ForegroundColor Yellow
            Invoke-CorpusBuild -label "Rebuilding PDF files from scratch..." -action {
                docker exec $ContainerName python /app/rebuild_user_corpus.py `
                    --user-id $UserId `
                    --source-dir $containerCorpusPath `
                    --reset-first `
                    --single-file-timeout-seconds 900
            } -name $ContainerName -userId $UserId -sourceDir $containerCorpusPath
        }
        Default {
            throw $corpusState.reason
        }
    }

    if ($corpusState.action -ne "skip") {
        $corpusState = Get-CorpusState -name $ContainerName -userId $UserId -sourceDir $containerCorpusPath
        Save-CorpusState -state $corpusState -path $hostCorpusStatePath
        Show-CorpusState -state $corpusState

        if ($corpusState.action -ne "skip") {
            throw "Corpus preparation did not finish successfully. Current action recommendation: $($corpusState.action)"
        }
    }

    Write-Host ""
    Write-Host "========================================" -ForegroundColor Yellow
    Write-Host "STEP 2/3: Running retrieval evaluation..." -ForegroundColor Yellow
    Write-Host "This will test 62 retrieval cases (approx 5-10 minutes)" -ForegroundColor Yellow
    Write-Host "========================================" -ForegroundColor Yellow
    Run-Command "Testing retrieval..." {
        docker exec $ContainerName python /app/eval/run_retrieval_compare.py `
            --user-id $UserId `
            --cases $containerRetrievalCasesPath `
            --baseline-mode $baselineMode `
            --output $containerRetrievalOutputPath
    }

    if (-not (Test-Path -LiteralPath $hostRetrievalReportPath)) {
        throw "Retrieval compare report not found: $hostRetrievalReportPath"
    }

    Write-Host "Waiting for report file to be ready..." -ForegroundColor Yellow
    Start-Sleep -Seconds 3

    try {
        $retrievalReport = Get-Content -LiteralPath $hostRetrievalReportPath -Raw -Encoding UTF8 | ConvertFrom-Json
    }
    catch {
        Write-Host "ERROR: Failed to parse retrieval report! Using Python fallback..." -ForegroundColor Red
        $retrievalReport = python -c "import json; print(json.dumps(json.load(open('$hostRetrievalReportPath','r',encoding='utf-8'))))" | ConvertFrom-Json
    }
    $currentSummary = $retrievalReport.current.summary
    $baselineSummary = $retrievalReport.PSObject.Properties[$baselineMode].Value.summary
    $deltaSummary = $retrievalReport.delta.summary

    $generationSummary = $null

    if (-not $SkipGeneration) {
        Write-Host ""
        Write-Host "========================================" -ForegroundColor Yellow
        Write-Host "STEP 3/3: Running generation evaluation..." -ForegroundColor Yellow
        Write-Host "This will test answer generation (approx 5-10 minutes)" -ForegroundColor Yellow
        Write-Host "========================================" -ForegroundColor Yellow
        Run-Command "Testing generation..." {
            docker exec $ContainerName python /app/eval/run_generation_eval.py `
                --user-id $UserId `
                --cases $containerGenerationCasesPath `
                --output $containerGenerationOutputPath `
                --model-timeout-seconds 120 `
                --max-attempts 5 `
                --retry-initial-seconds 10 `
                --retry-max-seconds 120
        }

        if (-not (Test-Path -LiteralPath $hostGenerationReportPath)) {
            throw "Generation report not found: $hostGenerationReportPath"
        }

        $generationReport = Get-Content -LiteralPath $hostGenerationReportPath -Raw -Encoding UTF8 | ConvertFrom-Json
        $generationSummary = $generationReport.summary

    }

    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "Project Outcomes" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ('  - Retrieval lift: Top-1 from {0} to {1}, MRR from {2} to {3}, Evidence@3 {4}' -f `
        (Format-Rate $baselineSummary.hit_at_1), `
        (Format-Rate $currentSummary.hit_at_1), `
        (Format-Score $baselineSummary.mrr), `
        (Format-Score $currentSummary.mrr), `
        (Format-Rate $currentSummary.evidence_hit_at_3))
    Write-Host ('  - Evidence localization: Evidence@1 +{0} pp, fixes "found the doc but not the number"' -f `
        (Format-PercentPoint $deltaSummary.evidence_hit_at_1))
    if ($generationSummary -ne $null) {
        Write-Host ('  - Hallucination control: cited answer rate {0}, abstain on out-of-KB questions {1}' -f `
            (Format-Rate $generationSummary.grounded_answer_rate), `
            (Format-Rate $generationSummary.abstain_success_rate))
    } else {
        Write-Host '  - Hallucination control: generation evaluation was skipped, so this metric was not refreshed.' -ForegroundColor Yellow
    }
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "All tasks completed successfully!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
}
catch {
    $scriptExitCode = 1
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Red
    Write-Host "ERROR occurred!" -ForegroundColor Red
    Write-Host "========================================" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    Write-Host ""
    Write-Host "Stack Trace:" -ForegroundColor Red
    Write-Host $_.ScriptStackTrace -ForegroundColor Red
}
finally {
    Write-Host ""
    if ($NoPause) {
        Write-Host "Exiting without pause because -NoPause was specified." -ForegroundColor Cyan
    } else {
        Write-Host "Press any key to exit..." -ForegroundColor Cyan
        $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    }
}

exit $scriptExitCode
