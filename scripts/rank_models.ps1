param(
    [string]$ApiUrl = "http://127.0.0.1:8000/predict/upload",
    [string]$CheckpointsDir = "C:\Projects\Thesis\BrainTumerModels\models\checkpoints",
    [string]$ClassificationImage = "C:\Projects\Thesis\BrainTumerModels\data\brisc2025\classification_task\test\glioma\brisc2025_test_00001_gl_ax_t1.jpg",
    [string]$SegmentationImage = "C:\Projects\Thesis\BrainTumerModels\data\brisc2025\segmentation_task\test\images\brisc2025_test_00001_gl_ax_t1.jpg",
    [string]$OutputRoot = "C:\Projects\Thesis\BrainTumerModels\outputs\comparisons",
    [string]$PythonExe = "C:/Users/maria/AppData/Local/Programs/Python/Python313/python.exe",
    [double]$SegmentationThreshold = 0.3
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-ComparableNumber {
    param([object]$Value)
    if ($null -eq $Value) {
        return -1000000000.0
    }

    try {
        return [double]$Value
    }
    catch {
        return -1000000000.0
    }
}

function Get-EvaluationMetrics {
    param(
        [string]$ModelName,
        [string]$RootDir
    )

    $metricsPath = Join-Path $RootDir ("{0}.json" -f $ModelName)
    if (-not (Test-Path $metricsPath)) {
        return $null
    }

    $payload = Get-Content -Raw -Path $metricsPath | ConvertFrom-Json
    return $payload.evaluation_metrics
}

function Invoke-ModelPrediction {
    param(
        [string]$ModelName,
        [string]$Task,
        [string]$ImagePath,
        [string]$Endpoint
    )

    $curlArgs = @(
        "-sS",
        "-X", "POST", $Endpoint,
        "-F", "file=@$ImagePath",
        "-F", "model_name=$ModelName",
        "-F", "explain=false",
        "-F", "target_task=$Task"
    )

    $raw = & curl.exe @curlArgs 2>&1

    if ($LASTEXITCODE -ne 0) {
        return [pscustomobject]@{
            Ok = $false
            Error = ($raw -join "`n")
            Json = $null
        }
    }

    try {
        $json = $raw | ConvertFrom-Json
        return [pscustomobject]@{
            Ok = $true
            Error = $null
            Json = $json
        }
    }
    catch {
        return [pscustomobject]@{
            Ok = $false
            Error = "Response was not valid JSON."
            Json = $null
        }
    }
}

function Save-PredictionJson {
    param(
        [Parameter(Mandatory = $true)]
        [object]$PredictionJson,
        [Parameter(Mandatory = $true)]
        [string]$TargetPath
    )

    $content = $PredictionJson | ConvertTo-Json -Depth 100
    [System.IO.File]::WriteAllText($TargetPath, $content, (New-Object System.Text.UTF8Encoding($false)))
}

function Save-SegmentationArtifacts {
    param(
        [Parameter(Mandatory = $true)]
        [string]$JsonPath,
        [Parameter(Mandatory = $true)]
        [string]$MaskPath,
        [Parameter(Mandatory = $true)]
        [string]$ProbPath,
        [Parameter(Mandatory = $true)]
        [double]$Threshold,
        [Parameter(Mandatory = $true)]
        [string]$PythonPath
    )

    $helperScript = Join-Path ([System.IO.Path]::GetTempPath()) ("rank_models_seg_{0}.py" -f ([guid]::NewGuid().ToString("N")))
    $pyCode = @'
import json
import sys

import numpy as np
from PIL import Image

json_path = sys.argv[1]
mask_path = sys.argv[2]
prob_path = sys.argv[3]
threshold = float(sys.argv[4])

with open(json_path, "r", encoding="utf-8-sig") as f:
    data = json.load(f)

arr = np.array(data.get("prediction"))
m = arr[0, 0] if arr.ndim == 4 else arr.squeeze()
prob = 1.0 / (1.0 + np.exp(-m))
mask = (prob > threshold).astype(np.uint8) * 255
prob_img = ((prob - prob.min()) / (prob.max() - prob.min() + 1e-8) * 255).astype(np.uint8)

Image.fromarray(mask).save(mask_path)
Image.fromarray(prob_img).save(prob_path)

print("max_prob=" + str(float(prob.max())))
print("mask_pixels=" + str(int((mask > 0).sum())))
'@

    Set-Content -Path $helperScript -Value $pyCode -Encoding UTF8
    try {
        $output = & $PythonPath $helperScript $JsonPath $MaskPath $ProbPath $Threshold 2>&1
    }
    finally {
        Remove-Item -Path $helperScript -ErrorAction SilentlyContinue
    }

    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create segmentation artifacts for $JsonPath. Output: $($output -join ' ')"
    }
}

if (-not (Test-Path $ClassificationImage)) {
    throw "Classification image not found: $ClassificationImage"
}

if (-not (Test-Path $SegmentationImage)) {
    throw "Segmentation image not found: $SegmentationImage"
}

$modelDirs = Get-ChildItem -Path $CheckpointsDir -Directory | Select-Object -ExpandProperty Name
$classModels = $modelDirs | Where-Object { $_ -like "classification.*" } | Sort-Object
$segModels = $modelDirs | Where-Object { $_ -like "segmentation.*" } | Sort-Object

$runStamp = Get-Date -Format "yyyyMMdd_HHmmss"
$runDir = Join-Path $OutputRoot ("rank_{0}" -f $runStamp)
$classOutputDir = Join-Path $runDir "classification"
$segOutputDir = Join-Path $runDir "segmentation"
New-Item -ItemType Directory -Path $classOutputDir -Force | Out-Null
New-Item -ItemType Directory -Path $segOutputDir -Force | Out-Null

if ($classModels.Count -eq 0) {
    throw "No classification model directories found under $CheckpointsDir"
}

if ($segModels.Count -eq 0) {
    throw "No segmentation model directories found under $CheckpointsDir"
}

Write-Host "\nRunning classification predictions..." -ForegroundColor Cyan
$classRows = foreach ($model in $classModels) {
    $prediction = Invoke-ModelPrediction -ModelName $model -Task "classification" -ImagePath $ClassificationImage -Endpoint $ApiUrl
    $eval = Get-EvaluationMetrics -ModelName $model -RootDir $CheckpointsDir
    $jsonPath = Join-Path $classOutputDir ("{0}.json" -f $model)

    $classIndex = $null
    $classScore = $null
    if ($prediction.Ok -and $null -ne $prediction.Json -and $null -ne $prediction.Json.prediction) {
        Save-PredictionJson -PredictionJson $prediction.Json -TargetPath $jsonPath

        if ($prediction.Json.prediction.PSObject.Properties.Name -contains "class_index") {
            $classIndex = $prediction.Json.prediction.class_index
        }

        if ($prediction.Json.prediction.PSObject.Properties.Name -contains "score") {
            $classScore = $prediction.Json.prediction.score
        }
    }

    [pscustomobject]@{
        Model = $model
        Accuracy = if ($null -ne $eval) { $eval.accuracy } else { $null }
        F1 = if ($null -ne $eval) { $eval.f1_score } else { $null }
        AUC = if ($null -ne $eval) { $eval.auc } else { $null }
        PredictOK = $prediction.Ok
        PredClassIndex = $classIndex
        PredScore = $classScore
        JsonPath = if ($prediction.Ok) { $jsonPath } else { "" }
        Error = if ($prediction.Ok) { "" } else { $prediction.Error }
    }
}

$rankedClass = $classRows | Sort-Object @{ Expression = { Get-ComparableNumber $_.Accuracy }; Descending = $true }, @{ Expression = { Get-ComparableNumber $_.F1 }; Descending = $true }, @{ Expression = { Get-ComparableNumber $_.AUC }; Descending = $true }

Write-Host "\nClassification ranking (by Accuracy, then F1, then AUC):" -ForegroundColor Green
$rankedClass | Format-Table Model, Accuracy, F1, AUC, PredictOK, PredClassIndex, PredScore -AutoSize

Write-Host "\nRunning segmentation predictions..." -ForegroundColor Cyan
$segRows = foreach ($model in $segModels) {
    $prediction = Invoke-ModelPrediction -ModelName $model -Task "segmentation" -ImagePath $SegmentationImage -Endpoint $ApiUrl
    $eval = Get-EvaluationMetrics -ModelName $model -RootDir $CheckpointsDir
    $jsonPath = Join-Path $segOutputDir ("{0}.json" -f $model)
    $maskPath = Join-Path $segOutputDir ("{0}.mask_t{1}.png" -f $model, ($SegmentationThreshold.ToString("0.##").Replace('.', 'p')))
    $probPath = Join-Path $segOutputDir ("{0}.prob_map.png" -f $model)
    $artifactError = ""

    if ($prediction.Ok -and $null -ne $prediction.Json -and $null -ne $prediction.Json.prediction) {
        Save-PredictionJson -PredictionJson $prediction.Json -TargetPath $jsonPath
        try {
            Save-SegmentationArtifacts -JsonPath $jsonPath -MaskPath $maskPath -ProbPath $probPath -Threshold $SegmentationThreshold -PythonPath $PythonExe
        }
        catch {
            $artifactError = $_.Exception.Message
        }
    }

    [pscustomobject]@{
        Model = $model
        Dice = if ($null -ne $eval) { $eval.dice_score } else { $null }
        IoU = if ($null -ne $eval) { $eval.iou } else { $null }
        Hausdorff = if ($null -ne $eval) { $eval.hausdorff_distance } else { $null }
        PredictOK = $prediction.Ok
        JsonPath = if ($prediction.Ok) { $jsonPath } else { "" }
        MaskPath = if ($prediction.Ok -and [string]::IsNullOrWhiteSpace($artifactError)) { $maskPath } else { "" }
        ProbPath = if ($prediction.Ok -and [string]::IsNullOrWhiteSpace($artifactError)) { $probPath } else { "" }
        Error = if ($prediction.Ok) { $artifactError } else { $prediction.Error }
    }
}

$rankedSeg = $segRows | Sort-Object @{ Expression = { Get-ComparableNumber $_.Dice }; Descending = $true }, @{ Expression = { Get-ComparableNumber $_.IoU }; Descending = $true }, @{ Expression = { Get-ComparableNumber $_.Hausdorff }; Descending = $false }

Write-Host "\nSegmentation ranking (by Dice, then IoU, then lower Hausdorff):" -ForegroundColor Green
$rankedSeg | Format-Table Model, Dice, IoU, Hausdorff, PredictOK -AutoSize

$bestClass = $rankedClass | Select-Object -First 1
$bestSeg = $rankedSeg | Select-Object -First 1

Write-Host "\nBest classification model: $($bestClass.Model)" -ForegroundColor Yellow
Write-Host "Best segmentation model:  $($bestSeg.Model)" -ForegroundColor Yellow
Write-Host "\nSaved per-model artifacts to: $runDir" -ForegroundColor Cyan

$classCsv = Join-Path $runDir "classification_ranking.csv"
$segCsv = Join-Path $runDir "segmentation_ranking.csv"
$rankedClass | Export-Csv -Path $classCsv -NoTypeInformation -Encoding UTF8
$rankedSeg | Export-Csv -Path $segCsv -NoTypeInformation -Encoding UTF8

Write-Host "Saved ranking CSVs:" -ForegroundColor Cyan
Write-Host " - $classCsv"
Write-Host " - $segCsv"
