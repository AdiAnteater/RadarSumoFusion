param(
    [string[]]$Campaigns = @(
        "campaign_current_12radar.json",
        "campaign_8radar.json",
        "campaign_4radar.json"
    )
)
$ErrorActionPreference = "Continue"
$env:PYTHONUNBUFFERED = "1"
# At a 3 m mount the radars return ~560 points/message and the CSV writer drains
# ~220 messages/s; unpaced 15 ms ticks overflow the per-radar deque and drop data.
# Wall-clock pacing only: simulation still advances 0.05 s per tick.
$env:DATASET_SYNC_MIN_PERIOD_S = "0.07"
$env:DATASET_RADAR_CAPTURE_DEQUE_MAX = "4096"
$py = "C:\Users\Colin\Downloads\CARLA_Latest\.venv\Scripts\python.exe"
$carlaDir = "C:\Users\Colin\Downloads\CARLA_Latest"
$root = "C:\Users\Colin\Downloads\CARLA_Latest\Scripts\RadarSumoFusion"
$log = Join-Path $root "campaign_logs\all_campaigns.log"
Set-Location $root

function Test-Carla {
    & $py -c "import carla,sys; c=carla.Client('127.0.0.1',2000); c.set_timeout(15.0); print(c.get_world().get_map().name)" 2>$null | Out-Null
    return ($LASTEXITCODE -eq 0)
}

function Ensure-Carla {
    if (Test-Carla) { return $true }
    Write-Output "[runner] CARLA not reachable; restarting it ..."
    Get-Process | Where-Object { $_.ProcessName -like '*Carla*' } | ForEach-Object { Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue }
    Start-Sleep -Seconds 5
    Start-Process -FilePath "$carlaDir\CarlaUE4.exe" -WorkingDirectory $carlaDir -ArgumentList "-vulkan","-quality-level=Low","-windowed","-ResX=800","-ResY=600"
    $deadline = (Get-Date).AddMinutes(4)
    Start-Sleep -Seconds 40
    while ((Get-Date) -lt $deadline) {
        if (Test-Carla) { Write-Output "[runner] CARLA is up."; Start-Sleep -Seconds 10; return $true }
        Start-Sleep -Seconds 10
    }
    return $false
}

$results = @()
foreach ($c in $Campaigns) {
    if (-not (Ensure-Carla)) {
        Write-Output "[runner] CARLA failed to come up; skipping $c"
        $results += "$c : SKIPPED (CARLA down)"
        continue
    }
    $baseDir = (Get-Content $c -Raw | ConvertFrom-Json).capture_base_dir
    $drive = if ($baseDir) { $baseDir.Substring(0, 1) } else { "C" }
    $freeGB = (Get-PSDrive $drive).Free / 1GB
    if ($freeGB -lt 25) {
        Write-Output ("[runner] only {0:N1} GB free on {1}:; skipping {2}" -f $freeGB, $drive, $c)
        $results += "$c : SKIPPED (low disk)"
        continue
    }
    $stamp = Get-Date -Format o
    Write-Output "===== START $c $stamp ====="
    & $py -u run_fusion.py --campaign $c 2>&1 | ForEach-Object { $_.ToString() }
    $code = $LASTEXITCODE
    Write-Output "===== END $c exit=$code $(Get-Date -Format o) ====="
    $results += "$c : exit=$code"
}
Write-Output "===== SUMMARY ====="
$results | ForEach-Object { Write-Output $_ }

