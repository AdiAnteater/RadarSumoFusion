param([int]$WaitPid = 0, [string]$Root = "D:\RadarSumoFusion_Data")
if ($WaitPid) { while (Get-Process -Id $WaitPid -ErrorAction SilentlyContinue) { Start-Sleep 30 } }

foreach ($dir in Get-ChildItem -LiteralPath $Root -Directory -Filter "sensor_capture_*_radar*_s1_s2_s11") {
    if ($dir.Name -notmatch '^sensor_capture_(\d{4})(\d{2})(\d{2})_\d{6}_radar(\d+)_s1_s2_s11$') { continue }
    if (-not (Test-Path -LiteralPath (Join-Path $dir.FullName "segments.json"))) {
        Write-Output "[rename] $($dir.Name): no segments.json, leaving as is"
        continue
    }
    $old = $dir.Name
    $new = "{0}radar_S1-S2-S11_{1}-{2}-{3}" -f $Matches[4], $Matches[1], $Matches[2], $Matches[3]
    if (Test-Path -LiteralPath (Join-Path $Root $new)) { $new = "${new}_$($old.Substring(24, 6))" }
    Rename-Item -LiteralPath $dir.FullName -NewName $new
    foreach ($rel in "run_meta.json", "radar_labeling_qa\summary.txt") {
        $f = Join-Path (Join-Path $Root $new) $rel
        if (Test-Path -LiteralPath $f) {
            (Get-Content -LiteralPath $f -Raw).Replace($old, $new) | Set-Content -LiteralPath $f -NoNewline -Encoding utf8
        }
    }
    Write-Output "[rename] $old -> $new"
}
