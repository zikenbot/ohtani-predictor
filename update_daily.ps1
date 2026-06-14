# 大谷翔平 予測システム 毎日自動更新スクリプト
# タスクスケジューラから 19:00 JST に実行される

$ProjectDir = "C:\Users\kenzi\shohei ohtani"
$Python = "py"
$LogFile = "$ProjectDir\logs\update_$(Get-Date -Format 'yyyyMMdd').log"

New-Item -ItemType Directory -Force -Path "$ProjectDir\logs" | Out-Null

function Log($msg) {
    $line = "$(Get-Date -Format 'HH:mm:ss') $msg"
    Write-Host $line
    Add-Content -Path $LogFile -Value $line -Encoding UTF8
}

Set-Location $ProjectDir
Log "=== 自動更新開始 ==="

# 1. スケジュール更新（今後 14 日）
Log "[1/3] スケジュール更新..."
& $Python "collect_schedule.py" "--days" "14" | ForEach-Object { Log "  $_" }

# 2. 大谷 Statcast 更新（今シーズン）
Log "[2/3] 大谷 Statcast 更新..."
& $Python "collect_statcast.py" "--years" "2026" "--force" | ForEach-Object { Log "  $_" }

# 3. 対戦相手データ更新
Log "[3/3] 対戦相手データ更新..."
& $Python "collect_opponents.py" "--year" "2026" "--force" | ForEach-Object { Log "  $_" }

# 4. Git commit & push
Log "[4/4] GitHub push..."
git add data/
$today = Get-Date -Format "yyyy-MM-dd"
git diff --cached --quiet
if ($LASTEXITCODE -ne 0) {
    git commit -m "Daily data refresh $today"
    git push
    Log "  push 完了"
} else {
    Log "  変更なし、skip"
}

Log "=== 自動更新完了 ==="
