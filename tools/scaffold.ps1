param(
  [string[]]$Libs = @("timesfm"),
  [string]$Root = "C:\moinfo"
)

$commonDirs = @(
  "common",
  "docs",
  "tools",
  "libs"
)

$perLibDirs = @(
  "00_raw",
  "01_sandbox",
  "02_src",
  "03_scripts",
  "04_outputs",
  "05_reports",
  "06_tests",
  "07_configs",
  "08_logs",
  "notebooks"
)

# ルート共通ディレクトリ
foreach ($d in $commonDirs) {
  New-Item -ItemType Directory -Force -Path (Join-Path $Root $d) | Out-Null
}

# ライブラリごとのディレクトリ
foreach ($lib in $Libs) {
  $libRoot = Join-Path (Join-Path $Root "libs") $lib
  foreach ($d in $perLibDirs) {
    New-Item -ItemType Directory -Force -Path (Join-Path $libRoot $d) | Out-Null
  }

  # 解析結果の分類用サブフォルダ（好みで増減OK）
  New-Item -ItemType Directory -Force -Path (Join-Path $libRoot "04_outputs\api")   | Out-Null
  New-Item -ItemType Directory -Force -Path (Join-Path $libRoot "04_outputs\tables")| Out-Null
  New-Item -ItemType Directory -Force -Path (Join-Path $libRoot "04_outputs\figs")  | Out-Null
}

Write-Host "Scaffold created under: $Root"
