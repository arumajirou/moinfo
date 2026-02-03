# 実装方針とディレクトリ構成

**User:** Anonymous (tajimaharu98@gmail.com)  
**Created:** 2/3/2026 15:54:50  
**Updated:** 2/3/2026 16:18:39  
**Exported:** 2/3/2026 16:33:02  
**Link:** [https://chatgpt.com/c/69819afa-9738-83a5-a69f-70a17f7b3a47](https://chatgpt.com/c/69819afa-9738-83a5-a69f-70a17f7b3a47)  

## Response:
思考時間: 2m 8s

結論：**「共通基盤（入出力・設定・評価）→ 時系列データ整形 → イベント特徴量 → モデル（Chronos/TimesFM）アダプタ → 実験・レポート自動生成」**の順で作るのが最短で壊れにくいです。ロジックは `.py` に寄せ、`C:\moinfo\moinfo.ipynb` は**実行・確認のハブ**にします。

* * *

0) 実装方針（最初に固定するルール）
-------------------

*   **Notebookは薄く**：前処理・学習・推論・評価は原則 `.py` 側へ（再利用とテストのため）
*   **共通I/F（インターフェース）統一**：`fit()` / `predict()` / `predict_quantiles()` を持つ `Forecaster` 形に寄せる（Chronos/TimesFM/ベースラインで共通化）
*   **成果物の置き場を固定**：テーブル/図/ログ/設定を既存の `04_outputs` / `08_logs` に寄せる（再現性が上がる）

* * *

1) 作成するディレクトリ＆ファイル（フルパス）
------------------------

既存構成を活かしつつ、**共通機能は `C:\moinfo\common` 配下に新設**、モデル固有は既存 `libs/*/02_src` に追加します。

### 共通（新規）

*   `C:\moinfo\common\moinfo_core\__init__.py`
*   `C:\moinfo\common\moinfo_core\config.py`（設定読み込み：YAML/JSON、デフォルト統合）
*   `C:\moinfo\common\moinfo_core\paths.py`（出力先・モデルパス解決）
*   `C:\moinfo\common\moinfo_core\logging.py`（ログ整形、実験ID付与）
*   `C:\moinfo\common\moinfo_data\__init__.py`
*   `C:\moinfo\common\moinfo_data\io.py`（CSVロード、列名マッピング）
*   `C:\moinfo\common\moinfo_data\schema.py`（時系列スキーマ検証）
*   `C:\moinfo\common\moinfo_data\preprocess.py`（欠損/外れ値/正規化）
*   `C:\moinfo\common\moinfo_data\windows.py`（スライディング窓生成、horizon対応）
*   `C:\moinfo\common\moinfo_data\splits.py`（時系列分割、バックテスト用分割）
*   `C:\moinfo\common\moinfo_features\__init__.py`
*   `C:\moinfo\common\moinfo_features\time_features.py`（曜日・月・周期など）
*   `C:\moinfo\common\moinfo_features\event_features.py`（イベントテーブル結合、将来イベントも扱う）
*   `C:\moinfo\common\moinfo_eval\__init__.py`
*   `C:\moinfo\common\moinfo_eval\metrics.py`（MAE/RMSE/sMAPE等）
*   `C:\moinfo\common\moinfo_eval\backtest.py`（rolling評価、horizon別集計）
*   `C:\moinfo\common\moinfo_experiments\__init__.py`
*   `C:\moinfo\common\moinfo_experiments\runner.py`（設定→学習/推論/評価→成果物出力）
*   `C:\moinfo\common\moinfo_experiments\registry.py`（モデル名→Forecaster生成）
*   `C:\moinfo\common\moinfo_baselines\__init__.py`
*   `C:\moinfo\common\moinfo_baselines\naive.py`（Naive/SeasonalNaive）
*   `C:\moinfo\common\moinfo_baselines\statsmodels_arima.py`（ARIMA系：イベント外生変数対応の比較用）

### Chronos側（追加）

*   `C:\moinfo\libs\chronos\02_src\moinfo_chronos_ext\adapter\forecaster.py`（Chronos用Forecaster実装）
*   `C:\moinfo\libs\chronos\02_src\moinfo_chronos_ext\utils\model_registry.py`（`chronos_bolt_local`/`chronos_hf_local` のロード解決）
*   `C:\moinfo\libs\chronos\03_scripts\run_forecast.py`（CLI実行：設定→推論→CSV出力）
*   `C:\moinfo\libs\chronos\06_tests\test_chronos_forecaster.py`

### TimesFM側（追加）

*   `C:\moinfo\libs\timesfm\02_src\moinfo_timesfm_ext\adapter\forecaster.py`（TimesFM用Forecaster実装）
*   `C:\moinfo\libs\timesfm\03_scripts\run_forecast.py`
*   `C:\moinfo\libs\timesfm\06_tests\test_timesfm_forecaster.py`

### 実験設定（新規）

*   `C:\moinfo\07_configs\experiments\baseline.yaml`
*   `C:\moinfo\07_configs\experiments\chronos2_local.yaml`
*   `C:\moinfo\07_configs\experiments\timesfm_local.yaml`
*   `C:\moinfo\07_configs\experiments\compare_all.yaml`

* * *

2) 「時系列 × イベント × 機能」実装工程計画票（確認項目つき）
-----------------------------------

> ここでは **イベント＝外生イベント（祝日/施策/障害など）** を主軸に設計します。Chronos/TimesFMが外生変数を直接受けない場合も多いので、その場合は「ベースラインモデルでイベント有効性を検証」＋「Chronos/TimesFMはターゲット単変量で比較」という二段構えにします。

| ID | 分類（時系列/イベント/機能） | 実装内容（期待する機能） | 作成/更新（フルパス） | 確認事項（DoD） | `moinfo.ipynb` での確認 |
| --- | --- | --- | --- | --- | --- |
| 1 | 機能：基盤 | 実験設定ロード・出力パス統一（設定→実行が再現可能） | `C:\moinfo\common\moinfo_core\config.py` `paths.py` | 同設定で同一結果/同一出力構造 | 設定1つで実行できる |
| 2 | 機能：基盤 | ログ整形（実験ID、例外時の原因追跡） | `C:\moinfo\common\moinfo_core\logging.py` | 例外時に「どの設定/どのデータで落ちたか」追える | わざと失敗させて確認 |
| 3 | 時系列：入出力 | CSV→標準スキーマへ正規化（timestamp/target/item\_id） | `C:\moinfo\common\moinfo_data\io.py` `schema.py` | `train.csv` が読み込め、欠損/型不一致を明示 | `data/train.csv` をロード |
| 4 | 時系列：前処理 | 欠損補完・外れ値処理・周波数推定（頑健な前処理） | `C:\moinfo\common\moinfo_data\preprocess.py` | 前処理前後で件数/欠損数がレポートされる | before/after集計 |
| 5 | 時系列：窓生成 | スライディング窓（context\_length/horizon）生成 | `C:\moinfo\common\moinfo_data\windows.py` | 期待shape・境界条件（短系列）OK | 窓の例を表示 |
| 6 | 時系列：分割 | 時系列分割＆rollingバックテスト分割 | `C:\moinfo\common\moinfo_data\splits.py` | リーク（未来混入）がない | 分割境界を可視化 |
| 7 | イベント：定義 | イベントテーブル仕様（event\_name, start/end, scope） | `C:\moinfo\common\moinfo_features\event_features.py` | イベントを時間軸にjoinできる | ダミーイベントで確認 |
| 8 | イベント：特徴量 | カレンダー特徴（曜日/祝日/四半期など） | `C:\moinfo\common\moinfo_features\time_features.py` | 再現性ある特徴が生成される | 生成列を表示 |
| 9 | イベント：将来既知 | 将来イベント（セール日程等）を未来horizonへ付与 | `C:\moinfo\common\moinfo_features\event_features.py` | 未来期間にも特徴が出る | 未来行の特徴確認 |
| 10 | 機能：評価 | 指標（MAE/RMSE/sMAPE等）とhorizon別集計 | `C:\moinfo\common\moinfo_eval\metrics.py` | 指標がNaNにならない（欠損耐性） | 指標テーブル表示 |
| 11 | 機能：バックテスト | rolling評価（複数cutoff） | `C:\moinfo\common\moinfo_eval\backtest.py` | cutoffごとの結果がCSV化 | `04_outputs/tables` 生成 |
| 12 | 機能：ベースライン | Naive/季節Naive（最低ラインの確立） | `C:\moinfo\common\moinfo_baselines\naive.py` | Chronos/TimesFMより弱くても動く基準 | baseline結果を比較 |
| 13 | イベント×機能 | ARIMA等でイベント外生変数の効き目検証 | `C:\moinfo\common\moinfo_baselines\statsmodels_arima.py` | イベント有無で改善/悪化が見える | 有/無の差を確認 |
| 14 | 機能：Chronos | ローカルモデル解決（bolt/hf） | `C:\moinfo\libs\chronos\02_src\moinfo_chronos_ext\utils\model_registry.py` | `C:\moinfo\chronos_*_local` を指定でロード | ロードのみ先に確認 |
| 15 | 時系列×機能 | Chronos Forecaster（predict/predict\_quantiles） | `C:\moinfo\libs\chronos\02_src\moinfo_chronos_ext\adapter\forecaster.py` | 1系列・多系列で推論できる | 予測プロット確認 |
| 16 | 機能：TimesFM | TimesFM Forecaster（同I/F） | `C:\moinfo\libs\timesfm\02_src\moinfo_timesfm_ext\adapter\forecaster.py` | Chronosと同じrunnerで動く | 同一データで比較 |
| 17 | 機能：実験運転 | 設定→モデル→評価→成果物の一気通貫 | `C:\moinfo\common\moinfo_experiments\runner.py` `registry.py` | `compare_all.yaml` で全モデルが回る | 1コマンド相当で実行 |
| 18 | 機能：CLI化 | Notebook外でも回せる（再現・自動化） | `C:\moinfo\libs\chronos\03_scripts\run_forecast.py` `C:\moinfo\libs\timesfm\03_scripts\run_forecast.py` | 実行すると `04_outputs` にCSV/ログが落ちる | notebookからsubprocessでもOK |
| 19 | 品質：テスト | 最低限の単体テスト（I/Fとshape） | `C:\moinfo\libs\chronos\06_tests\...` `C:\moinfo\libs\timesfm\06_tests\...` | 主要I/Fが壊れない | テスト実行ログ |
| 20 | 機能：成果物 | 予測・指標・設定の保存（監査可能） | 既存 `C:\moinfo\libs\*\04_outputs\...` | 「どの設定で出た結果か」追跡可能 | 出力の整合性確認 |

* * *

3) `moinfo.ipynb` 側の運用設計（薄くする）
------------------------------

`C:\moinfo\moinfo.ipynb` は以下の“確認セル”だけ持つのが理想です：

1.  環境/パス設定（`sys.path` or `pip install -e` の確認）
2.  データロード→前処理→分割（数行で）
3.  モデル選択（Chronos/TimesFM/ベースライン）→ `runner` 呼び出し
4.  結果テーブル・図の表示（`04_outputs` を読むだけ）

* * *

### 補足（重要な現実）

*   **Chronos/TimesFMがイベント外生変数を直接入力できない**可能性があるので、イベントの価値はまず **外生対応ベースライン（ARIMA系など）でファクトチェック**し、Chronos/TimesFMは「単変量の強さ」で比較する設計が安全です（これで“イベントが効いたのか/モデルが強いだけか”が分離できます）。

この計画に沿えば、次の一手は **ID1〜6（基盤＋時系列データ）を先に固める**のが最短です。ここが固まると Chronos/TimesFM の実装は“差し替え部品”になります。



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)