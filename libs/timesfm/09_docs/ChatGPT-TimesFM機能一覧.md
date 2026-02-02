# TimesFM機能一覧

**User:** Anonymous (tajimaharu98@gmail.com)  
**Created:** 2/2/2026 14:57:19  
**Updated:** 2/2/2026 15:03:37  
**Exported:** 2/2/2026 15:17:38  
**Link:** [https://chatgpt.com/c/69803c1a-e81c-83a7-99d3-11a79085640d](https://chatgpt.com/c/69803c1a-e81c-83a7-99d3-11a79085640d)  

## Response:
思考時間: 5m 11s

結論：Google Research版 TimesFM（※公開チェックポイント）は、基本的に「時系列の予測（forecasting/将来値予測）」が中心で、**2.0は単変量（univariate/目的変数が1本）特化**、\*\*評価・可視化・特徴量生成・埋め込み抽出は“同梱ツールとしては薄め”\*\*です（ただしBigQuery版は評価や異常検知の関数が用意されています）。[GitHub+3Google Research+3ハギングフェイス+3](https://research.google/blog/a-decoder-only-foundation-model-for-time-series-forecasting/?utm_source=chatgpt.com)

* * *

機能一覧（イベント＝工程ごと）
---------------

| イベント（工程） | 機能カテゴリ | 2.0（`timesfm-2.0-500m-pytorch`）で“実際にあるもの” | 2.5（参考：最新APIの方向性）で“実際にあるもの” | 補足（できない/注意点） |
| --- | --- | --- | --- | --- |
| 目的 | タスク | **単変量の時系列予測**（univariate forecasting）[ハギングフェイス](https://huggingface.co/google/timesfm-2.0-500m-pytorch) | 予測（＋共変量サポート復活）[GitHub](https://github.com/google-research/timesfm) | 分類・クラスタリング等は“標準機能”ではない |
| モデル取得 | チェックポイント | Hugging Face上のチェックポイントを参照[ハギングフェイス](https://huggingface.co/google/timesfm-2.0-500m-pytorch) | `from_pretrained("google/timesfm-2.5-200m-pytorch")` の例あり[GitHub](https://github.com/google-research/timesfm) | 2.0はHFの該当レポIDを指定する流れ |
| バージョン整合 | 互換性 | 2.0/1.0は「旧コード（`v1`）」扱い。`timesfm==1.3.0`で読み込み可と明記[GitHub](https://github.com/google-research/timesfm) | 2.5が最新、推論API刷新あり[GitHub](https://github.com/google-research/timesfm) | “同じtimesfmでもAPIが違う”のが最大の罠 |
| モデル初期化 | ロード（Python） | `timesfm.TimesFm(hparams=TimesFmHparams(...), checkpoint=TimesFmCheckpoint(huggingface_repo_id=...))`[ハギングフェイス](https://huggingface.co/google/timesfm-2.0-500m-pytorch) | `TimesFM_2p5_200M_torch.from_pretrained(...)` の例[GitHub](https://github.com/google-research/timesfm) | 2.0はパラメータの一部が固定（500M用）[ハギングフェイス](https://huggingface.co/google/timesfm-2.0-500m-pytorch) |
| 入力形式 | 配列入力 | `tfm.forecast()`（配列入力APIあり）[ハギングフェイス](https://huggingface.co/google/timesfm-2.0-500m-pytorch) | `model.forecast(horizon=..., inputs=[np.array, ...])`（例）[GitHub](https://github.com/google-research/timesfm) | どちらも「系列のリスト＝バッチ処理」発想 |
| 入力形式 | DataFrame入力 | `tfm.forecast_on_df(inputs=df, freq="M", value_name="y", num_jobs=...)`[ハギングフェイス](https://huggingface.co/google/timesfm-2.0-500m-pytorch) | （2.5はREADME上は配列例中心）[GitHub](https://github.com/google-research/timesfm) | 2.0のDFは `unique_id, ds, y` 形式が前提（例がREADMEに掲載）[ハギングフェイス](https://huggingface.co/google/timesfm-2.0-500m-pytorch) |
| 予測地平線 | horizon（予測ステップ数） | 2.0は `horizon_len` をHparamsで指定する設計[ハギングフェイス](https://huggingface.co/google/timesfm-2.0-500m-pytorch) | 2.5は `forecast(horizon=12, ...)` のように呼び出し時指定[GitHub](https://github.com/google-research/timesfm) | 2.0→2.5で“どこでhorizonを渡すか”が変わる |
| 出力 | 点予測 | 点予測（point forecast）に注力と明記[ハギングフェイス](https://huggingface.co/google/timesfm-2.0-500m-pytorch) | 点予測 `point_forecast` の例（shape例も）[GitHub](https://github.com/google-research/timesfm) | “平均/期待値”のように使うのが基本 |
| 出力 | 分位（quantile/分布の端） | 10分位ヘッドは\*\*実験的で未較正（uncalibrated）\*\*と明記[ハギングフェイス](https://huggingface.co/google/timesfm-2.0-500m-pytorch) | 連続分位（continuous quantile）対応・quantile headが任意、交差修正などフラグ言及[GitHub](https://github.com/google-research/timesfm) | 2.0の分位は「出しても信頼できるとは限らない」 |
| 前処理 | 欠損（NaN） | “穴なし（contiguous）が理想”、NaNは**線形補間（linear interpolation）で埋める**と明記[ハギングフェイス](https://huggingface.co/google/timesfm-2.0-500m-pytorch) | （2.5は `normalize_inputs` などの設定例）[GitHub](https://github.com/google-research/timesfm) | 欠損が多い系列は結果が壊れやすい |
| 周波数 | frequency（粒度） | 周波数カテゴリ `{0,1,2}` を入力。DFは freq文字→カテゴリに変換[ハギングフェイス+1](https://huggingface.co/google/timesfm-2.0-500m-pytorch) | 2.5は frequency 指標を廃止と明記[GitHub](https://github.com/google-research/timesfm) | 2.0のfrequencyは“ハイパラ的つまみ”として扱えるとも明記[ハギングフェイス](https://huggingface.co/google/timesfm-2.0-500m-pytorch) |
| 単変量 | univariate | **単変量**と明記[ハギングフェイス](https://huggingface.co/google/timesfm-2.0-500m-pytorch) | （基本は同様。共変量は後述）[GitHub](https://github.com/google-research/timesfm) | 目的変数（y）が1本であることが前提 |
| 多系列 | バッチ処理 | 入力をリストで渡し複数系列を同時予測（例あり）[ハギングフェイス](https://huggingface.co/google/timesfm-2.0-500m-pytorch) | 入力リストで複数系列（例あり）[GitHub](https://github.com/google-research/timesfm) | “多変量”ではなく“複数の単変量系列をまとめて” |
| 多変量 | multivariate（目的変数がベクトル） | 公式記述としては単変量前提[ハギングフェイス](https://huggingface.co/google/timesfm-2.0-500m-pytorch) | 同上（README上は目的変数ベクトルの言及なし）[GitHub](https://github.com/google-research/timesfm) | ここで言う多変量（複数チャネルy）は**別設計**になりがち |
| 共変量 | 外生変数（covariates / XReg） | 2.0 README（HF）では少なくとも主要導線は “予測”中心[ハギングフェイス](https://huggingface.co/google/timesfm-2.0-500m-pytorch) | 2.5で**XReg（外生回帰）サポートを追加し直した**と明記、`.[xreg]` 追加インストールも記載[GitHub](https://github.com/google-research/timesfm) | “多変量”の代わりに「y1本＋外生変数」で表現するのが現実的 |
| 推論高速化 | compile | （2.0 HFカードでは明示なし） | `model.compile(ForecastConfig(...))` 例あり[GitHub](https://github.com/google-research/timesfm) | 実運用ではここが効く（レイテンシ短縮） |
| モデル保存 | 保存（save） | HFカード上は「チェックポイントを使う」導線が中心[ハギングフェイス](https://huggingface.co/google/timesfm-2.0-500m-pytorch) | （2.5のPython公開APIには `save_pretrained`/`push_to_hub` 系が存在※） | ※ローカル抽出したtimesfm公開API上で確認（同梱CSV） |
| 評価 | 指標計算 | ライブラリ同梱としての評価器は前面に出ていない[ハギングフェイス](https://huggingface.co/google/timesfm-2.0-500m-pytorch) | 同上 | ふつうは `MAE/RMSE/MAPE/sMAPE/MASE/CRPS` 等を自前 or sktime等で計算 |
| 異常検知 | anomaly detection | Python版TimesFMの“標準機能”としては明示されていない | 同上 | ただしBigQuery版TimesFMは `AI.DETECT_ANOMALIES` を提供（Preview）[Google Cloud Documentation](https://docs.cloud.google.com/bigquery/docs/timesfm-model) |
| 可視化 | plot | 同梱なし（想定は `matplotlib` 等） | 同梱なし | 予測の帯（分位）を塗る、残差プロット、予測区間の被覆率などが定番 |
| 解析・分析 | モデル理解 | 研究ブログ/論文への参照が中心（使い方は予測APIが主）[Google Research+1](https://research.google/blog/a-decoder-only-foundation-model-for-time-series-forecasting/?utm_source=chatgpt.com) | 同上＋推論API刷新の説明[GitHub](https://github.com/google-research/timesfm) | “解釈性ツール”は別途（SHAP等も時系列では工夫が必要） |
| ベクトル抽出 | embedding（埋め込み） | 公式の「埋め込み抽出API」は前面に出ていない[ハギングフェイス](https://huggingface.co/google/timesfm-2.0-500m-pytorch) | 同上（README例は予測中心）[GitHub](https://github.com/google-research/timesfm) | やるなら内部層のhidden stateをフックする改造寄り（安定APIではない） |
| 特徴量生成 | feature engineering | 同梱の特徴量生成器は基本なし | 同上 | 現実解：①予測値/分位/残差を特徴量化 ②外生変数をXRegで統合[GitHub](https://github.com/google-research/timesfm) |
| マネージド利用 | SQLで実行 | （Pythonの範囲外） | （Pythonの範囲外） | BigQueryでは `AI.FORECAST` / `AI.EVALUATE` / `AI.DETECT_ANOMALIES` が用意されていて「モデル管理不要」が売り[Google Cloud Documentation](https://docs.cloud.google.com/bigquery/docs/timesfm-model) |

* * *

### 超短い補足（誤解が起きやすい点）

*   「多変量」は、**(A) 複数の単変量系列をまとめて投げる**のか、\*\*(B) 1系列が多チャネル（ベクトルy）\*\*なのかで意味が割れます。TimesFM 2.0の公式説明は(A)はできるが、(B)は前提にしていません。[ハギングフェイス+1](https://huggingface.co/google/timesfm-2.0-500m-pytorch)
*   「評価・可視化・特徴量生成・埋め込み抽出」は、TimesFM本体よりも**周辺ツール（NumPy/pandas/matplotlib、評価ライブラリ、SQL側の関数）で組む**のが基本ルートです。[Google Cloud Documentation](https://docs.cloud.google.com/bigquery/docs/timesfm-model)



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)