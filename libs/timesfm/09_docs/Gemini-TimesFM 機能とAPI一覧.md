# TimesFM 機能とAPI一覧

**Exported:** 2/2/2026 15:01:31  
**Link:** [https://gemini.google.com/app/f52fd64fc0cd8706?hl=ja](https://gemini.google.com/app/f52fd64fc0cd8706?hl=ja)

## Response:
```
import pandas as pd

# Load the main API CSV
df = pd.read_csv('timesfm_api_with_doc_clusters.csv')

# Define keywords to search for in 'name', 'qualname', or 'doc_summary'
keywords = {
    'Loading': ['load', 'from_checkpoint', 'read'],
    'Saving': ['save', 'export', 'write'],
    'Forecasting': ['forecast', 'predict'],
    'Training/Fine-tuning': ['train', 'fit', 'compile', 'backward', 'optimizer'],
    'Evaluation': ['eval', 'metric', 'loss', 'accuracy', 'validate'],
    'Visualization': ['plot', 'visualize', 'draw', 'show'],
    'Embeddings/Features': ['embed', 'feature', 'hidden', 'represent', 'encode'],
    'Multivariate/Covariates': ['covariate', 'multivariate', 'regressor', 'xreg'],
    'Configuration': ['config', 'option', 'setting']
}

results = {}

for category, keys in keywords.items():
    mask = df['name'].str.contains('|'.join(keys), case=False, na=False) | \
           df['doc_summary'].str.contains('|'.join(keys), case=False, na=False)
    
    # Get relevant columns
    subset = df[mask][['qualname', 'name', 'kind', 'doc_summary']].head(10)
    results[category] = subset

# Print results to inspect
for cat, subset in results.items():
    print(f"--- {cat} ---")
    print(subset.to_string())
    print("\n")

```

コードの出力

```
--- Loading ---
                                                                        qualname                                name    kind                                                                                                                                                                                                                                                 doc_summary
5                                 timesfm.TimesFM_2p5_200M_torch.from_pretrained                     from_pretrained  method  Download a model from the Huggingface Hub and instantiate it. Args: pretrained_model_name_or_path (`str`, `Path`): - Either the `model_id` (string) of a model hosted on the Hub, e.g. `bigscience/bloom`. - Or a path to a `directory` containing model …
6                                 timesfm.TimesFM_2p5_200M_torch.load_checkpoint                     load_checkpoint  method                                                                                                                                                                                                                    Loads a TimesFM model from a checkpoint.
7                                     timesfm.TimesFM_2p5_200M_torch.push_to_hub                         push_to_hub  method  Upload model checkpoint to the Hub. Use `allow_patterns` and `ignore_patterns` to precisely filter which files should be pushed to the hub. Use `delete_patterns` to delete existing remote files in the same commit. See [`upload_folder`] reference for…
34                     timesfm.torch.dense.RandomFourierFeatures.load_state_dict                     load_state_dict  method  Copy parameters and buffers from :attr:`state_dict` into this module and its descendants. If :attr:`strict` is ``True``, then the keys of :attr:`state_dict` must exactly match the keys returned by this module's :meth:`~torch.nn.Module.state_dict` fu…
39                       timesfm.torch.dense.RandomFourierFeatures.named_modules                       named_modules  method  Return an iterator over all modules in the network, yielding both the name of the module as well as the module itself. Args: memo: a memo to store the set of modules already added to the result prefix: a prefix that will be added to the name of the …
48  timesfm.torch.dense.RandomFourierFeatures.register_load_state_dict_post_hook  register_load_state_dict_post_hook  method  Register a post-hook to be run after module's :meth:`~nn.Module.load_state_dict` is called. It should have the following signature:: hook(module, incompatible_keys) -> None The ``module`` argument is the current module that this hook is registered o…
49   timesfm.torch.dense.RandomFourierFeatures.register_load_state_dict_pre_hook   register_load_state_dict_pre_hook  method  Register a pre-hook to be run before module's :meth:`~nn.Module.load_state_dict` is called. It should have the following signature:: hook(module, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs) -> None # noqa: …
55                     timesfm.torch.dense.RandomFourierFeatures.set_extra_state                     set_extra_state  method  Set extra state contained in the loaded `state_dict`. This function is called from :func:`load_state_dict` to handle any extra state found within the `state_dict`. Implement this function and a corresponding :func:`get_extra_state` for your module i…
85                             timesfm.torch.dense.ResidualBlock.load_state_dict                     load_state_dict  method  Copy parameters and buffers from :attr:`state_dict` into this module and its descendants. If :attr:`strict` is ``True``, then the keys of :attr:`state_dict` must exactly match the keys returned by this module's :meth:`~torch.nn.Module.state_dict` fu…
90                               timesfm.torch.dense.ResidualBlock.named_modules                       named_modules  method  Return an iterator over all modules in the network, yielding both the name of the module as well as the module itself. Args: memo: a memo to store the set of modules already added to the result prefix: a prefix that will be added to the name of the …

--- Saving ---
                                         qualname             name    kind                                                                                                                                                                                                                                                 doc_summary
8  timesfm.TimesFM_2p5_200M_torch.save_pretrained  save_pretrained  method  Save weights in local directory. Args: save_directory (`str` or `Path`): Path to directory in which the model weights and configuration will be saved. config (`dict` or `DataclassInstance`, *optional*): Model configuration specified as a key/value d…

--- Forecasting ---
                                                  qualname                      name    kind                                                                                                                                                                                                                                                 doc_summary
0                                   timesfm.ForecastConfig            ForecastConfig   class  Options for forecasting. Attributes: max_context: The maximum context length. This is used by the complied decode function at inference time during batched inference. Any input time series with length less than max_context will be padded with zeros,…
2                   timesfm.TimesFM_2p5_200M_torch.compile                   compile  method         Attempts to compile the model for fast decoding. See configs.ForecastConfig for more details on the supported flags. Args: forecast_config: Configuration for forecasting flags. **kwargs: Additional keyword arguments to pass to model.compile().
3                  timesfm.TimesFM_2p5_200M_torch.forecast                  forecast  method                                                                                                                                                                                                                                  Forecasts the time series.
4  timesfm.TimesFM_2p5_200M_torch.forecast_with_covariates  forecast_with_covariates  method  Forecasts on a list of time series with covariates. To optimize inference speed, avoid string valued categorical covariates. Args: inputs: A list of time series forecast contexts. Each context time series should be in a format convertible to JTensor…
9                           timesfm.configs.ForecastConfig            ForecastConfig   class  Options for forecasting. Attributes: max_context: The maximum context length. This is used by the complied decode function at inference time during batched inference. Any input time series with length less than max_context will be padded with zeros,…

--- Training/Fine-tuning ---
                                                            qualname                    name    kind                                                                                                                                                                                                                                                 doc_summary
2                             timesfm.TimesFM_2p5_200M_torch.compile                 compile  method         Attempts to compile the model for fast decoding. See configs.ForecastConfig for more details on the supported flags. Args: forecast_config: Configuration for forecasting flags. **kwargs: Additional keyword arguments to pass to model.compile().
5                     timesfm.TimesFM_2p5_200M_torch.from_pretrained         from_pretrained  method  Download a model from the Huggingface Hub and instantiate it. Args: pretrained_model_name_or_path (`str`, `Path`): - Either the `model_id` (string) of a model hosted on the Hub, e.g. `bigscience/bloom`. - Or a path to a `directory` containing model …
8                     timesfm.TimesFM_2p5_200M_torch.save_pretrained         save_pretrained  method  Save weights in local directory. Args: save_directory (`str` or `Path`): Path to directory in which the model weights and configuration will be saved. config (`dict` or `DataclassInstance`, *optional*): Model configuration specified as a key/value d…
20                 timesfm.torch.dense.RandomFourierFeatures.compile                 compile  method               Compile this Module's forward using :func:`torch.compile`. This Module's `__call__` method is compiled and all arguments are passed as-is to :func:`torch.compile`. See :func:`torch.compile` for details on the arguments for this function.
22                    timesfm.torch.dense.RandomFourierFeatures.cuda                    cuda  method  Move all model parameters and buffers to the GPU. This also makes associated parameters and buffers different objects. So it should be called before constructing the optimizer if the module will live on GPU while being optimized. .. note:: This meth…
24                    timesfm.torch.dense.RandomFourierFeatures.eval                    eval  method  Set the module in evaluation mode. This has an effect only on certain modules. See the documentation of particular modules for details of their behaviors in training/evaluation mode, i.e. whether they are affected, e.g. :class:`Dropout`, :class:`Bat…
33                     timesfm.torch.dense.RandomFourierFeatures.ipu                     ipu  method  Move all model parameters and buffers to the IPU. This also makes associated parameters and buffers different objects. So it should be called before constructing the optimizer if the module will live on IPU while being optimized. .. note:: This meth…
36                    timesfm.torch.dense.RandomFourierFeatures.mtia                    mtia  method  Move all model parameters and buffers to the MTIA. This also makes associated parameters and buffers different objects. So it should be called before constructing the optimizer if the module will live on MTIA while being optimized. .. note:: This me…
41              timesfm.torch.dense.RandomFourierFeatures.parameters              parameters  method  Return an iterator over module parameters. This is typically passed to an optimizer. Args: recurse (bool): if True, then yields parameters of this module and all submodules. Otherwise, yields only parameters that are direct members of this module. Y…
42  timesfm.torch.dense.RandomFourierFeatures.register_backward_hook  register_backward_hook  method  Register a backward hook on the module. This function is deprecated in favor of :meth:`~torch.nn.Module.register_full_backward_hook` and the behavior of this function will change in future versions. Returns: :class:`torch.utils.hooks.RemovableHandle…

--- Evaluation ---
                                               qualname   name    kind                                                                                                                                                                                                                                                 doc_summary
24       timesfm.torch.dense.RandomFourierFeatures.eval   eval  method  Set the module in evaluation mode. This has an effect only on certain modules. See the documentation of particular modules for details of their behaviors in training/evaluation mode, i.e. whether they are affected, e.g. :class:`Dropout`, :class:`Bat…
61      timesfm.torch.dense.RandomFourierFeatures.train  train  method  Set the module in training mode. This has an effect only on certain modules. See the documentation of particular modules for details of their behaviors in training/evaluation mode, i.e., whether they are affected, e.g. :class:`Dropout`, :class:`Batc…
75               timesfm.torch.dense.ResidualBlock.eval   eval  method  Set the module in evaluation mode. This has an effect only on certain modules. See the documentation of particular modules for details of their behaviors in training/evaluation mode, i.e. whether they are affected, e.g. :class:`Dropout`, :class:`Bat…
112             timesfm.torch.dense.ResidualBlock.train  train  method  Set the module in training mode. This has an effect only on certain modules. See the documentation of particular modules for details of their behaviors in training/evaluation mode, i.e., whether they are affected, e.g. :class:`Dropout`, :class:`Batc…
125            timesfm.torch.normalization.RMSNorm.eval   eval  method  Set the module in evaluation mode. This has an effect only on certain modules. See the documentation of particular modules for details of their behaviors in training/evaluation mode, i.e. whether they are affected, e.g. :class:`Dropout`, :class:`Bat…
162           timesfm.torch.normalization.RMSNorm.train  train  method  Set the module in training mode. This has an effect only on certain modules. See the documentation of particular modules for details of their behaviors in training/evaluation mode, i.e., whether they are affected, e.g. :class:`Dropout`, :class:`Batc…
176            timesfm.torch.transformer.LayerNorm.eval   eval  method  Set the module in evaluation mode. This has an effect only on certain modules. See the documentation of particular modules for details of their behaviors in training/evaluation mode, i.e. whether they are affected, e.g. :class:`Dropout`, :class:`Bat…
213           timesfm.torch.transformer.LayerNorm.train  train  method  Set the module in training mode. This has an effect only on certain modules. See the documentation of particular modules for details of their behaviors in training/evaluation mode, i.e., whether they are affected, e.g. :class:`Dropout`, :class:`Batc…
227   timesfm.torch.transformer.MultiHeadAttention.eval   eval  method  Set the module in evaluation mode. This has an effect only on certain modules. See the documentation of particular modules for details of their behaviors in training/evaluation mode, i.e. whether they are affected, e.g. :class:`Dropout`, :class:`Bat…
264  timesfm.torch.transformer.MultiHeadAttention.train  train  method  Set the module in training mode. This has an effect only on certain modules. See the documentation of particular modules for details of their behaviors in training/evaluation mode, i.e., whether they are affected, e.g. :class:`Dropout`, :class:`Batc…

--- Visualization ---
Empty DataFrame
Columns: [qualname, name, kind, doc_summary]
Index: []

--- Embeddings/Features ---
                                                    qualname                         name    kind                                                                                                                                                                                             doc_summary
10               timesfm.configs.RandomFourierFeaturesConfig  RandomFourierFeaturesConfig   class                                                                                                                                                  Framework-agnostic config for random fourier features.
14                 timesfm.torch.dense.RandomFourierFeatures        RandomFourierFeatures   class                                                                                                                                                                          Random Fourier features layer.
25      timesfm.torch.dense.RandomFourierFeatures.extra_repr                   extra_repr  method  Return the extra representation of the module. To print customized extra information, you should re-implement this method in your own modules. Both single-line and multi-line strings are acceptable.
76              timesfm.torch.dense.ResidualBlock.extra_repr                   extra_repr  method  Return the extra representation of the module. To print customized extra information, you should re-implement this method in your own modules. Both single-line and multi-line strings are acceptable.
126           timesfm.torch.normalization.RMSNorm.extra_repr                   extra_repr  method  Return the extra representation of the module. To print customized extra information, you should re-implement this method in your own modules. Both single-line and multi-line strings are acceptable.
177           timesfm.torch.transformer.LayerNorm.extra_repr                   extra_repr  method  Return the extra representation of the module. To print customized extra information, you should re-implement this method in your own modules. Both single-line and multi-line strings are acceptable.
228  timesfm.torch.transformer.MultiHeadAttention.extra_repr                   extra_repr  method  Return the extra representation of the module. To print customized extra information, you should re-implement this method in your own modules. Both single-line and multi-line strings are acceptable.
279         timesfm.torch.transformer.PerDimScale.extra_repr                   extra_repr  method  Return the extra representation of the module. To print customized extra information, you should re-implement this method in your own modules. Both single-line and multi-line strings are acceptable.
329             timesfm.torch.transformer.RMSNorm.extra_repr                   extra_repr  method  Return the extra representation of the module. To print customized extra information, you should re-implement this method in your own modules. Both single-line and multi-line strings are acceptable.
369      timesfm.torch.transformer.RotaryPositionalEmbedding    RotaryPositionalEmbedding   class                                                                                                                                                                            Rotary positional embedding.

--- Multivariate/Covariates ---
                                                  qualname                      name    kind                                                                                                                                                                                                                                                 doc_summary
4  timesfm.TimesFM_2p5_200M_torch.forecast_with_covariates  forecast_with_covariates  method  Forecasts on a list of time series with covariates. To optimize inference speed, avoid string valued categorical covariates. Args: inputs: A list of time series forecast contexts. Each context time series should be in a format convertible to JTensor…

--- Configuration ---
                                                   qualname                         name    kind                                                                                                                                                                                                                                                 doc_summary
0                                    timesfm.ForecastConfig               ForecastConfig   class  Options for forecasting. Attributes: max_context: The maximum context length. This is used by the complied decode function at inference time during batched inference. Any input time series with length less than max_context will be padded with zeros,…
2                    timesfm.TimesFM_2p5_200M_torch.compile                      compile  method         Attempts to compile the model for fast decoding. See configs.ForecastConfig for more details on the supported flags. Args: forecast_config: Configuration for forecasting flags. **kwargs: Additional keyword arguments to pass to model.compile().
8            timesfm.TimesFM_2p5_200M_torch.save_pretrained              save_pretrained  method  Save weights in local directory. Args: save_directory (`str` or `Path`): Path to directory in which the model weights and configuration will be saved. config (`dict` or `DataclassInstance`, *optional*): Model configuration specified as a key/value d…
9                            timesfm.configs.ForecastConfig               ForecastConfig   class  Options for forecasting. Attributes: max_context: The maximum context length. This is used by the complied decode function at inference time during batched inference. Any input time series with length less than max_context will be padded with zeros,…
10              timesfm.configs.RandomFourierFeaturesConfig  RandomFourierFeaturesConfig   class                                                                                                                                                                                                      Framework-agnostic config for random fourier features.
11                      timesfm.configs.ResidualBlockConfig          ResidualBlockConfig   class                                                                                                                                                                                                             Framework-agnostic config for a residual block.
12                timesfm.configs.StackedTransformersConfig    StackedTransformersConfig   class                                                                                                                                                                                                       Framework-agnostic config for a stacked transformers.
13                        timesfm.configs.TransformerConfig            TransformerConfig   class                                                                                                                                                                                                                Framework-agnostic config for a transformer.
37  timesfm.torch.dense.RandomFourierFeatures.named_buffers                named_buffers  method  Return an iterator over module buffers, yielding both the name of the buffer as well as the buffer itself. Args: prefix (str): prefix to prepend to all buffer names. recurse (bool, optional): if True, then yields buffers of this module and all submo…
64      timesfm.torch.dense.RandomFourierFeatures.zero_grad                    zero_grad  method  Reset gradients of all model parameters. See similar function under :class:`torch.optim.Optimizer` for more context. Args: set_to_none (bool): instead of setting to zero, set the grads to None. See :meth:`torch.optim.Optimizer.zero_grad` for details…

```

TimesFM (Time Series Foundation Model) の機能概要と、各プロセス（ロード、保存、解析、可視化など）に対応するAPIや方法を整理した一覧表を作成しました。

### TimesFM 機能概要

Google Researchが開発した時系列予測のための基盤モデル（Foundation Model）です。

*   **アーキテクチャ**: Transformerベース（Decoder-only）
*   **主な特徴**: ゼロショット予測（学習なしで未知のデータに対し予測可能）、可変長のコンテキストと予測ホライゾンへの対応。
*   **最新モデル**: `timesfm-2.0-500m-pytorch` (500M params), `timesfm-1.0-200m` など。

### 機能別 API・対応一覧表

以下は、Pythonライブラリ（`timesfm`）および一般的に行われる処理ごとの機能分類です。

| 分類  | イベント・機能 | 概要  | 対応する主なAPI・方法 |
| --- | --- | --- | --- |
| **ロード** | **モデル読み込み** | Hugging Face Hubまたはローカルから事前学習済みモデルをロードします。 | `TimesFM.from_pretrained()`<br><br>  <br><br>`TimesFM.load_checkpoint()` |
| **設定** | **予測設定** | コンテキスト長、予測期間、量子化ヘッドの使用などを設定します。 | `timesfm.ForecastConfig`<br><br>  <br><br>`timesfm.configs.ForecastConfig` |
| **予測** | **単変量予測** | 過去の系列データから将来の値を予測します（点予測）。 | `TimesFM.forecast()`<br><br>  <br><br>`TimesFM.forecast_on_df()` |
|     | **外部変数付き予測** | 静的・動的な外部共変量（Covariates）を考慮して予測します（JAX/XReg依存）。 | `TimesFM.forecast_with_covariates()` |
|     | **多変量対応** | 基本は単変量モデルですが、多変量データは各変数を独立して処理（Independent processing）することで対応します。 | `forecast()` に多変量DataFrameを渡す（各列を独立予測） |
| **最適化** | **高速化・コンパイル** | `torch.compile` を利用して推論速度を最適化します。 | `TimesFM.compile()` |
| **保存** | **モデル保存** | モデルの重みと設定をローカルディレクトリに保存します。 | `TimesFM.save_pretrained()` |
|     | **状態保存** | 学習・推論の状態（State Dict）を保存・ロードします。 | `save_state_dict()`, `load_state_dict()` (PyTorch標準) |
| **分析・評価** | **特徴量抽出** | **(要実装)** モデルを特徴量抽出器（Backbone）として利用し、埋め込みベクトルを取得します。専用APIはないため、内部モジュールの出力を取得します。 | `model.model.forward()` 等でHidden Statesを取得<br><br>  <br><br>(または `register_forward_hook` 利用) |
|     | **評価 (Metrics)** | ライブラリ内に評価関数は含まれません。予測結果（配列/DF）に対し、外部ライブラリで計算します。 | `sklearn.metrics` (MSE, MAE等)<br><br>  <br><br>`utilsfunc` 等で自作 |
|     | **異常検知** | (BigQuery版には `AI.DETECT_ANOMALIES` があるが、Python版は予測値と実測値の乖離を見て自作が必要) | `forecast()` 結果と実測値の残差計算 |
| **可視化** | **プロット** | ライブラリ内にプロット機能は含まれません。PandasやMatplotlibを使用します。 | `matplotlib.pyplot`<br><br>  <br><br>`pandas.DataFrame.plot()` |
| **学習** | **ファインチューニング** | **(要実装)** 特定ドメインデータへの適応（Few-shot / Fine-tuning）。PyTorchの標準的な学習ループを構築して行います。 | `model.train()`<br><br>  <br><br>標準的なPyTorch Training Loop |

### 補足: 多変量とベクトル抽出について

*   **多変量 (Multivariate)**: TimesFM自体は**単変量 (Univariate)** モデルとして設計されています。多変量データを入力した場合、内部的には「チャネル独立 (Channel Independence)」戦略を取り、各変数を個別の単変量時系列として並列に予測します。変数間の相関を直接モデリングするAttention機構（Cross-variable attentionなど）は持ちません。
*   **ベクトル抽出 (Feature Extraction)**: 公式APIとして `get_embedding()` のようなメソッドは提供されていませんが、PyTorchモデルであるため、`timesfm.torch.transformer` などの下位モジュールにフックを仕掛けることで、最終層や中間層のベクトル表現（Embedding）を取り出し、分類やクラスタリングなどのダウンストリームタスクに流用可能です。

### 具体的なコード例（ロード〜予測）

```
import timesfm

# 1. モデルのロード (Loading)
tfm = timesfm.TimesFM(
    context_len=512,       # 入力コンテキスト長
    horizon_len=128,       # 予測期間
    input_patch_len=32,
    output_patch_len=128,
    num_layers=20,
    model_dims=1280,
    backend="cpu"          # "gpu" or "cpu"
)
tfm.load_from_checkpoint(repo_id="google/timesfm-1.0-200m")

# 2. 予測 (Forecasting)
# inputs: リストまたは配列 [series1, series2, ...]
# freq: 頻度設定 (0:高頻度, 1:中頻度, 2:低頻度)
forecast_result = tfm.forecast(inputs=[[0.1, 0.2, 0.3, ...]], freq=[0])

# 3. 解析・可視化 (Analysis & Visualization) - 外部ライブラリ連携
import pandas as pd
import matplotlib.pyplot as plt

# 結果のプロット
plt.plot(forecast_result[0])
plt.show()
```



---
Powered by [Gemini Exporter](https://www.ai-chat-exporter.com)