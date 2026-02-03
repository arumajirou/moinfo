import sqlite3
import pandas as pd
import itertools
import json
import os
import torch
from datetime import datetime
from typing import List, Dict, Any

# Chronosライブラリのインポート（環境に合わせてパスを通してください）
try:
    from chronos import Chronos2Pipeline
    # 学習用クラスなどは実際の実装に合わせてimport
except ImportError:
    print("⚠️ Chronos library not found. Mocking for structure demonstration.")

class ExperimentRegistry:
    """メタ実行表（DB）を管理するクラス"""
    def __init__(self, db_path="experiments.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """テーブル初期化"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        # 実験管理テーブル
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT,
                task_type TEXT,          -- 'zero_shot', 'finetune', 'embedding', 'analysis'
                use_covariates BOOLEAN,  -- 外生変数を使用するか
                context_length INTEGER,
                prediction_length INTEGER,
                num_samples INTEGER,     -- 確率予測のサンプル数
                cross_learning BOOLEAN,  -- アイテム間クロス学習/推論
                status TEXT DEFAULT 'TODO', -- 'TODO', 'RUNNING', 'DONE', 'ERROR'
                result_metrics JSON,     -- 評価結果 (MSE, WQLなど)
                output_path TEXT,        -- 保存先パス
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()

    def register_grid(self, param_grid: Dict[str, List[Any]]):
        """グリッドサーチの組み合わせを生成し、未登録ならDBに追加"""
        keys = param_grid.keys()
        combinations = itertools.product(*param_grid.values())
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        count = 0
        for combo in combinations:
            params = dict(zip(keys, combo))
            
            # 重複チェック（同じ設定がすでに存在するか）
            query = "SELECT id FROM experiments WHERE " + " AND ".join([f"{k}=?" for k in keys])
            cursor.execute(query, tuple(str(v) if isinstance(v, (list, dict)) else v for v in params.values()))
            
            if not cursor.fetchone():
                # 新規登録
                cols = ", ".join(keys)
                placeholders = ", ".join(["?"] * len(keys))
                insert_sql = f"INSERT INTO experiments ({cols}) VALUES ({placeholders})"
                cursor.execute(insert_sql, tuple(params.values()))
                count += 1
        
        conn.commit()
        conn.close()
        print(f"✨ Registered {count} new experiments.")

    def get_next_task(self):
        """未実行(TODO)のタスクを1つ取得してRUNNINGにする"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM experiments WHERE status='TODO' LIMIT 1")
        row = cursor.fetchone()
        
        if row:
            task = dict(row)
            cursor.execute("UPDATE experiments SET status='RUNNING', updated_at=CURRENT_TIMESTAMP WHERE id=?", (task['id'],))
            conn.commit()
            conn.close()
            return task
        
        conn.close()
        return None

    def update_task_result(self, task_id, status, metrics=None, output_path=None):
        """実行結果を保存"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        metrics_json = json.dumps(metrics) if metrics else None
        cursor.execute('''
            UPDATE experiments 
            SET status=?, result_metrics=?, output_path=?, updated_at=CURRENT_TIMESTAMP 
            WHERE id=?
        ''', (status, metrics_json, output_path, task_id))
        
        conn.commit()
        conn.close()

class ChronosExecutor:
    """Chronos-2の各機能を実行するクラス"""
    
    def __init__(self, data_dir="libs/chronos/00_raw"):
        self.data_dir = data_dir
    
    def load_data(self, use_covariates: bool):
        """データの読み込みと前処理（Mock）"""
        # ここで pandas.read_csv 等を行い、Chronos2Dataset形式に変換
        print(f"   Dataset loading... (Covariates: {use_covariates})")
        return {"train": None, "test": None} # 実装時はDataFrame等を返す

    def execute(self, task: Dict[str, Any]):
        """タスクタイプに応じた処理の振り分け"""
        task_id = task['id']
        task_type = task['task_type']
        
        print(f"🚀 Processing Task ID: {task_id} | Type: {task_type}")
        
        try:
            # 1. データの準備
            data = self.load_data(task['use_covariates'])
            
            # 2. 機能別実行
            if task_type == 'zero_shot':
                result = self._run_zero_shot(task, data)
            elif task_type == 'finetune':
                result = self._run_finetune(task, data)
            elif task_type == 'embedding':
                result = self._run_embedding(task, data)
            elif task_type == 'analysis':
                result = self._run_covariate_analysis(task, data)
            else:
                raise ValueError(f"Unknown task type: {task_type}")
                
            return "DONE", result, f"outputs/{task_id}"
            
        except Exception as e:
            print(f"❌ Error in task {task_id}: {e}")
            import traceback
            traceback.print_exc()
            return "ERROR", {"error": str(e)}, None

    def _run_zero_shot(self, task, data):
        """ゼロショット予測 & 評価"""
        print("   Running Zero-shot Inference...")
        # pipeline = Chronos2Pipeline.from_pretrained(task['model_name'])
        # preds = pipeline.predict(...)
        # metrics = calculate_metrics(preds, data['test'])
        return {"mse": 0.05, "wql": 0.02} # Mock result

    def _run_finetune(self, task, data):
        """モデルの再学習 & 保存"""
        print("   Running Fine-tuning...")
        # trainer = Chronos2Trainer(...)
        # trainer.train()
        # trainer.save_model(...)
        return {"training_loss": 0.01, "validation_loss": 0.02}

    def _run_embedding(self, task, data):
        """埋め込みベクトルによる特徴量作成"""
        print("   Extracting Embeddings...")
        # model = Chronos2Model.from_pretrained(...)
        # embeddings = model.encode(data['train'])
        return {"embedding_shape": [100, 768], "saved_at": "embeddings.pt"}

    def _run_covariate_analysis(self, task, data):
        """外生変数の寄与率解析（Sensitivity Analysis）"""
        print("   Analyzing Covariate Contribution...")
        # 1. 全変数ありで予測
        # 2. 特定の外生変数を0またはランダムにして予測
        # 3. 予測結果のズレ（Delta）を寄与率とする
        return {"covariate_importance": {"price": 0.4, "temperature": 0.1}}

# --- メイン実行ブロック ---
if __name__ == "__main__":
    # 1. DB管理マネージャーの初期化
    registry = ExperimentRegistry()
    
    # 2. グリッドリサーチの設定（実行したい全パターン）
    param_grid = {
        "model_name": ["amazon/chronos-t5-small"],
        "task_type": ["zero_shot", "finetune", "embedding", "analysis"],
        "use_covariates": [True, False],
        "context_length": [512],
        "prediction_length": [24],
        "cross_learning": [True, False] # アイテム間学習の有無
    }
    
    # 3. 未登録の実験をDBに登録（差分のみ追加）
    registry.register_grid(param_grid)
    
    # 4. エグゼキュータの初期化
    executor = ChronosExecutor()
    
    # 5. 未実行タスクのループ実行
    while True:
        task = registry.get_next_task()
        if not task:
            print("🎉 All tasks completed!")
            break
            
        status, metrics, output_path = executor.execute(task)
        registry.update_task_result(task['id'], status, metrics, output_path)

    # 6. 結果確認（簡易表示）
    conn = sqlite3.connect("experiments.db")
    df = pd.read_sql("SELECT * FROM experiments", conn)
    print("\n=== 📊 Meta Execution Table Status ===")
    print(df[['id', 'task_type', 'use_covariates', 'status', 'result_metrics']].to_string())
    conn.close()