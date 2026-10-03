# Contributing

## 開發環境

```bash
python -m pip install -e ".[dev,parquet]"
```

## 送出修改前

```bash
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
```

三個都要通過。

## 專案的幾條硬性規則

1. **目標是 Python 3.10。** 本機直譯器較新，所以測試通過不代表 3.10 能跑。
   `tests/test_python310_compat.py` 會用 `ast.parse(feature_version=(3, 10))`
   檢查語法，並掃描 3.11+ 的標準庫名稱與 NumPy 2.0 移除的別名。新增模組時請一併
   加上 `from __future__ import annotations`。

2. **測試只用程式合成的 fixture。** 不要在測試中讀取任何真實資料檔。

3. **數值計算走對數域。** 似然的組合一律使用 `symloc.combiners` 的對數域遞推；
   線性域連乘在 AP 數一多就會下溢。

4. **不要用 `print` 當輸出機制。** 診斷訊息走 `logging`，結果走 `symloc.io_`，
   只有 CLI 與 `examples/` 可以直接印到終端機。

5. **新增指標時附上解析解對照。** `tests/test_metrics.py` 以兩個高斯分布的
   closed-form 結果驗證 KL 與 Bhattacharyya，新指標請比照辦理。

6. **docstring 與註解用英文**（numpy-style），README 與使用者文件用繁體中文。
