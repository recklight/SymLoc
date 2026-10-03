# SymLoc

[![CI](https://github.com/recklight/SymLoc/actions/workflows/ci.yml/badge.svg)](https://github.com/recklight/SymLoc/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

室內 Wi-Fi 指紋定位中，抗竄改似然函數家族的機率分布分析。

當定位系統把多個存取點（AP）的訊號強度和指紋資料庫比對時，最常用的作法是把每個 AP 的高斯似然值連乘起來。這在所有 AP 都誠實回報時是最佳解，但只要有**一個** AP 被竄改，整個乘積就會在所有候選位置上一起崩塌，定位結果隨之失效。

本專案研究的組合方式改用「任取 $N-M$ 個 AP 相乘後全部相加」取代連乘。這個式子的正式名稱是**基本對稱多項式**（elementary symmetric polynomial，以下簡稱 ESP），而 $M$ 正是「這個分數設計上能容忍幾個被竄改的 AP」。

---

## 目錄

- [專案簡介](#專案簡介)
- [方法說明](#方法說明)
- [安裝步驟](#安裝步驟)
- [快速開始](#快速開始)
- [使用範例](#使用範例)
- [CLI 指令對照表](#cli-指令對照表)
- [專案結構](#專案結構)
- [資料準備](#資料準備)
- [設定檔](#設定檔)
- [結果與評估指標](#結果與評估指標)
- [Improvements](#improvements)
- [開發與測試](#開發與測試)
- [References](#references)

---

## 專案簡介

整份分析是**純解析式**的：所有數值都由模型即時算出，不需要任何資料檔、不需要下載、也不需要實際量測。這讓每一項結論都可以被完整重現。

| 階段 | 做什麼 | 對應模組 |
| --- | --- | --- |
| 1. 候選網格 | 建立候選位置，可以是平均訊號強度掃描，也可以是真實參考點 | `symloc.grid` |
| 2. 無線電地圖 | 由路徑損耗模型算出每個 AP 在每個候選位置的期望訊號強度 | `symloc.pathloss` |
| 3. 單 AP 似然 | 對每個 AP、每個候選位置計算高斯對數似然 | `symloc.likelihood` |
| 4. 組合 | 依各階 $M$ 把 $N$ 個似然值合成一個分數 | `symloc.combiners` |
| 5. 正規化 | 轉成機率分布，同時保留正規化會丟掉的資訊 | `symloc.normalize` |
| 6. 指標 | 銳度、散度、定位誤差，並附上信賴區間 | `symloc.metrics`、`symloc.montecarlo` |
| 7. 偵測 | 判斷一組觀測值究竟有沒有被竄改（交叉驗證） | `symloc.detection` |
| 8. 落地 | 圖表、表格與設定檔一起寫出，附上 run manifest | `symloc.io_` |

`symloc.legacy` 另外忠實重現了原始公式（含其中的缺陷），只作為回歸比對的基準。

---

## 方法說明

### 單 AP 的高斯似然

在對數常態陰影衰落（log-normal shadowing）模型下，AP $k$ 在候選位置 $j$ 的 dBm 讀值服從以無線電地圖預測值為中心的常態分布，因此似然為

$$
p_k(j) = \frac{1}{\sigma_k \sqrt{2\pi}}
         \exp\left[-\frac{(r_k - \mu_{kj})^2}{2\sigma_k^2}\right]
$$

其中 $r_k$ 是觀測值、$\mu_{kj}$ 是地圖預測的平均值、$\sigma_k$ 是陰影衰落的標準差（單位 dB）。

用高斯分布來描述 dBm 讀值不是便宜行事，而是模型本身的結果：陰影衰落項在 **dB 域**就是高斯的（見 `symloc.pathloss`）。

### 基本對稱多項式組合器

ESP($M$) 的定義是：從 $N$ 個 AP 中任取 $N-M$ 個組成子集合，子集合內相乘，再把所有子集合加總：

$$
\mathrm{ESP}(M) \;=\;
\sum_{\substack{S \subseteq \{1,\dots,N\} \\ |S| = N-M}} \; \prod_{k \in S} p_k
\;=\; e_{N-M}(p_1, \dots, p_N)
$$

也就是 $N-M$ 階的**基本對稱多項式** $e_{N-M}$。把這個名字叫出來是整個專案的關鍵：原始作法是把每一階手動展開成一長串代數式，一種 AP 個數就得重寫一次，所以分析永遠被綁死在 3 個和 4 個 AP 的情況。認出它是 $e_k$ 之後，任意 $N$ 都可以用 $O(Nk)$ 的遞推一次算完。

兩個極端很好理解：

- $M = 0$ 得到 $e_N = \prod_k p_k$，就是標準的 naive-Bayes 指紋定位似然——**一個** AP 被竄改就會讓它趨近於零；
- $M = N-1$ 得到 $e_1 = \sum_k p_k$，單純加總，任何單一 AP 都無法否決它，但它的峰形也非常平坦。

所以 $M$ 就是**穩健度與銳度之間的取捨旋鈕**。

### 遞推式與對數域

遞推來自下列多項式的係數展開：

$$
\prod_{k=1}^{N} (x + p_k) = \sum_{j=0}^{N} e_j(p)\, x^{N-j}
$$

本專案在**對數域**執行這個遞推：乘法變成加法，加法變成 `numpy.logaddexp`。這件事很重要——在線性域下，15 個 AP 的機率密度連乘在 float64 中會直接下溢為 0，正規化之後整條曲線變成 `nan`；對數域遞推則全程保持精確。`tests/test_combiners.py` 用 `itertools.combinations` 的暴力枚舉法驗證遞推結果，並示範了線性域崩潰、對數域存活的情形。

### 路徑損耗與真實無線電地圖

對數距離路徑損耗模型：

$$
\mu(d) = P_{d_0} - 10\,n \log_{10}\!\left(\frac{d}{d_0}\right)
$$

其中 $n$ 是路徑損耗指數（自由空間約 2，室內隔間約 3 到 5）。把 AP 擺在樓層平面上之後，每個 AP 在每個參考點都有**自己的**期望值，候選座標軸才真的帶有幾何意義，定位誤差也才能用公尺來衡量。

原始分析假設所有 AP 在同一個候選位置共用同一個平均值，所以那條座標軸其實只是一維的平均訊號強度掃描，並不是樓層上的位置。本專案兩種都支援（`radio_map.kind` 設為 `uniform` 或 `pathloss`），而且會在 run manifest 中明確標註退化與否。

### 正規化：以及正規化丟掉了什麼

把曲線除以自身總和會得到機率質量函數（pmf），這讓不同曲線的**形狀**可以互相比較。但它同時毀掉兩樣論證真正需要的東西，本專案兩樣都保留：

**絕對量級。** 「竄改 AP 會讓指紋比對崩潰」這個主張，講的是絕對似然值的變化。把每條曲線各自除以自己的總和，正好把這個差異完全抹掉——被攻擊與未被攻擊的曲線最後都總和為 1。因此 `Curve` 會記錄 `log_evidence`，也就是未正規化總和的對數。

**掉出網格外的機率質量。** 當觀測值遠在掃描範圍之外（例如被竄改成 −99 dBm，而網格只到 −85 dBm），該 AP 的大部分似然質量都在網格之外，正規化就會依「掉出去多少」對它任意縮放。放著不管會製造出假結果：用原始腳本留下的參數實算，被攻擊的 naive-Bayes 曲線峰值**高於**未被攻擊的基準。`truncated_mass()` 會量化它，`check_truncation()` 會發出警告。

此外，除以總和得到的是 pmf，峰值會隨網格解析度縮放；除以「總和 × 步長」得到的才是密度，峰值不隨解析度改變。兩種模式都提供，而且用了哪一種、步長是多少，都會跟著結果一起寫進 manifest。

---

## 安裝步驟

需要 Python 3.10 以上。

```bash
git clone https://github.com/recklight/SymLoc.git
cd SymLoc
python -m pip install -e ".[dev]"
```

選用的額外相依套件：

```bash
python -m pip install -e ".[parquet]"   # 結果表格額外輸出 Parquet（CSV 一定會寫）
python -m pip install -e ".[dl]"        # PyTorch；本套件核心功能完全不需要
```

不想安裝也可以直接從原始碼樹執行：

```bash
PYTHONPATH=src python -m symloc.cli --help
```

---

## 快速開始

```bash
# 重現封存結果，並量化原始碼中的正負號錯誤
symloc reproduce-legacy

# 跑一個情境，輸出曲線、指標、圖檔與 manifest
symloc run --config configs/one_attack.yaml

# 掃描攻擊強度 × sigma × AP 數 × M，並畫出鑑別度熱圖
symloc sweep --config configs/sweep_attack_strength.yaml --metric js_divergence

# 訓練竄改偵測器並做分層交叉驗證
symloc evaluate
```

不帶任何參數就能跑完的端到端示範：

```bash
python examples/end_to_end_synthetic.py
```

---

## 使用範例

### CLI

```bash
# 1. 準備：由合成試驗產生有標籤的竄改偵測資料集
symloc prepare --set detection.n_trials_per_class=500

# 2. 訓練：擬合偵測器並存檔
symloc train --output results/detector.joblib

# 3. 評估：分層 k 折交叉驗證，輸出完整分類指標
symloc evaluate --set detection.model=random_forest

# 4. 推論：對一組觀測值給出判斷
#    dBm 恆為負值，所以可以直接給絕對值；本例即 -70, -70, -110 dBm
symloc predict 70,70,110 --model results/detector.joblib
```

任何設定都可以用 `--set 鍵=值` 當場覆寫，不必改設定檔：

```bash
symloc run --set grid.step_db=0.25 --set normalize.mode=density --set scenario.combiner=trimmed
```

### Python API

```python
from symloc import Config, analyse_scenario, evidence_collapse

# 設定即輸入：不需要改任何一行原始碼
config = Config.from_yaml("configs/one_attack.yaml")
result = analyse_scenario(config)

# 每一階 M 的峰值
for m, peak in sorted(result.peaks().items()):
    print(f"M={m}: {peak:.7f}")

# 相鄰階的峰值比（原始分析用的鑑別度指標）
print(result.adjacent_peak_ratios())

# 正規化會抹掉、但論證最需要的量：絕對似然的變化（單位 nat）
print(evidence_collapse(result))

# 完整指標表（tidy long format）
print(result.metrics.head())
```

竄改偵測：

```python
from symloc import Config, build_dataset, cross_validate

config = Config().override({"detection.model": "logistic", "detection.n_splits": 5})
dataset = build_dataset(config)          # 合成試驗，固定亂數種子
outcome = cross_validate(dataset, config)

print(outcome.report())
# logistic: UAR 0.658 +/- 0.039, sensitivity 0.540, specificity 0.776,
#           ROC-AUC 0.734 (5-fold, seed 20120601)
print(outcome.confusion)                  # 混淆矩陣（列為真實、行為預測）
```

---

## CLI 指令對照表

| 指令 | 說明 | 主要輸出 |
| --- | --- | --- |
| `symloc run` | 跑完單一情境的整條流程 | 曲線、指標表、四張圖、設定檔、manifest |
| `symloc sweep` | 多維參數掃描 | long format 結果表、樞紐表、鑑別度熱圖 |
| `symloc figure` | 只重畫圖，不重算表格 | 圖檔 |
| `symloc reproduce-legacy` | 重現封存結果並量化其中的缺陷 | 終端機報表，可選輸出對照圖 |
| `symloc inspect-fig` | 讀取使用者自備的 MATLAB `.fig`，印出裡面的曲線 | 終端機報表 |
| `symloc prepare` | 產生有標籤的竄改偵測資料集 | 資料集表格、manifest |
| `symloc train` | 擬合偵測器並存檔 | `.joblib` 模型檔 |
| `symloc evaluate` | 分層 k 折交叉驗證 | 每折指標表、混淆矩陣圖、ROC 圖 |
| `symloc predict` | 對單一組觀測值做推論 | 終端機判斷結果 |

共用選項（並非每個子指令都有，右欄列出接受它的子指令）：

| 選項 | 說明 | 適用子指令 |
| --- | --- | --- |
| `--config`、`-c` | 指定 YAML 設定檔 | `run`、`sweep`、`figure`、`prepare`、`train`、`evaluate`、`predict` |
| `--set 鍵=值`、`-s` | 覆寫單一設定，可重複使用 | 同上 |
| `--label` | 指定輸出資料夾名稱前綴 | `run`、`sweep`、`figure`、`prepare`、`evaluate` |
| `--figures` / `--no-figures` | 是否輸出圖檔 | `run`、`evaluate` |
| `--output`、`-o` | 輸出路徑 | `train`（模型檔）、`reproduce-legacy`（對照圖資料夾） |
| `--metric` | 熱圖要用的指標 | `sweep` |
| `--model`、`-m` | 要載入的模型檔 | `predict` |

---

## 專案結構

```
SymLoc/
├── configs/                      設定檔：情境、掃描軸、重現目標
│   ├── default.yaml              預設值，也是每個設定鍵的說明文件
│   ├── one_attack.yaml           3 個 AP、1 個被竄改
│   ├── two_attack.yaml           4 個 AP、2 個被竄改
│   ├── legacy_exact.yaml         封存結果的重現目標（算式已修正）
│   ├── pathloss_floor.yaml       真實樓層平面與路徑損耗地圖
│   └── sweep_attack_strength.yaml  攻擊強度 × sigma × AP 數 × M
├── data/legacy_reference/
│   └── peaks.json                從封存圖反解出的唯讀基準值（僅衍生數值）
├── examples/
│   ├── end_to_end_synthetic.py   純合成資料的端到端示範
│   ├── reproduce_legacy_and_fix.py  重現封存結果並逐項對照修正前後
│   └── combiner_robustness.py    各組合器在真實無線電地圖上的穩健度研究
├── src/symloc/
│   ├── grid.py                   候選網格（dBm 掃描或 2-D 參考點）
│   ├── ap.py                     APSpec、RadioMap、Scenario
│   ├── pathloss.py               對數距離路徑損耗與陰影衰落
│   ├── likelihood.py             單 AP 高斯似然（對數域）
│   ├── combiners.py              e_k 遞推與各種組合策略
│   ├── normalize.py              正規化模式與截斷診斷
│   ├── metrics.py                散度、銳度、定位誤差
│   ├── montecarlo.py             重複試驗與信賴區間
│   ├── detection.py              竄改偵測與分層交叉驗證
│   ├── pipeline.py               單一情境的完整流程
│   ├── sweep.py                  多維參數掃描
│   ├── legacy.py                 原始公式的忠實重現（含缺陷開關）
│   ├── figio.py                  MATLAB .fig 唯讀讀取工具
│   ├── io_.py                    結果落地與 run manifest
│   ├── config.py                 具型別與驗證的設定物件
│   ├── cli.py                    typer CLI
│   ├── logging_utils.py          logging 設定
│   ├── py.typed                  型別標記（PEP 561）
│   └── plotting/                 圖表（配色、分布圖、長條圖、熱圖）
├── tests/                        pytest，全部使用程式合成的 fixture
├── pyproject.toml
└── LICENSE
```

---

## 資料準備

**本專案不需要任何資料檔。** 所有數值都由模型即時產生，`tests/` 也只使用程式合成的 fixture。直接 clone 下來就能跑完全部內容。

以下兩種情況才需要你自己準備檔案：

### 1. 檢視自己的 MATLAB `.fig` 封存圖

`symloc inspect-fig` 會讀取你指定路徑的 `.fig` 檔，抽出裡面的 `XData`、`YData` 與圖例名稱。檔案不會被複製進專案，也不會被修改。

```bash
symloc inspect-fig /path/to/your/figure.fig
```

`.fig` 的實體是 MAT-file v5 容器。如果某個檔案其實是存成 EPS 卻掛上 `.fig` 副檔名，工具會明講它到底是什麼格式，而不是丟出難以理解的解析錯誤。

### 2. 接上真實的指紋資料庫

若要用實際量測的 radio map 取代路徑損耗模型，請把 AP 座標與參考點座標填進設定檔（參考 `configs/pathloss_floor.yaml`），或在 Python 中直接建構 `RadioMap`：

```python
import numpy as np
from symloc import Grid, RadioMap, Scenario, APSpec

grid = Grid.from_points(reference_point_coordinates)   # (n_points, 2)，單位公尺
radio_map = RadioMap(measured_means_dbm, ("AP1", "AP2", "AP3"))  # (n_ap, n_points)，單位 dBm
scenario = Scenario.build("my_site", grid, aps, radio_map=radio_map, true_index=42)
```

外部資料集（例如 UJIIndoorLoc，見 References）請自行下載並放在專案之外，路徑透過設定檔指定。本專案不散布、也不內含任何資料集。

---

## 設定檔

每一個在原始作法中寫死在腳本開頭的常數，在這裡都是設定項目。`configs/default.yaml` 逐鍵附有說明，可以直接當作參考文件。

| 區塊 | 主要內容 |
| --- | --- |
| `grid` | 候選網格的端點與步長 |
| `radio_map` | `uniform`（退化）或 `pathloss`（真實幾何）與其參數 |
| `scenario` | AP 清單、標準差、哪些被竄改、偏移量、要評估的 $M$、組合器 |
| `normalize` | `sum` / `density` / `none`，以及截斷警告門檻 |
| `montecarlo` | 試驗次數、亂數種子、信賴水準 |
| `detection` | 分類器、折數、每類試驗數、種子、攻擊強度 |
| `sweep` | 各掃描軸 |
| `output` | 輸出資料夾、圖檔格式、解析度 |

未知的鍵會直接**報錯**而不是被忽略，所以設定檔裡打錯字會立刻被發現，不會靜靜地留下一個預設值。

---

## 結果與評估指標

每次執行都會建立一個以「設定內容雜湊值」命名的資料夾，裡面包含表格、圖檔、當次完整設定的 YAML，以及記錄套件版本、直譯器、函式庫版本、亂數種子與時間戳的 `manifest.json`。**檔名永遠不可能和參數對不上，因為檔名就是從參數算出來的。**

### 分布指標

| 指標 | 說明 |
| --- | --- |
| `peak`、`peak_ratio`、`peak_difference` | 原始作法唯一使用的指標，保留作為對照 |
| `kl_to_reference`、`kl_from_reference` | KL 散度，兩個方向都報（它不對稱） |
| `js_divergence` | Jensen-Shannon 散度，對稱且有界 |
| `bhattacharyya_distance`、`hellinger_distance` | 兩個分布的重疊程度 |
| `total_variation` | 全變差距離 |
| `shannon_entropy`、`renyi_entropy_2` | 分布的不確定度 |
| `fwhm` | 半高全寬；曲線在網格內沒有降回半高時回傳 `nan`（這本身就是截斷的訊號） |
| `excess_kurtosis` | 超額峰度，高斯為 0 |
| `peak_to_secondary_ratio` | 峰值與次高局部極大值之比，用來看定位是否有歧義 |
| `effective_support_90` | 涵蓋 90% 機率質量所需的最小寬度，單位與座標軸相同 |
| `log_evidence`、`log_evidence_gap` | 絕對似然，以及正規化會抹掉的那個差異 |
| `localization_error` | argmax 與真值的距離；在幾何網格上單位是公尺 |

### 分類指標（竄改偵測）

以**分層** k 折交叉驗證評估，亂數種子固定，標準化步驟放在 pipeline 內部，因此只會在訓練折上擬合：

`uar`（未加權平均召回率，主要指標）、`accuracy`、`sensitivity`、`specificity`、`ppv`、`npv`、`f1`、`roc_auc`，以及混淆矩陣。

UAR 之所以是主要指標：如果類別不平衡，一個永遠回答「沒有被攻擊」的模型可以拿到很漂亮的正確率，但它的 UAR 只有 0.5。

偵測難度會隨攻擊強度單調變化，這是模型正確的必要條件。以下用預設情境（3 個 AP、$\sigma$ 為 8/10/10 dB、logistic、5 折、每類 250 筆、種子 20120601），每一列都可以用該列的指令原樣重現：

| 攻擊偏移量 | UAR | ROC-AUC | 重現指令 |
| --- | --- | --- | --- |
| −5 dB | 0.448 | 0.431 | `symloc evaluate --no-figures --set 'detection.attack_offsets_db=[-5.0]'` |
| −15 dB | 0.624 | 0.654 | `symloc evaluate --no-figures --set 'detection.attack_offsets_db=[-15.0]'` |
| −30 dB | 0.816 | 0.904 | `symloc evaluate --no-figures --set 'detection.attack_offsets_db=[-30.0]'` |
| −45 dB | 0.958 | 0.994 | `symloc evaluate --no-figures --set 'detection.attack_offsets_db=[-45.0]'` |

−5 dB 完全落在 8 到 10 dB 的陰影衰落雜訊裡，偵測不出來是應該的——UAR 0.448 就是隨機水準（0.5）上下的抖動，不是「比隨機還差」的模型缺陷；−45 dB 遠在雜訊之外，幾乎必然被抓到。

---

## Improvements

以下逐項對應原始作法的缺口。每一項都可以用 `examples/` 底下的腳本實際看到。

### 已修正的缺陷

**1. 正負號錯誤（已污染封存的圖）。** 三個 AP 的 $e_2$ 展開式中，第三個成對乘積的觀測值少了負號，等於在 **+70 dBm** 處取機率密度，值約 $8 \times 10^{-39}$ 而非約 $7 \times 10^{-4}$，該項無聲地被歸零。封存圖中 $M=1$ 的峰值 0.0305445 與含 bug 的算式吻合到小數第七位；數學上正確的值是 0.0283886。**封存圖的峰值高估了 7.59%，而這正是整個鑑別度論證所倚賴的量。**

| 曲線 | 封存值 | 本專案重現 | 修正後 |
| --- | --- | --- | --- |
| 無攻擊基準 | 0.0387524 | 0.0387524 | 0.0387524 |
| M=0 | 0.0390380 | 0.0390380 | 0.0390380 |
| M=1 | 0.0305445 | 0.0305445 | **0.0283886** |
| M=2 | 0.0242794 | 0.0242794 | 0.0242794 |

兩個鑑別度比值也跟著改變：$\rho(0,1)$ 從 1.27807 變成 1.37513，$\rho(1,2)$ 從 1.25805 變成 1.16925。`symloc reproduce-legacy` 會把兩邊並排印出來。

**2. 截斷正規化製造出的假象。** 用原始腳本留下的參數實算，被攻擊的 $M=0$ 曲線峰值 0.0387726 **高於**無攻擊基準 0.0387524。攻擊並沒有讓定位變好；只是被竄改的 AP 有更多機率質量掉到網格外，除以比較小的總和之後，剩下的部分就被墊高了。`truncated_mass()` 會量化每個 AP 掉出去多少，超過門檻就發警告。

**3. 峰值隨網格解析度縮放，卻沒有任何地方記錄步長。** 同一個情境在不同步長下：

| 步長 | 網格點數 | M=0 峰值 |
| --- | --- | --- |
| 1.0 dB | 26 | 0.0775811 |
| 0.5 dB | 51 | 0.0390380 |
| 0.25 dB | 101 | 0.0195594 |

現在提供 `density` 模式（峰值與步長無關），而且步長一定會寫進 manifest。

**4. 檔名與參數不符。** 封存圖的檔名宣稱被攻擊 AP 為 −99 dBm，以四條曲線峰值精確反解得到的其實是 −90 dBm。現在每次執行都會寫出 run manifest，資料夾名稱直接由設定內容的雜湊值產生，檔名不可能和參數脫節。

**5. 座標軸標籤與資料不一致。** 原始繪圖函式強制設定 x 軸刻度**標籤**卻沒有設定刻度**位置**，於是標籤落在自動選出的刻度上並且循環重複，圖上標示的「Reference location 0–30」完全對應不到底層的 −60 至 −85 dBm。現在座標軸標籤與單位跟著 `Grid` 物件走，繪圖端一個字都不會自己編。

**6. 圖例拼字錯誤。** 封存圖全部寫成 `Byaesian`，現已更正為 `Bayesian`，並有測試把關。

**7. 維度寫死。** 原始繪圖樣板硬寫 `set(plot1(1))` 到 `set(plot1(4))`，AP 數不等於 4 就壞掉；`repmat(X,4,1)` 也把 4 寫死。現在配色與標記循環取用，8 個 AP、8 階 $M$ 的圖照樣畫得出來（有測試涵蓋）。

**8. 格式異常的封存檔。** 某個 `.fig` 其實是 EPS 卻掛著 `.fig` 副檔名。`symloc inspect-fig` 會先檢查檔頭魔術位元組，直接說明它究竟是什麼格式。

### 數值與方法

**9. 全程對數域計算。** 原始作法在線性域連乘機率密度，4 個 AP 時最小值已到 $10^{-8}$，AP 一多就下溢為 0。現在改用對數域的 $e_k$ 遞推配合 `logaddexp`，15 個 AP 仍然精確（有測試示範線性域已經崩潰、對數域仍然正確）。

**10. 不再手動展開組合式。** 改用 $O(Nk)$ 遞推一次算出所有階數，AP 個數任意。測試用 `itertools.combinations` 暴力枚舉驗證。

**11. 保留絕對似然。** 新增 `log_evidence` 與 `log_evidence_gap`，記錄正規化會抹掉的絕對量級差異。

**12. 真實的無線電地圖。** 新增對數距離路徑損耗模型，讓每個 AP 在每個參考點有各自的期望值。座標軸因此帶有幾何意義，定位誤差可以用公尺衡量。退化與否會記在 manifest 裡。

**13. 完整的評估指標。** 從「相鄰階峰值比與差」擴充到 KL（雙向）、JS、Bhattacharyya、Hellinger、全變差、Shannon/Rényi 熵、FWHM、超額峰度、峰值與次峰比、有效支撐寬度、定位誤差。

**14. 蒙地卡羅與信賴區間。** 原始研究是單點確定性計算，無法陳述任何變異度。現在對觀測雜訊重複抽樣，回報均值與百分位區間（不用常態近似，因為好幾個指標有界或高度偏斜），亂數種子固定。

**15. 竄改偵測與交叉驗證。** 原始作法主張了一個抗攻擊的似然家族，卻從未提出它蘊含的決策問題：**到底有沒有被攻擊？** 現在由蒙地卡羅產生平衡且有標籤的資料集，以固定的特徵映射轉成描述子，再做分層 k 折交叉驗證，回報 UAR、靈敏度、特異度、混淆矩陣與 ROC-AUC。特徵完全不含標籤、不含攻擊偏移量、也不知道哪個 AP 被竄改；其中最有鑑別力的是「在整組 AP 共識位置上，各 AP 對數似然的離散程度」（特徵 `consensus_logp_range`，在預設情境下 Cohen's $d \approx 0.88$）。

### 工程

**16. 設定檔取代裸常數。** AP 數、觀測值、標準差、攻擊情境全部由設定檔驅動，並附完整驗證（$\sigma > 0$、$M < N$、觀測值是否落在網格內、未知鍵直接報錯）。

**17. 參數化的單一流程。** 原本複製貼上的 `y1` / `y2` 兩個區塊、展開五次又四次的正規化迴圈，全部收斂成一條由設定驅動的流程。無攻擊基準也走同一條路徑，不再是另外寫死的算式。

**18. 多維掃描。** `symloc sweep` 以 `itertools.product` 掃描攻擊強度 × sigma × AP 數 × $M$ × 組合器，輸出 long format 結果表與鑑別度熱圖。

**19. 結果落地與版本管理。** 表格寫成 CSV（必定）與 Parquet（有引擎時），加上完整 manifest。原始作法的 `MP3` 算完就躺在 workspace，從未印出也從未寫檔。

**20. 測試、logging、輸入驗證、type hints。** 578 項測試（含 doctest），全部使用程式合成的 fixture；以 `logging` 取代 `print`；完整型別註解與 numpy-style docstring；另有專門的 Python 3.10 相容性閘門，用 `ast.parse(feature_version=(3, 10))` 加上標準庫與 NumPy 2.0 移除項目的掃描，因為在 3.13 上測試通過並不代表 3.10 能跑。

### 新的研究結果

**21. leave-M-out 的加總設計並沒有帶來預期的穩健度。** 這是原始作法不可能發現的，因為它的座標軸沒有幾何，根本量不出距離。

把 5 個 AP 擺在 20 × 15 公尺的樓層上，竄改其中一個，掃描 40 個真實位置 × 5 種被竄改的 AP，共 200 組組合，以各組合器**最好的**中間階 $M$ 計分。以下就是 `python examples/combiner_robustness.py` 的輸出（預設種子）：

| 組合器 | 平均誤差 | 中位數誤差 | p90 誤差 | 最大誤差 | 完全命中率 |
| --- | --- | --- | --- | --- | --- |
| `esp`（基本對稱多項式） | 0.662 m | 0.000 m | 0.00 m | 14.318 m | 91.0% |
| `trimmed`（最小截尾） | 0.000 m | 0.000 m | 0.00 m | 0.000 m | **100.0%** |
| `median` | 0.000 m | 0.000 m | 0.00 m | 0.000 m | **100.0%** |
| `softmin` | 7.031 m | 7.000 m | 11.18 m | 15.133 m | 1.0% |

原因很直接：$e_{N-M}$ 是對**所有**大小為 $N-M$ 的子集合取平均，其中仍然包含被竄改 AP 的那些子集合，而且它們保有完整權重。一個被竄改的讀值若剛好在某個錯誤位置上吻合得很好，就能在平均之後存活下來。截尾法則是**挑出**最吻合的子集合，而不是對所有子集合取平均，因此能直接把那個讀值丟掉。

soft-min 表現最差，理由是同一件事反過來說：它給**最不吻合**的 AP 最大權重，於是跟著被竄改的讀值跑。把它列進來，正是為了顯示「直覺上最保守的聚合方式」在這裡是錯的工具。

用 `python examples/combiner_robustness.py` 重現。

---

## 開發與測試

```bash
python -m pytest -q                       # 578 項測試（含 doctest）
python -m ruff check .                    # lint
python -m ruff format --check .           # 格式
```

測試分為幾類：

- **數值正確性**：遞推式對照暴力枚舉；KL 與 Bhattacharyya 對照兩個高斯分布的解析解；FWHM 對照 $2\sqrt{2\ln 2}\,\sigma$。
- **回歸比對**：`tests/test_legacy_parity.py` 驗證封存數值可被重現，並釘住缺陷的量級。
- **不變量**：無攻擊時各曲線與其對照組完全相同；攻擊越強、後驗離基準越遠；$M$ 越大越穩健但越不銳利。
- **相容性**：`tests/test_python310_compat.py` 是 Python 3.10 的閘門。
- **端到端**：每個 CLI 子指令都以使用者的方式執行過一次，並檢查它承諾的產出確實存在。

所有測試 fixture 皆由程式合成，不讀取任何真實資料檔。torch 並非必要相依，整個套件在沒有安裝 torch 的環境下可以完整匯入並通過測試。

---

## References

### Fingerprint localization

- P. Bahl and V. N. Padmanabhan, "RADAR: An In-Building RF-Based User Location and Tracking System," *Proc. IEEE INFOCOM*, 2000, pp. 775-784.
- T. Roos, P. Myllymäki, H. Tirri, P. Misikangas, and J. Sievänen, "A Probabilistic Approach to WLAN User Location Estimation," *International Journal of Wireless Information Networks*, vol. 9, no. 3, pp. 155-164, 2002.
- M. Youssef and A. Agrawala, "The Horus WLAN Location Determination System," *Proc. ACM MobiSys*, 2005, pp. 205-218.
- T. S. Rappaport, *Wireless Communications: Principles and Practice*, 2nd ed., Prentice Hall, 2002.

### Attack-resistant localization

- D. Liu, P. Ning, and W. Du, "Attack-Resistant Location Estimation in Sensor Networks," *Proc. ACM/IEEE IPSN*, 2005, pp. 99-106.
- Z. Li, W. Trappe, Y. Zhang, and B. Nath, "Robust Statistical Methods for Securing Wireless Localization in Sensor Networks," *Proc. ACM/IEEE IPSN*, 2005, pp. 91-98.
- Y. Chen, W. Trappe, and R. P. Martin, "Detecting and Localizing Wireless Spoofing Attacks," *Proc. IEEE SECON*, 2007, pp. 193-202.

### Symmetric functions and robust statistics

- I. G. Macdonald, *Symmetric Functions and Hall Polynomials*, 2nd ed., Oxford University Press, 1995, Chapter I.2.
- P. J. Rousseeuw, "Least Median of Squares Regression," *Journal of the American Statistical Association*, vol. 79, no. 388, pp. 871-880, 1984.
- P. J. Huber, *Robust Statistics*, John Wiley & Sons, 1981.

### Divergence measures

- S. Kullback and R. A. Leibler, "On Information and Sufficiency," *Annals of Mathematical Statistics*, vol. 22, no. 1, pp. 79-86, 1951.
- A. Bhattacharyya, "On a Measure of Divergence Between Two Statistical Populations Defined by Their Probability Distributions," *Bulletin of the Calcutta Mathematical Society*, vol. 35, pp. 99-109, 1943.
- T. Kailath, "The Divergence and Bhattacharyya Distance Measures in Signal Selection," *IEEE Transactions on Communication Technology*, vol. 15, no. 1, pp. 52-60, 1967.
- J. Lin, "Divergence Measures Based on the Shannon Entropy," *IEEE Transactions on Information Theory*, vol. 37, no. 1, pp. 145-151, 1991.

### Public datasets

這些資料集本專案並未內含或散布，若要使用請自行下載並放在專案之外。

- J. Torres-Sospedra et al., "UJIIndoorLoc: A New Multi-building and Multi-floor Database for WLAN Fingerprint-based Indoor Localization Problems," *Proc. International Conference on Indoor Positioning and Indoor Navigation (IPIN)*, 2014, pp. 261-270.
- "Wireless Indoor Localization" data set, UCI Machine Learning Repository.

---

## License

MIT，見 [LICENSE](LICENSE)。作者 RL。
