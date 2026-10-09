# Changelog

本專案的版本格式遵循 [Semantic Versioning](https://semver.org/lang/zh-TW/)。

## [Unreleased]

### Added

- 英文版 README（`README.md`，GitHub 預設顯示），繁體中文版在 `README.zh-TW.md`。

## [0.1.0] - 2026-09-22

第一個版本。

### Added

- `symloc.combiners`：$e_k$ 的 $O(Nk)$ 對數域遞推，以及 `esp` / `product` / `sum` /
  `trimmed` / `median` / `softmin` 六種可註冊的組合策略。
- `symloc.likelihood`：單 AP 高斯似然，全程對數域，完全向量化。
- `symloc.pathloss`：對數距離路徑損耗與對數常態陰影衰落，可產生非退化的無線電地圖。
- `symloc.normalize`：`sum` / `density` / `none` 三種正規化模式，並保留 `log_evidence`
  與網格截斷診斷。
- `symloc.metrics`：KL（雙向）、JS、Bhattacharyya、Hellinger、全變差、Shannon/Rényi 熵、
  FWHM、超額峰度、峰值與次峰比、有效支撐寬度、定位誤差。
- `symloc.montecarlo`：重複試驗與百分位信賴區間，亂數種子固定。
- `symloc.detection`：竄改偵測的特徵映射與分層 k 折交叉驗證，回報 UAR、靈敏度、特異度、
  混淆矩陣與 ROC-AUC。
- `symloc.sweep`：攻擊強度 × sigma × AP 數 × $M$ × 組合器的多維掃描。
- `symloc.legacy`：原始公式的忠實重現，含正負號缺陷開關，作為回歸比對基準。
- `symloc.figio`：MATLAB `.fig`（MAT-file v5）唯讀讀取工具，能辨識掛錯副檔名的 EPS。
- `symloc.io_`：以設定內容雜湊命名的輸出資料夾，附完整 run manifest。
- `symloc.cli`：`run` / `sweep` / `figure` / `reproduce-legacy` / `inspect-fig` /
  `prepare` / `train` / `evaluate` / `predict` 九個子指令。
- 六份設定檔、三支端到端示範腳本、578 項測試（含 doctest）。

### Fixed（相對於原始作法）

- $e_2$ 展開式中的正負號錯誤，該錯誤使封存圖中的 $M=1$ 峰值高估 7.59%。
- 以線性域連乘造成的下溢，改為對數域遞推。
- 截斷正規化造成的假象（被攻擊曲線峰值高於無攻擊基準）。
- 座標軸標籤與資料不一致、圖例拼字錯誤、繪圖函式寫死 4 條曲線。
- 檔名與參數不符：改由設定內容雜湊產生資料夾名稱，並一律寫出 manifest。
