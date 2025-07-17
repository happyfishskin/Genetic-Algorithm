# 使用 DEAP 的基因演算法優化專案

這是一個使用 Python 和 `deap` 函式庫來展示基因演算法（Genetic Algorithm, GA）在解決不同類型單目標優化問題的專案。專案包含了四個獨立的腳本，分別針對兩種不同的測試函數（Sphere 和 Schwefel），並採用了不同的基因演算法策略組合。

## 核心概念

本專案旨在探索基因演算法在不同特性函數上的表現，並比較不同運算元（如交叉、變異、選擇）組合的效果。

### 測試函數

1.  **Sphere Function (球狀函數)**
    * **特性**: 一個簡單、凸、單峰（unimodal）的函數。它的最小值在 (0, 0, ..., 0)。
    * **挑戰**: 雖然簡單，但可用於測試演算法的基本收斂能力。
    * **腳本**: `one_g1.py`, `one_g2.py`

2.  **Schwefel Function (施韋費爾函數)**
    * **特性**: 一個複雜、非凸、多峰（multimodal）的函數，具有大量的局部最小值。
    * **挑戰**: 全域最小值與次優的局部最小值距離很遠，對演算法的全局搜索能力和避免早熟收斂的能力構成極大挑戰。
    * **腳本**: `mul_g.py`, `mul_g2.py`

## 檔案說明

本專案包含四個主要的 Python 腳本，每個腳本都實現了一個完整的基因演算法來解決一個特定的優化問題。

### `one_g1.py`
* **目標函數**: Sphere Function (10 維)。
* **基因演算法策略**:
    * **交叉 (Crossover)**: `cxBlend` (混合交叉)。
    * **變異 (Mutation)**: `mutGaussian` (高斯變異)。
    * **選擇 (Selection)**: `selTournament` (錦標賽選擇)。
* **特點**: 採用了適合連續實數空間的運算元組合。

### `one_g2.py`
* **目標函數**: Sphere Function (30 維)。
* **基因演算法策略**:
    * **交叉 (Crossover)**: `cxTwoPoint` (兩點交叉)。
    * **變異 (Mutation)**: `mutGaussian` (高斯變異)。
    * **選擇 (Selection)**: `selTournament` (錦標賽選擇)。
* **特點**: 與 `one_g1.py` 相比，使用了不同的交叉策略來觀察其對收斂過程的影響。

### `mul_g.py`
* **目標函數**: Schwefel Function (30 維)。
* **基因演算法策略**:
    * **交叉 (Crossover)**: `cxBlend` (混合交叉)。
    * **變異 (Mutation)**: `mutGaussian` (高斯變異)。
    * **選擇 (Selection)**: `selTournament` (錦標賽選擇)。
* **特點**: 引入了**精英保留 (Elitism)** 策略，確保每一代的最佳個體不會在交叉和變異中丟失，有助於在複雜問題中穩定收斂。

### `mul_g2.py`
* **目標函數**: Schwefel Function (30 維)。
* **基因演算法策略**:
    * **交叉 (Crossover)**: `cxOnePoint` (單點交叉)。
    * **變異 (Mutation)**: `mutUniformInt` (均勻整數變異)。
    * **選擇 (Selection)**: `selRoulette` (輪盤賭選擇)。
* **特點**: 採用了一套與 `mul_g.py` 完全不同的經典基因演算法運算元，用於對比不同策略在解決複雜多峰問題上的效果。

## 環境依賴

您需要安裝以下 Python 函式庫才能執行此專案。建議在虛擬環境中進行安裝。

```bash
pip install deap numpy matplotlib
```

## 如何執行

您可以獨立執行任何一個腳本來觀察其優化過程和結果。

```bash
# 執行 Sphere 函數優化 (策略一)
python one_g1.py

# 執行 Sphere 函數優化 (策略二)
python one_g2.py

# 執行 Schwefel 函數優化 (策略一，含精英保留)
python mul_g.py

# 執行 Schwefel 函數優化 (策略二)
python mul_g2.py
```

## 視覺化結果

每個腳本在執行完畢後都會生成兩種視覺化圖表：

1.  **適應度收斂曲線**: 顯示最佳個體的適應度（目標函數值）隨著演化世代數的變化趨勢。這有助於分析演算法的收斂速度和穩定性。
2.  **3D 函數表面與族群分佈圖**:
    * 繪製出目標函數在二維空間的 3D 表面圖。
    * 將演化結束時的族群個體（藍色點）和找到的最佳個體（紅色點）標示在函數表面上，直觀地展示族群的探索範圍和最終收斂位置。

