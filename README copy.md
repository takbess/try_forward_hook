# CIFAR-10 における forward hook を用いた、知識蒸留

## 概要

本実験では **ImageNet 事前学習済み ResNet** を用い、CIFAR-10 において以下を比較した。

* 通常学習（ResNet18 / ResNet34）
* 特徴量ベースの Knowledge Distillation（KD）
* **Cross-Head Knowledge Distillation（CrossKD）**

特に **ResNet34 → ResNet18** の蒸留設定において、CrossKD が学習効率・最終精度の両面で大きな改善を示すことを確認した。

---

## 実験設定

* Dataset: CIFAR-10
* Backbone: ResNet18 / ResNet34
* Pretraining: ImageNet
* Optimizer / LR: 
* Epochs: 10

---

## 実験結果

### 1. 通常学習

#### ResNet18

* 10 epoch 時点 Test Accuracy: **74.75%**

#### ResNet34

* 10 epoch 時点 Test Accuracy: **72.55%**

> 深いモデル（ResNet34）でも、短い epoch 数では ResNet18 を上回らない。

---

### 2. Knowledge Distillation（ResNet34 → ResNet18）

#### Feature KD

* Test Accuracy: **82.50%**
* 通常学習に対して +7〜8% の改善

> Teacher の CIFAR-10 学習が完全ではない点に注意（今後の改善点）。

---

### 3. CrossKD（ResNet34 → ResNet18）

* Test Accuracy: **85.44%（最高）**
* Train Accuracy: **86.77%（10 epoch）**

**特徴**

* 初期 Loss は大きいが、急速に収束
* epoch7 前後で一時的な Test Acc の低下が発生

  * head / feature alignment の揺れと考えられる
* その後、一気に性能が向上

---

## 学習曲線の比較

（※ ここに Loss / Train Acc / Test Acc のグラフ画像を貼る）

### Loss

* CrossKD が最も速く低下
* **CrossKD > KD > 通常学習** の関係が明確
* 「最初は難しいが、正しい方向に強く引っ張られる」挙動

### Train Accuracy

* 序盤から
  **CrossKD > KD > ResNet18 > ResNet34**
* 学習効率が顕著に改善

### Test Accuracy（最重要）

| Model                  | Test Acc |
| ---------------------- | -------- |
| ResNet18               | ~75%     |
| ResNet34               | ~73%     |
| ResNet18 + KD          | ~82%     |
| **ResNet18 + CrossKD** | **~85%** |

---

## 考察

* CrossKD は

  * feature alignment
  * head 間の知識伝播
    を同時に促進している可能性が高い

* 通常 KD よりも

  * 収束が速い
  * 最終性能が高い

* 軽量モデル（ResNet18）の性能底上げに非常に有効

---

## 今後の課題

* Teacher（ResNet34）を CIFAR-10 で十分に学習させた再実験
* epoch 数を増やした場合の最終性能
* 他アーキテクチャ（ViT, MobileNet 等）への適用
* CrossKD の損失重み・アラインメント位置のアブレーション

---

## まとめ

**CrossKD は CIFAR-10 + ImageNet 事前学習環境において**

* 学習速度
* 学習安定性
* 最終 Test Accuracy

のすべてで通常学習・通常 KD を上回る結果を示した。

👉 **「軽量モデルを強くしたい」用途では、非常に有望な蒸留手法**である。
