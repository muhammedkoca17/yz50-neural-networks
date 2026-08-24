# 🧠 YZ50 - Neural Networks: Zero to Hero (Pure Python Implementation)

Bu proje, **PyTorch**, **NumPy** veya **TensorFlow** gibi harici hiçbir yüksek seviyeli kütüphane kullanılmaksızın, tamamen saf Python veri yapıları (`list`, `float`) ve temel matematiksel prensipler aracılığıyla sıfırdan inşa edilmiş bir yapay sinir ağı (Artificial Neural Network) simülasyonudur.

Projenin temel amacı; Andrej Karpathy'nin *Micrograd* yaklaşımı ve 3Blue1Brown'un sezgisel *Gradient Descent* görselleştirmelerinden esinlenerek, modern derin öğrenme çatılarının arka planında gerçekleşen ileriye doğru hesaplama (**Forward Pass**), kayıp ölçümü (**MSE Loss**), duyarlılık analizi (**Loss Curve / Dead ReLU**) ve parametre optimizasyonu (**Numerical Gradient Descent**) mekanizmalarını şeffaf bir şekilde ortaya koymaktır.

---

## 📌 İçindekiler
- [Proje Mimarisi ve İçerik](#-proje-mimarisi-ve-i̇çerik)
- [Matematiksel Teori ve Adım Adım İnceleme](#-matematiksel-teori-ve-adım-adım-i̇nceleme)
  - [Adım 1: Saf Python ile Tek Nöron Forward Pass](#adım-1-saf-python-ile-tek-nöron-forward-pass)
  - [Adım 2: Katman Mimari Yapısı (Layer Forward Pass)](#adım-2-katman-mimari-yapısı-layer-forward-pass)
  - [Adım 3: Loss Fonksiyonu (Mean Squared Error - MSE)](#adım-3-loss-fonksiyonu-mean-squared-error---mse)
  - [Adım 4: Parametre Değişimi ve Loss Eğrisi (Optimization Landscape)](#adım-4-parametre-değişimi-ve-loss-eğrisi-optimization-landscape)
  - [Adım 5: Sayısal Türev ve Gradient Descent Döngüsü](#adım-5-sayısal-türev-ve-gradient-descent-döngüsü)
- [Deney Sonuçları ve Başarı Metrikleri](#-deney-sonuçları-ve-başarı-metrikleri)
- [Kurulum ve Kullanım](#-kurulum-ve-kullanım)
- [Kaynaklar ve İlgili Çalışmalar](#-kaynaklar-ve-i̇lgili-çalışmalar)

---

## 🏗 Proje Mimarisi ve İçerik

Repository içerisinde yer alan ana çalışma dosyası:
* **`neural_network.ipynb`**: Adım adım teorik açıklamaları, matematiksel notasyonları ve doğrulanmış Python kod bloklarını içeren Jupyter Notebook dosyası.

---

## 📐 Matematiksel Teori ve Adım Adım İnceleme

### Adım 1: Saf Python ile Tek Nöron Forward Pass

Biyolojik sinir hücrelerinden esinlenen yapay nöron (Perceptron); girdi vektörünü alan, bu girdileri öğrenilebilir ağırlık parametreleriyle ölçekleyip sapma (bias) değerini ekleyen ve ardından doğrusal olmayan bir süzgeçten (aktivasyon) geçiren matematiksel bir işlevdir.

**Genel Formül:**
$$y = f\left( \sum_{i=1}^{n} (x_i \cdot w_i) + b \right)$$

* **Ağırlıklı Toplam ($z$):** $z = \mathbf{w}^T \mathbf{x} + b = \sum_{i=1}^{n} (x_i \cdot w_i) + b$
* **Aktivasyon (ReLU):** $\text{ReLU}(z) = \max(0, z)$

```python
def relu(z):
    return z if z > 0 else 0

def single_neuron_forward(inputs, weights, bias):
    weighted_sum = sum(x * w for x, w in zip(inputs, weights))
    z = weighted_sum + bias
    return relu(z)
