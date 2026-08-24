# 🧠 YZ50 - Neural Networks: (Pure Python Implementation)

Bu proje, **`PyTorch`**, **`NumPy`** veya **`TensorFlow`** gibi harici hiçbir yüksek seviyeli kütüphane kullanılmaksızın, tamamen **saf Python** veri yapıları ve temel matematiksel prensipler aracılığıyla sıfırdan inşa edilmiş bir yapay sinir ağı (Artificial Neural Network) simülasyonudur.

Projenin temel amacı; Andrej Karpathy'nin *Micrograd* yaklaşımı ve 3Blue1Brown'un sezgisel *Gradient Descent* görselleştirmelerinden esinlenerek, modern derin öğrenme mimarilerinin arka planında çalışan ileriye doğru hesaplama (**Forward Pass**), kayıp ölçümü (**MSE Loss**), kayıp yüzeyi analizi (**Optimization Landscape / Dead ReLU**) ve parametre optimizasyonu (**Numerical Gradient Descent**) mekanizmalarını en yalın ve şeffaf biçimde ortaya koymaktır.

---

## 📌 İçindekiler
- [Proje Mimarisi ve İçerik](#-proje-mimarisi-ve-i̇çerik)
- [Teorik Altyapı ve Metodoloji](#-teorik-altyapı-ve-metodoloji)
  - [1. Tek Nöron Forward Pass](#1-tek-nöron-forward-pass)
  - [2. Katman Mimari Yapısı (Layer Forward Pass)](#2-katman-mimari-yapısı-layer-forward-pass)
  - [3. Kayıp Fonksiyonu (Mean Squared Error - MSE)](#3-kayıp-fonksiyonu-mean-squared-error---mse)
  - [4. Parametre Analizi ve Kayıp Eğrisi (Optimization Landscape)](#4-parametre-analizi-ve-kayıp-eğrisi-optimization-landscape)
  - [5. Sayısal Türev ve Gradient Descent Döngüsü](#5-sayısal-türev-ve-gradient-descent-döngüsü)
- [Modelin Eğitimi ve Deney Sonuçları](#-model-eğitimi-ve-deney-sonuçları)
- [Kurulum ve Kullanım](#-kurulum-ve-kullanım)
- [Kaynaklar](#-kaynaklar)

---

## 🏗 Proje Mimarisi ve İçerik

Çalışma ortamı, teorik anlatımları, matematiksel notasyonları ve doğrulanmış Python kod hücrelerini içeren tek bir ana Jupyter Notebook dosyası üzerinde kurgulanmıştır:

* **`neural_network.ipynb`**: Projenin tüm teorik, pratik ve görselleştirme adımlarını içeren ana çalışma dosyası.

---

## 📐 Teorik Altyapı ve Metodoloji

### 1. Tek Nöron Forward Pass
Yapay bir nöron, sisteme beslenen girdileri öğrenilebilir ağırlıklar ($w$) ile ölçekleyip esneklik katan sapma ($b$) değerini ekleyen ve ardından doğrusal olmayan bir süzgeçten (aktivasyon) geçiren temel işlevdir.

$$\mathbf{y} = f\left( \sum_{i=1}^{n} (x_i \cdot w_i) + b \right)$$

* **Doğrusal Bileşke ($z$):** Girdi ve ağırlık vektörlerinin iç çarpımı artı sapma değeri.
* **Aktivasyon (ReLU):** Nörona doğrusal olmama yeteneği kazandıran $\text{ReLU}(z) = \max(0, z)$ işlevi.

---

### 2. Katman Mimari Yapısı (Layer Forward Pass)
Tekil nöronların karmaşık veri kalıplarını öğrenmedeki yetersizliğini aşmak için nöronlar paralel yapıda katmanlar halinde bir araya getirilir. Eş zamanlı çalışan nöronlar aynı girdiyi alır, ancak kendi bağımsız parametreleriyle farklı özellikleri temsil eder.

$$\mathbf{z} = \mathbf{W}\mathbf{x} + \mathbf{b}$$
$$\mathbf{a} = f(\mathbf{z}) = f(\mathbf{W}\mathbf{x} + \mathbf{b})$$

* **Ağırlık Matrisi ($\mathbf{W}$):** $m$ adet nöron ve $n$ adet girdi içeren $m \times n$ boyutlu matris.
* **Katman Çıktısı ($\mathbf{a}$):** Doğrusal toplam vektörünün eleman bazlı (*element-wise*) aktivasyonundan elde edilen $m \times 1$ boyutlu vektör.

---

### 3. Kayıp Fonksiyonu (Mean Squared Error - MSE)
Modelin ürettiği tahmin çıktısı ($\hat{y}$) ile gerçek hedef ($y$) arasındaki sapmayı nicelleştiren ve öğrenme sürecinde minimuma indirilmesi hedeflenen metriktir.

$$L = \text{MSE} = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)^2$$

Kare alma işlemi hataların pozitifliğini korurken, büyük sapmaları daha ağır cezalandırarak modelin kararlılığını artırır.

---

### 4. Parametre Analizi ve Kayıp Eğrisi (Optimization Landscape)
Bir sinir ağının öğrenmesi, aslında **"kayıp (loss) değerini düşüren ideal parametreleri arama"** sürecidir. Bu aşamada, diğer parametreler sabit tutulup yalnızca $w_3$ ağırlığı kademeli olarak değiştirilmiş ($w_3 \in [-2.0, 5.0]$) ve hatanın bu değişime nasıl tepki verdiği incelenmiştir. Grafikte iki farklı davranış bölgesi gözlemlenmiştir:

* **Eğimin Olduğu (Öğrenilebilir) Bölge ($w_3 < 0.4$):** 
  Girdi değerimiz negatif ($x_3 = -1.0$) olduğu için, ağırlık negatifleştikçe çarpım ($x_3 \cdot w_3$) pozitif bir değere dönüşür. Bu durum nöronun toplam yükünü artırarak **ReLU süzgecini geçmesini (nöronun ateşlenmesini)** sağlar. Nöron çıktı üretmeye başladıkça hedef değer olan $5.0$'a yaklaşır ve hata (loss) kademeli olarak düşer.

* **Ölü Nöron Bölgesi / Dead ReLU ($w_3 \ge 0.4$):** 
  Ağırlık pozitifleştiğinde, negatif girdi ile çarpımı negatif bir sonuç verir. Nöronun içindeki toplam değer sıfırın altında kaldığı için **ReLU fonksiyonu nöronu tamamen kapatır ($\hat{y} = 0$)**. Nöron sıfır ürettiği için hedef olan $5.0$ değerinden sürekli aynı miktarda sapar ve hata $L = (0 - 5.0)^2 = 25.0$ değerinde sabitlenir (düzleşir). Eğimin sıfır olduğu bu bölgede nöron pasiftir ve değişime tepki vermez.

---

### 5. Sayısal Türev ve Gradient Descent Döngüsü
Model parametrelerini elle tek tek denemek yerine, ağın **kendi kendine en doğru ağırlıkları bulmasını** isteriz. Bu otomatik öğrenme süreci iki aşamadan oluşur:

#### A. Yön Bulma: Sayısal Türev (Numerical Derivative)
Türev, basitçe **"bir parametreyi çok küçük bir miktarda ($h$) değiştirirsem, hata (loss) ne kadar değişir?"** sorusunun cevabıdır. Bilgisayar ortamında bu değişimi (eğimi) ölçmek için parametreye çok küçük bir itme payı ($h = 0.0001$) eklenir ve hatadaki değişim oranlanır:

$$\text{grad} = \frac{\partial L}{\partial w_i} \approx \frac{L(w_i + h) - L(w_i)}{h}$$

* **Eğim Pozitifse ($+ \text{grad}$):** Parametreyi artırmak hatayı büyütüyor demektir $\rightarrow$ Parametreyi **küçültmeliyiz**.
* **Eğim Negatifse ($- \text{grad}$):** Parametreyi artırmak hatayı küçültüyor demektir $\rightarrow$ Parametreyi **büyütmeliyiz**.

#### B. Adım Atma: Gradyan İnişi (Gradient Descent)
Hesaplanan eğim (gradyan), hatanın artış yönünü gösterir. Modelin amacı hatayı azaltmak olduğu için kayıp eğrisinde **"yokuş aşağı"** inmesi gerekir. Bu nedenle parametreler, eğimin tersi yönünde ve belirlenen bir adım boyutuyla ($\eta$ = learning rate) güncellenir:

$$w_{\text{yeni}} = w_{\text{eski}} - (\eta \cdot \text{grad})$$

Bu güncelleme döngüsü her adımda (epoch) tekrarlanarak modelin hatası adım adım sıfıra yaklaştırılır ve en ideal ağırlık parametreleri otomatik olarak tespit edilir.

---

## 📈 Modelin Eğitimi ve Deney Sonuçları

Adım 5'te kodlanan Gradient Descent döngüsü, modeli rastgele başlangıç ağırlıklarıyla çalıştırıp adım adım (epoch) eğitmiştir. Model, $x = [2.0, 3.0, -1.0]$ girdisi verildiğinde $y_{\text{true}} = 5.0$ hedefine ulaşmak için parametrelerini otomatik olarak güncellemiştir.

Eğitim sürecindeki hatanın düşüşü ve tahminin hedefe yakınşaması adım adım şu şekildedir:

| Adım (Epoch) | Hata / Kayıp (MSE Loss) | Modelin Tahmini ($\hat{y}$) | Durum ve Değerlendirme |
| :--- | :--- | :--- | :--- |
| **Başlangıç (0)** | **22.0900** | **0.3000** | Yüksek Hata (Rastgele Ağırlıklar) |
| **Epoch 5** | 1.2736 | 3.8714 | Hızlı İyileşme (Gradient İnişi Başladı) |
| **Epoch 10** | 0.0360 | 4.8102 | Hedefe Yaklaşma |
| **Epoch 15** | 0.0010 | 4.9680 | Minimum Hata |
| **Epoch 20** | **0.0000** | **4.9945** | Tam Optimizasyon (Tamamlandı) |

---

### 🎯 Eğitim Sonucu Analizi

* **Tahmin Başarısı:** Başlangıçta hedef değerden çok uzak olan model tahmini ($\hat{y} = 0.3000$), 20 adım sonunda **`4.9961`** değerine ulaşarak hedef olan **`5.0`** değerini %99.9 doğrulukla yakalamıştır.
* **Kayıp (Loss) Yakınşaması:** Başlangıçta **`22.0900`** olan MSE hatası, sayısal türevler ile doğru yönün bulunması sayesinde **`0.0000`** seviyesine kadar gerilemiştir.
* **Öğrenilen Optimum Parametreler:**
  * **Güncellenmiş Ağırlıklar ($W$):** $[1.1262, 0.3392, -0.2131]$
  * **Güncellenmiş Sapma ($b$):** $1.5131$

## 💻 Kurulum ve Kullanım

1. **Depoyu Klonlayın:**
```bash
   git clone [https://github.com/muhammedkoca17/yz50-neural-networks.git](https://github.com/muhammedkoca17/yz50-neural-networks.git)
   cd yz50-neural-networks
```

2. **Bağımlılıklar:**
*  Proje temel işlemlerde tamamen saf Python standart kütüphanelerini kullanır. Yalnızca Adım 4'teki kayıp eğrisi görselleştirmesi için `matplotlib` kütüphanesine ihtiyaç duyulur.

```bash
pip install matplotlib jupyter
```

3. **Çalıştırma**
```bash
jupyter notebook neural_network.ipynb
```

**Kaynaklar**

* 🧠 [The Mechanics of Neural Networks and Backpropagation](https://gemini.google.com/notebook/f5d9d368-e581-4235-9784-c58e17b0fc36) — *NotebookLM Çalışma Alanı*
* 🎥 **Andrej Karpathy:** *The spelled-out intro to neural networks and backpropagation: building micrograd*
* 🎥 **3Blue1Brown:** *Gradient descent, how neural networks learn | Deep Learning Chapter 2*

---

**Geliştirici:** Muhammed Koca  
**İletişim:** [LinkedIn](https://linkedin.com/in/mkoca) | [GitHub](https://github.com/muhammedkoca17)
