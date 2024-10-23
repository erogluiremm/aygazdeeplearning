
# Balık Türlerini Sınıflandırma Projesi

Bu proje, bir veri seti kullanarak balık türlerinin sınıflandırılmasını amaçlamaktadır. Veri setindeki her balık türü, farklı klasörlerde bulunan görsellerden oluşmaktadır. Model, bu görselleri kullanarak her balık türünü sınıflandırmayı öğrenir.

## İçindekiler
- [Proje Hakkında](#proje-hakkında)
- [Kullanılan Kütüphaneler](#kullanılan-kütüphaneler)
- [Veri Seti](#veri-seti)
- [Modelin Eğitimi](#modelin-eğitimi)
- [Sonuçlar](#sonuçlar)
- [Nasıl Çalıştırılır](#nasıl-çalıştırılır)

## Proje Hakkında

Bu projede, Kaggle üzerinde bulunan `A Large-Scale Fish Dataset` kullanılarak balık türleri sınıflandırması yapılmıştır. Görseller öncelikle bir sabit boyuta (`image_size = (128, 128)`) getirilir, daha sonra bir derin öğrenme modeli eğitilerek balık türleri tahmin edilmeye çalışılır.

### Amaç
Balık türlerinin görsel veriler yardımıyla sınıflandırılması ve doğru tür tahmini yapılması hedeflenmiştir.

## Kullanılan Kütüphaneler

Projede kullanılan temel kütüphaneler aşağıdaki gibidir:

- `numpy`: Sayısal veriler ve diziler üzerinde işlemler yapmak için.
- `pandas`: Veri işleme ve analiz için.
- `PIL.Image`: Görselleri açmak ve işlemek için.
- `tensorflow.keras`: Derin öğrenme modelini oluşturmak ve eğitmek için.
- `sklearn.preprocessing`: Etiketleri sayısal hale getirmek için (LabelEncoder).
- `sklearn.model_selection`: Veriyi eğitim ve test olarak ayırmak için (train_test_split).

## Veri Seti

Bu proje için kullanılan veri seti Kaggle üzerinde bulunmaktadır. Her balık türü için farklı klasörlerde çok sayıda görsel vardır. Aşağıdaki adımları izleyerek veri seti Kaggle'da notebook'a eklenmiştir:

- Kaggle üzerinde notebook çalıştırılırken, sağ taraftaki `Add Data` seçeneğinden veri seti eklenmiştir.
- `base_path` olarak veri setinin dizini şu şekilde ayarlanmıştır:
  ```python
  base_path = "../input/a-large-scale-fish-dataset/Fish_Dataset"
  ```

Görseller bu dizin üzerinden okunup işlenmiştir.

## Modelin Eğitimi

Model eğitimi için izlenen adımlar:

### 1. Verilerin Hazırlanması

- Görseller önce numpy dizisine çevrildi ve `128x128` boyutuna yeniden boyutlandırıldı.
- Görsellerin piksel değerleri `0-255` aralığından `0-1` aralığına normalleştirildi.

```python
# Görüntü boyutu
image_size = (128, 128)

# Görsellerin ve etiketlerin saklanacağı listeler
data = []
labels = []

# Görsellerin işlenmesi
for folder in os.listdir(base_path):
    folder_path = os.path.join(base_path, folder)
    for file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, file)
        image = Image.open(img_path).resize(image_size)
        data.append(np.array(image))
        labels.append(folder)

# Verileri numpy dizisine çevir
data = np.array(data) / 255.0  # Normalleştirme
```

### 2. Etiketlerin Sayısal Hale Getirilmesi

`LabelEncoder` kullanılarak kategorik etiketler sayısal değerlere dönüştürülmüştür. Daha sonra bu sayısal değerler, `to_categorical` kullanılarak one-hot encode edilmiştir.

```python
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

le = LabelEncoder()
labels = le.fit_transform(labels)  # Etiketlerin sayısal hale getirilmesi
labels = to_categorical(labels)  # One-hot encode işlemi
```

### 3. Eğitim ve Test Verilerinin Ayrılması

Veriler eğitim ve test verileri olarak `train_test_split` ile %80 eğitim, %20 test olacak şekilde ayrılmıştır.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
```

### 4. Modelin Yapısı

Model, Conv2D (konvolüsyonel katmanlar) ve MaxPooling2D katmanları kullanılarak oluşturulmuştur. Son katman olarak `softmax` kullanılarak, her bir sınıfa ait olma olasılığı hesaplanmıştır.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Modelin oluşturulması
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(5, activation='softmax'))  # 5 sınıf için softmax katmanı

# Modelin derlenmesi
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Model özeti
model.summary()
```

### 5. Modelin Eğitilmesi

Model 10 epoch boyunca eğitilmiştir:

```python
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
```

## Sonuçlar

Model, eğitim ve doğrulama seti üzerinde çalıştırıldıktan sonra doğruluk (accuracy) ve kayıp (loss) değerleri ölçülmüştür. Eğitim süreci boyunca modelin performansı `history` değişkeninde saklanmıştır.

## Nasıl Çalıştırılır

1. Kaggle'da yeni bir notebook açın.
2. Projeye `A Large-Scale Fish Dataset` veri setini ekleyin.
3. Yukarıda verilen adımları takip ederek kodu çalıştırın.
4. Modeli eğitip test sonuçlarını gözlemleyin.

## Kaynaklar

- [Kaggle - A Large-Scale Fish Dataset](https://www.kaggle.com/crowww/a-large-scale-fish-dataset)
