# Iris Studio

Senin eğittiğin PyTorch modelini kullanan FastAPI web uygulaması. Arayüz HTML, CSS ve JavaScript ile yazıldı. Node.js, npm, API anahtarı veya GPU gerektirmez.

## Başlat — Windows / VS Code

1. Bilgisayarında Python 3.12 kurulu olsun. Python kurulumunda **Add Python to PATH** seçeneğini işaretle.
2. ZIP'i çıkart. VS Code'da **File → Open Folder** ile `iris-studio` klasörünü aç. `README.md`, `app`, `frontend` ve `models` aynı klasörde görünmeli.
3. **Terminal → New Terminal** ile PowerShell terminali aç. Şunları sırayla çalıştır:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run.py
```

4. Tarayıcıda **http://127.0.0.1:8000** adresini aç.

Sunucuyu durdurmak için terminalde `Ctrl+C` kullan. Sonraki açılışlarda yalnızca son komut yeterlidir. Sanal ortamı etkinleştirmek gerekmez; komutlar doğrudan doğru Python'u kullanır.

`py` komutu tanınmıyorsa Python kurulumunu kontrol et. `python --version` komutun 3.12 gösteriyorsa ilk komutta `py -3.12` yerine `python` yazabilirsin. VS Code bir yorumlayıcı sorarsa **Python: Select Interpreter** üzerinden `.venv` içindeki Python'u seç.

F5 ile çalıştırmak için Microsoft Python ve Python Debugger uzantılarını kur; `.venv` yorumlayıcısını seç ve **Iris Studio · FastAPI** yapılandırmasını başlat.

> Arayüzü VS Code Live Server veya `index.html` dosyasına çift tıklayarak açma. HTML ve API, FastAPI üzerinden aynı adresten sunuluyor.

## Başlat — macOS / Linux

Python 3.12 ile proje klasöründe:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python run.py
```

## Arayüzde neler var?

- Açık renkli, lavanta ve adaçayı tonlarında botanik laboratuvar tasarımı.
- Dört ölçüm için kaydırıcı ve doğrudan sayı girişi; klavye ile kullanım.
- Ölçümlerle biçimi değişen özgün SVG Iris çizimi. Çizim temsili; birebir fiziksel ölçek veya tür teşhis görseli değildir.
- **Canlı tahmin**: ölçüm değişikliğinden 350 ms sonra gerçek FastAPI isteği. Kapatıldığında **Türü keşfet** düğmesiyle analiz.
- Setosa, Versicolor ve Virginica için tek tıkla hazır ölçümler.
- Üç sınıfın softmax dağılımı ve model hesaplama süresi.
- Veri aralığı aşıldığında uyarı; hatalı/eksik girdide açıklayıcı mesaj.
- Son sekiz başarılı tahmin için tarayıcı geçmişi; kayıtları yeniden kullanma ve temizleme.
- Sonucun JSON olarak indirilmesi.
- Model mimarisi penceresi, mobil ekran stilleri ve azaltılmış hareket desteği.

Arayüz harici font, CDN veya görsel hizmeti kullanmaz. İlk bağımlılık kurulumu internet gerektirir; kurulum tamamlandıktan sonra yerel tahminler internet bağlantısı olmadan çalışır. Otomatik Swagger `/docs` sayfası ise FastAPI'nin varsayılan CDN varlıklarını kullanır.

## Modelinle uyumu

Yüklediğin `iris_model(1).pth` dosyası, pakete **değiştirilmeden** `models/iris_model.pth` adıyla konuldu. Yeniden eğitim yapılmadı. `state_dict` tek başına mimariyi tanımlamadığı için notebook'un son kullanılan sınıfı `app/model.py` içinde yeniden oluşturuldu:

```python
self.linear_layer_stack = nn.Sequential(
    nn.Linear(4, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 3),
)
```

Notebook'ta iki `IrisClassifier` tanımı var. İlk, `layer_1` ile başlayan tanımın yerini ikinci `linear_layer_stack` tanımı alıyor. Gönderdiğin ağırlıkların anahtarları da ikinci tanımla eşleşiyor. Kaydedilen model, notebook'taki `model2` nesnesi.

| Girdi sırası | JSON alanı | Notebook sütunu | Birim |
| --- | --- | --- | --- |
| 1 | `sepal_length` | `SepalLengthCm` | cm |
| 2 | `sepal_width` | `SepalWidthCm` | cm |
| 3 | `petal_length` | `PetalLengthCm` | cm |
| 4 | `petal_width` | `PetalWidthCm` | cm |

**`Id` modele verilmez. StandardScaler veya başka bir ölçeklendirme uygulanmaz.** Notebook'ta ham cm değerleriyle eğitildiğinden çıkarımda da aynı dönüşüm korunur. Girdiler `float32` tensörüne çevrilir. Model 4.675 parametrelidir.

| Çıkış indeksi | Etiket |
| --- | --- |
| 0 | `Iris-setosa` |
| 1 | `Iris-versicolor` |
| 2 | `Iris-virginica` |

Bu sıra notebook'taki `LabelEncoder` sırasıdır. Yeni bir model eğitirsen mimari, özellik sırası, ön işleme ve etiket sırası aynı kalmalıdır; değişirse uygulamayı da güncelle.

Model `lifespan` sırasında bir kez, CPU üzerinde `torch.load(..., map_location="cpu", weights_only=True)` ile yüklenir. `strict=True` yanlış katman/şekil eşleşmesini yakalar. Çıkarımda `model.eval()` ve `torch.inference_mode()` kullanılır. `.pth` dosyası statik klasöre konulmadığı için web arayüzünden indirmeye açılmaz.

Ölçüm aralığı uyarıları, notebook'taki `df.describe()` çıktısına dayanır; yalnızca eğitim alt kümesinin min/max değerleri olduğu iddia edilmez. Ham CSV bu teslimatta bulunmadığı için uygulama test doğruluğu hesaplamaz ve arayüzde bir doğruluk iddiası göstermez. **Ekrandaki yüzde yalnızca softmax çıktısıdır; başarı oranı ya da kalibre edilmiş kesinlik değildir.**

## Dosyalar

| Dosya | Görevi |
| --- | --- |
| `app/main.py` | FastAPI, modelin yüklenmesi, tahmin ve statik dosya yolları |
| `app/model.py` | Notebook ile aynı PyTorch mimarisi |
| `app/schemas.py` | Girdi doğrulama, sınıf/özellik sırası ve API yanıt sözleşmesi |
| `frontend/index.html` | Sayfa, SVG çizim ve model penceresi |
| `frontend/styles.css` | Masaüstü/mobil tasarım ve animasyonlar |
| `frontend/app.js` | Kaydırıcılar, API istekleri, geçmiş ve JSON indirme |
| `models/iris_model.pth` | Senin orijinal model ağırlıkların |
| `run.py` | Yerel geliştirme sunucusu |
| `.vscode/launch.json` | F5 hata ayıklama yapılandırması |
| `tests/test_api.py` | Gerçek ağırlıklarla API testleri |

Başlık/metinler için `index.html`; renkler için `styles.css` başındaki `:root`; etkileşim için `app.js` dosyasını düzenle.

## API

| Yöntem | Yol | Amaç |
| --- | --- | --- |
| `GET` | `/` | Arayüz |
| `GET` | `/api/health` | Model hazır mı? |
| `GET` | `/api/model` | Mimari, özellikler, sınıflar ve ağırlık sürümü |
| `POST` | `/api/predict` | Gerçek modelle tahmin |
| `GET` | `/docs` | Swagger arayüzü |

Örnek JSON isteği:

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

Yanıt: `species`, `class_index`, `probability`, üç sınıfı içeren `probabilities`, gönderilen `measurements`, `warnings`, `inference_ms`, `model_version`, `request_id` ve UTC `predicted_at`. Olasılıklar API'de 0–1, arayüzde 0–100 aralığındadır. `inference_ms` yalnızca sunucudaki model hesaplamasını ölçer; ağ gecikmesi dahil değildir.

PowerShell ile deneme:

```powershell
$sample = @{
    sepal_length = 5.1
    sepal_width = 3.5
    petal_length = 1.4
    petal_width = 0.2
} | ConvertTo-Json
Invoke-RestMethod -Uri http://127.0.0.1:8000/api/predict -Method Post -ContentType "application/json" -Body $sample
```

Eksik/fazla alan, sayı yerine metin, boolean ve aralık dışı girdiler `422` ile reddedilir. Kabul edilen uygulama aralıkları: çanak uzunluğu 0,1–10; çanak genişliği 0,1–6; taç uzunluğu 0,1–10; taç genişliği 0,1–4 cm. Sayı girişinde arayüz 0,1 cm adım kullanır. API ayrıca bu aralıklardaki daha hassas sayıları kabul eder.

Arayüz bir girdi değiştiğinde önceki tahmini soldurur ve eski sonucun indirilmesini kapatır. Yeni istek eski isteği iptal eder; geç gelen yanıtlar güncel tahminin üzerine yazamaz. Sunucu çalışmıyorsa uygulama tahmin uydurmaz; bağlantı mesajı gösterir.

## Kontroller

Windows:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
```

macOS / Linux:

```bash
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q
```

Teslim öncesi 18 test geçti. Üç tür için beklenen sonuçlar, modelin doğrudan çalıştırılmasıyla API olasılıklarının eşitliği, JSON anahtar sırasından bağımsızlık, geçersiz girdiler, veri aralığı uyarıları, statik dosyalar, eksik ve uyumsuz ağırlıklar kontrol edildi. JavaScript sözdizimi kontrol edildi. Test ortamı: Python 3.12, PyTorch 2.15.0+cpu, FastAPI 0.162.0, Pydantic 2.14.1, Uvicorn 0.54.0. `requirements.txt` bu sürümlerle doğrulanan API'ler için uyumlu sürüm aralıkları içerir; tam kilit dosyası değildir.

Masaüstü ve 390 px genişlikte mobil yerleşim, hatalı ölçüm mesajı ve model penceresi statik tarayıcı önizlemesinde kontrol edildi. Önizleme hizmetinde Python bulunmadığından tarayıcıdan gerçek FastAPI tahminine kadar uçtan uca doğrulama yapılamadı; gerçek tahminler FastAPI TestClient ile kontrol edildi. JSON indirme, geçmiş ve canlı tahminin tarayıcıdan gerçek API ile birlikte kullanımı yerel çalıştırmada ayrıca denenmelidir.

## Sık karşılaşılan durumlar

- **Model dosyası bulunamadı:** `models/iris_model.pth` yerinde mi? ZIP içindeki bu dosyayı taşıma veya silme.
- **Address already in use / WinError 10048:** 8000 portundaki önceki sunucuyu durdur ya da `python -m uvicorn app.main:app --port 8001` ile başlat; arayüzü de 8001 adresinden aç.
- **ModuleNotFoundError:** `pip` ve çalıştırma komutlarında aynı `.venv` Python'unu kullan.
- **Pip bağlantı hatası:** Kurulum internet ve paket sunucularına erişim gerektirir. Kurumsal bilgisayarda kurumunun onaylı paket/proxy ayarlarını kullan.

Bu paket yerel geliştirme içindir. İnternete yayınlama ayrıca bir dağıtım adımıdır; `run.py` yalnızca `127.0.0.1` adresinde dinler.

## Başvurulan resmî dokümantasyon

- [PyTorch — Saving and Loading Models](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)
- [FastAPI — Lifespan Events](https://fastapi.tiangolo.com/advanced/events/)
- [FastAPI — Static Files](https://fastapi.tiangolo.com/tutorial/static-files/)
