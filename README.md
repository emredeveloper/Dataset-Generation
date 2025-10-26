# Dataset Generation & JSON Alma Araçları

Bu depo iki temel amaca hizmet eden küçük komut satırı araçları içerir:

1. **Sentetik müşteri verisi oluşturma** (`Fake_Customer_Dataset.py`).
2. **JSON verisini uzak bir URL'den veya yerel bir dosyadan alma** (`json_data_extraction.py` ve `json_data_extraction.js`).

Aşağıdaki bölümlerde her aracın kurulumu ve kullanımı anlatılmaktadır.

## 1. Sentetik müşteri verisi oluşturma

Python betiği, Faker kütüphanesi ile gerçekçi görünümlü müşteri verileri üretir.

### Kurulum

```bash
python -m venv .venv
source .venv/bin/activate  # Windows için .venv\\Scripts\\activate
pip install faker pandas numpy scikit-learn
```

### Kullanım

```bash
python Fake_Customer_Dataset.py --records 500 --seed 42 --report --output data/customers.csv
```

Anahtar seçenekler:

- `--records`: Üretilecek kayıt sayısı (varsayılan 1000).
- `--seed`: Rastgele sayı üreticilerini sabitleyerek deterministik sonuçlar sağlar.
- `--locale`: Faker'ın kullanacağı yerel ayar (varsayılan `en_US`).
- `--no-normalize`: Gelir ve harcama sütunlarında yapılan min-maks normalizasyonu devre dışı bırakır.
- `--report`: Basit bir istatistiksel özet ve kategorik değer dağılımı üretir.
- `--output`: Veriyi CSV formatında kaydetmek için dosya yolu.

Betiği modül olarak içe aktararak da kullanabilirsiniz:

```python
from Fake_Customer_Dataset import generate_customer_data

df = generate_customer_data(num_records=250, seed=1337)
```

## 2. JSON verisi alma

### Python sürümü (`json_data_extraction.py`)

Betiği çalıştırmak için `requests` kütüphanesine ihtiyacınız vardır:

```bash
pip install requests
```

Komutu çalıştırın:

```bash
python json_data_extraction.py https://raw.githubusercontent.com/emredeveloper/Database/main/db.json --pretty --output data/db.json
```

Parametreler:

- `source`: URL veya yerel dosya yolu (boş bırakılırsa varsayılan GitHub örneği kullanılır).
- `--timeout`: HTTP istekleri için zaman aşımı süresi (saniye cinsinden, varsayılan 10).
- `--pretty`: JSON çıktısını girintili biçimde yazdırır ve kaydeder.
- `--output`: Sonucu JSON dosyasına kaydeder.

Yerel dosyayı okumak için:

```bash
python json_data_extraction.py db.json --pretty
```

### Node.js sürümü (`json_data_extraction.js`)

Betiği Node 18+ sürümü ile doğrudan çalıştırabilirsiniz. Daha eski sürümler için `node-fetch` paketini yüklemek gerekir.

```bash
# Node 18+ ile
node json_data_extraction.js https://raw.githubusercontent.com/emredeveloper/Database/main/db.json

# Node 16 gibi eski sürümlerde
npm install node-fetch
node json_data_extraction.js
```

Komutu bir dosya yolu ile çağırarak yerel JSON dosyalarını da okuyabilirsiniz:

```bash
node json_data_extraction.js db.json
```

### Örnek Çıktı

Her iki sürüm de varsayılan olarak `db.json` içeriğini şu şekilde yazdırır:

```json
{
  "cities": [
    {
      "id": 1,
      "name": "Istanbul",
      "country": "Turkey",
      "temperature": 25,
      "weather": "Partly Cloudy",
      "humidity": 60
    }
  ]
}
```

## Projeye katkı

Katkı sağlamak için önce mevcut kodu gözden geçirip ihtiyaç duyduğunuz bağımlılıkları kurduğunuzdan emin olun. İyileştirme önerileriniz veya hata bildirimleriniz için pull request açabilirsiniz.
