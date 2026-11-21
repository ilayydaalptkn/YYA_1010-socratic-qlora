# YYA_1010 — Socratic QLoRA Ön Çalışma

**Takım:** YYA_1010  
**Öğrenci:** İlayda ALPTEKİN  
**Danışman:** Dr. Öğr. Üyesi Yasin YALÇIN  
**Proje:** Küçük açık ağırlıklı bir LLM’in QLoRA ile Sokratik programlama öğreticisine özelleştirilmesi

## Bu depoda ne var?

| Yol | Açıklama |
|---|---|
| `data/train_seed.jsonl` | Al-Hossami BEA tohumundan üretilmiş 86 diyalog (chat format) |
| `data/scripts/txt_to_jsonl.py` | Ham `.txt` diyalogları JSONL’e çevirir |
| `data/scripts/train_qlora_smoke.py` | 20 adımlık mini QLoRA smoke eğitimi |
| `data/scripts/test_smoke_adapter.py` | Base Qwen vs smoke adaptör karşılaştırması |
| `outputs/smoke_adapter/` | Mini eğitimden çıkan LoRA adaptörü |

> Not: Tam Qwen2.5-7B ağırlıkları bu repoda yoktur (çok büyük). Yerelde Hugging Face cache’ten yüklenir.

## Ortam

- Windows 11, RTX 5060 Laptop 8 GB VRAM
- Python 3.13 venv (`qlora_env`): torch (CUDA), transformers, peft, bitsandbytes, trl, datasets

```powershell
# venv örneği (kendi yolunuza göre)
& "$env:USERPROFILE\OneDrive\Masaüstü\qlora_env\Scripts\Activate.ps1"
cd data\scripts
python train_qlora_smoke.py
```

## Ön bulgular (özet)

- 4-bit QLoRA smoke eğitim tamamlandı (20 step)
- Peak VRAM ≈ 5.9 GiB (8 GB kart içinde)
- Base model daha açıklayıcı; smoke adaptör daha soru soran (Sokratik) cevap verdi

## Yapay zekâ kullanımı

Kod iskeleti, hata ayıklama ve dokümantasyon düzenlemesinde Cursor (Composer) / üretken YZ araçlarından yardım alındı. Tasarım kararları, parametreler ve doğrulama öğrenciye aittir. Depodaki her dosya açıklanabilir durumdadır.
