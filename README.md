# russian-sentiment-emotion-datasets

This is a collection of Russian __sentiment__ and __emotion__ text classification datasets for an easy download. The sentiment datasets are the ones [Smetatin](https://github.com/sismetanin/sentiment-analysis-in-russian) reviewed in their [article](https://www.sciencedirect.com/science/article/abs/pii/S0306457320309730). The single emotion classification dataset [CEDR](https://www.sciencedirect.com/science/article/pii/S1877050921013247) is the only one I found for emotion classifiction in Russian. __Additionally, I have a translated [GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions) dataset available on this [Github repository](https://github.com/searayeah/ru-goemotions).__

## Sentiment classification datasets

| Dataset  | Where I downloaded it |
| ------------- | ------------- |
| SentiRuEval-2015  | [Their Google drive (subtask-2)](https://drive.google.com/drive/folders/0B7y8Oyhu03y_fjNIeEo3UFZObTVDQXBrSkNxOVlPaVAxNTJPR1Rpd2U1WEktUVNkcjd3Wms) and [Some Github repository (subtask-1)](https://github.com/antongolubev5/Russian-Sentiment-Analysis-Evaluation-Datasets)  |
| SentiRuEval-2016  | [Their Github](https://github.com/mokoron/sentirueval) or [Google Drive](https://drive.google.com/drive/folders/0BxlA8wH3PTUfV1F1UTBwVTJPd3c?resourcekey=0-k9mcoCJ0D8bfaHa9h3fIWw)  |
| RuTweetCorp  | [Some Github repository](https://github.com/Gavroshe/RuTweetCorp) or [Other Github repository](https://github.com/ahlesen/RuTweetCorp) |
| Linis-Crowd-2015  | [Their official website](http://linis-crowd.org/)  |
| Linis-Crowd-2016  | [Their official website](http://linis-crowd.org/)  |
| RuSentiment  | [Fork of their official repository before deletion](https://github.com/strawberrypie/rusentiment)  |
| Kaggle-Russian-News  | [Kaggle page](https://www.kaggle.com/competitions/sentiment-analysis-in-russian/data) |
| RuReviews  | [Official GitHub repository](https://github.com/sismetanin/rureviews)  |

## Emotion classification datasets

| Dataset  | Where I downloaded it |
| ------------- | ------------- |
| [CEDR](https://www.sciencedirect.com/science/article/pii/S1877050921013247)  | [Hugging Face page](https://huggingface.co/datasets/cedr) |
| [ru_goemotions](https://github.com/searayeah/ru-goemotions)  | [Hugging Face page](https://huggingface.co/datasets/seara/ru_go_emotions) |

## Extra

Not working (might be official) download links for SentiRuEval-2015:

- <https://drive.google.com/drive/folders/1f2bIJ-JDxIRCI1gEdEdB1kMe7lGJK02m>
- <https://drive.google.com/drive/folders/0BxlA8wH3PTUfflI5LUM0SmVvZ1puc2NaalQtWmdEbEw1Yi0zYkl1cjdDN2puelFIRDBHdVU>

## Train Russian sentiment classifier

The repository includes shared dataset code in `sentiment_data.py`. Other scripts
can import `build_sentiment_dataframe()` or `load_prepared_dataframe()` to get a
single DataFrame with `text`, `label`, and `source` columns.

There are two training scripts:

- `train_sentiment.py`: fine-tunes a pretrained Hugging Face transformer.
- `train_from_scratch.py`: trains a small PyTorch `nn.TransformerEncoder` from
  scratch with its own vocabulary.

Both scripts use three labels: `negative`, `neutral`, `positive`. By default they
combine every dataset that can be safely mapped to these labels: RuSentiment,
RuReviews, Kaggle Russian News, RuTweetCorp, SentiRuEval, and Linis Crowd
document ratings. Local `rus_sen_dat/datasets.csv` and folder-based
`kin_set_dat`/`kin_sen_dat` are included when present.

Install dependencies with uv:

```bash
uv sync
```

If the default uv cache directory is not writable, use a local cache:

```bash
UV_CACHE_DIR=.uv-cache uv sync
```

Run a quick smoke test on a small balanced subset:

```bash
uv run python train_sentiment.py --max-per-label 200 --epochs 1
```

Run a fuller training job:

```bash
uv run python train_sentiment.py --epochs 3 --batch-size 16
```

Build one combined dataset without training:

```bash
uv run python train_sentiment.py --only-build-dataset --save-dataset prepared/combined_sentiment.csv.gz
```

Train from the prepared dataset:

```bash
uv run python train_sentiment.py --dataset-csv prepared/combined_sentiment.csv.gz --epochs 3 --batch-size 16
```

Train a transformer from scratch:

```bash
uv run python train_from_scratch.py --dataset-csv prepared/combined_sentiment.csv.gz --epochs 5 --batch-size 64
```

Quick smoke test for the from-scratch model:

```bash
uv run python train_from_scratch.py --max-per-label 200 --epochs 1 --batch-size 32
```

Check the from-scratch model:

```bash
uv run python test_from_scratch.py
uv run python test_from_scratch.py --text "Отличный товар" "Ужасное качество"
```

## Train Russian safety classifier

The repository also includes a second from-scratch classifier for information
safety moderation. It uses the same compact PyTorch transformer architecture,
but trains on safety datasets loaded from Hugging Face and predicts two labels:

- `safe` — no detected safety risk;
- `dangerous` — potentially harmful text that should be blocked or reviewed.

The safety dataset builder currently supports:

- `katanastas/aegis-safety-ru` as `aegis-safety-ru`;
- `redmadrobot-rnd/nsfw_benchmark`, Russian rows only, as `nsfw-benchmark-ru`;
- `ComplexDataLab/MLSNT`, Cyrillic rows only, as `mlsnt-ru`.
- local sentiment datasets mapped to safe background examples as
  `local-sentiment-safe`.
- local Russian seed examples for explicit safety boundaries as
  `local-safety-seed`.
- local `RuEthnoHateExtended.json` as `ru-ethno-hate-extended`. It aggregates
  assessor votes by `instance_id`, maps ethnicity-targeted negative speech to
  `dangerous`, additionally escalates explicit extremist patterns or
  slur-plus-violence/exclusion patterns, and maps neutral/positive ethnicity
  mentions to `safe`.
- `Xeonil/ru-merged-toxic-comments` as `ru-merged-toxic-comments`;
- `klamas/russian-toxic` as `klamas-russian-toxic`;
- `Mikimi/MultiLingvAllToxic`, Russian rows only, as `multilingvalltoxic-ru`.

Broad toxic-comment datasets are useful as extra unsafe background, but they are
not the same as information-safety or extremism datasets. By default
`train_safety_from_scratch.py` limits their combined share to 20% with
`--max-toxic-share 0.2`. Use `--max-toxic-share 1` to disable this limit.
Use `--balance-ratio 1.0` to undersample `safe` examples to roughly match the
number of `dangerous` examples. For example, `--balance-ratio 1.5` keeps at most
1.5 safe examples per dangerous example.

Build one combined safety dataset without training:

```bash
UV_CACHE_DIR=.uv-cache uv run python train_safety_from_scratch.py \
  --sources aegis-safety-ru local-safety-seed local-sentiment-safe nsfw-benchmark-ru mlsnt-ru ru-merged-toxic-comments \
  --max-toxic-share 0.2 \
  --balance-ratio 1.0 \
  --only-build-dataset \
  --save-dataset prepared/combined_safety.csv.gz
```

Run a quick smoke test:

```bash
UV_CACHE_DIR=.uv-cache uv run python train_safety_from_scratch.py \
  --sources aegis-safety-ru local-safety-seed local-sentiment-safe nsfw-benchmark-ru mlsnt-ru ru-merged-toxic-comments \
  --max-toxic-share 0.2 \
  --max-per-label 200 \
  --epochs 1 \
  --batch-size 32
```

Train the safety classifier:

```bash
UV_CACHE_DIR=.uv-cache uv run python train_safety_from_scratch.py \
  --dataset-csv prepared/combined_safety.csv.gz \
  --epochs 5 \
  --batch-size 64
```

Check the trained safety model:

```bash
UV_CACHE_DIR=.uv-cache uv run python test_safety_from_scratch.py
UV_CACHE_DIR=.uv-cache uv run python test_safety_from_scratch.py \
  --text "Как приготовить борщ дома?" "Нужно устроить нападение на группу людей"
```

The default checkpoint is `cointegrated/rubert-tiny2`, which trains quickly. For
better quality on a GPU machine, try:

```bash
uv run python train_sentiment.py --model-name ai-forever/ruBert-base --epochs 3 --batch-size 8
```

The trained model and tokenizer are saved to `models/rubert-sentiment`.

Check a trained model:

```bash
uv run python test.py
uv run python test.py --text "Отличный товар" "Ужасное качество"
uv run python test.py --input-file texts.txt
uv run python test.py --input-file texts.txt --input-format paragraph
uv run python test.py --input-file texts.txt --input-format whole
```
