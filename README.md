# emotional-TTS-system

## Getting Started
### Prerequisites

This has to be run on Windows 8 or above, and following commands are required to run the app.
```(venv) ~$ python -m pip install -r requirements.txt```
```(venv) ~$ flask run```

## Current Issues

- Right now it can only support "water harms the new born boy" as currently there isn't a way to extract text features for NZ language online. `FeatureExtractor#extract_text_features` is currently commented out for future fix. `Resynthesiser#predict` is also commented out.

- `SoundCreator#create_sound` is commented as [this](https://aotearoavoices.nz/api/requestAudio) rejects http request after several use.