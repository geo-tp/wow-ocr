WoW Screenshot OCR
==============

`wow-ocr` is an OCR model to extract text from WoW screenshots.

#### It reads into :

- Chat
- Combat log
- Nameplates
- UI frames
- Map

Installation
-----

### ```pip install wow-ocr```

Usage
----

`wow-ocr` is packaged with trained weights. It's very easy to use : [Try it on Colab](https://colab.research.google.com/drive/1w4YIS--7qSzdSrwKPcQfqO988PlrxuCM?usp=sharing)

```
import wow_ocr

# 1 - Init pipeline, detector and recognizer models with trained weights
pipeline = wow_ocr.pipeline.Pipeline()


# 2 - Provide screenshots urls
images = [
    wow_ocr.tools.read(url)
    for url in [
        "https://image_url.com/1.jpg",
        "https://image_url.com/2.jpg",
    ]
]

# 3 - Get predictions
prediction_groups = pipeline.recognize(images)
# # Each list of predictions in prediction_groups is a list of
# # (word, box) tuples.

```

![](p1.webp)
![](p2.webp)


Training
-------

The recognizer model was fine tuned to be able to work with WoW Fonts. Here is the recognizer fine tuning process : [Fine Tuning Recognizer](https://github.com/geo-tp/Keras-Colaboratory-Models/blob/main/WoW_Screenshot_OCR_Training_Recognizer.ipynb)


Parsing
------

`wow-ocr` has been used to extract text from over 20,000 screenshots. You can see the parsing process here: [Parsing Big Dataset](https://github.com/geo-tp/Keras-Colaboratory-Models/blob/main/WoW_OCR_Parsing.ipynb)