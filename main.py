import wow_ocr

# Init pipeline, detector and recognizer models with pre trained weights
pipeline = wow_ocr.pipeline.Pipeline()


# Screenshots example
images = [
    wow_ocr.tools.read(url)
    for url in [
        "https://archive.thealphaproject.eu/media/Alpha-Project-Archive/UNSORTED/www_judgehype_com/Galerie%20d-images%20-%20Beta%20screenshots%20-%2020%20octobre%20-%20Mathieu%20Raid%20-%20World%20of%20Warcraft%20-%20JudgeHype%20-%2004-09-2021%2002-20-20/20%20october%2004%20-%2015.jpg",
        "https://archive.thealphaproject.eu/media/Alpha-Project-Archive/Images/Azeroth/Eastern%20Kingdoms/Stranglethorn%20Vale/WoWScrnShot_061904_102357.jpg",
        "https://archive.thealphaproject.eu/media/Alpha-Project-Archive/UNSORTED/www_ign_com/5440326.jpg",
    ]
]

# Results - Image to Text
prediction_groups = pipeline.recognize(images)
# # Each list of predictions in prediction_groups is a list of
# # (word, box) tuples.
