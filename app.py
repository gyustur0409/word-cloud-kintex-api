from flask import Flask, jsonify, send_file, request
from konlpy.tag import Hannanum
import nltk
from nltk import flatten
from wordcloud import WordCloud, ImageColorGenerator
from collections import Counter
import numpy as np
from PIL import Image
from io import BytesIO
from werkzeug.utils import secure_filename

nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")

test_lyrics = """I walk a lonely road
The only one that I have ever known
Don't know where it goes
But it's home to me, and I walk alone
I walk this empty street
On the Boulevard of Broken Dreams
Where the city sleeps
And I'm the only one, and I walk alone
I walk alone, I walk alone
I walk alone, I walk a-
My shadow's the only one that walks beside me
My shallow heart's the only thing that's beating
Sometimes, I wish someone out there will find me
'Til then, I walk alone
Ah-ah, ah-ah, ah-ah, ah-ah
Ah-ah, ah-ah, ah-ah
I'm walking down the line
That divides me somewhere in my mind
On the borderline
Of the edge, and where I walk alone
Read between the lines
What's fucked up, and everything's alright
Check my vital signs
To know I'm still alive, and I walk alone
I walk alone, I walk alone
I walk alone, I walk a-
My shadow's the only one that walks beside me
My shallow heart's the only thing that's beating
Sometimes, I wish someone out there will find me
'Til then, I walk alone
Ah-ah, ah-ah, ah-ah, ah-ah
Ah-ah, ah-ah, I walk alone, I walk a-
I walk this empty street
On the Boulevard of Broken Dreams
Where the city sleeps
And I'm the only one, and I walk a-
My shadow's the only one that walks beside me
My shallow heart's the only thing that's beating
Sometimes, I wish someone out there will find me
'Til then, I walk alone
"""

app = Flask(__name__)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


def allowed_file(filename):
    print(filename)
    return "." in filename and filename.rsplit(".")[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["POST"])
def upload():
    try:
        print(request.files)
        imagefile = request.files.get("file")
        if imagefile is None or imagefile.filename == "":
            print("error: no file")
            return jsonify({"error": "no file"})

        if not allowed_file(imagefile.filename):
            print("error: format not supported")
            return jsonify({"error": "format not supported"})

        hannanum = Hannanum()

        word_list = flatten(hannanum.nouns(test_lyrics))

        if not word_list:
            tokenized = nltk.word_tokenize(test_lyrics)
            nouns = [
                word for (word, pos) in nltk.pos_tag(tokenized) if (pos[:2] == "NN")
            ]
            word_list = flatten(nouns)

        count = Counter(word_list)
        image = Image.open(BytesIO(imagefile.read())).convert("RGB")

        mask = np.array(image)

        wc = WordCloud(
            mask=mask,
            background_color="white",
        )
        wc = wc.generate_from_frequencies(count)

        image_colors = ImageColorGenerator(mask)

        result_image = wc.recolor(color_func=image_colors)

        pil_result = result_image.to_image()

        img_io = BytesIO()
        pil_result.save(img_io, format="PNG")

        img_io.seek(0)

        return send_file(img_io, mimetype="image/png")

    except Exception as err:
        print(err)
        return jsonify({"error": "error during make wordcloud"})
