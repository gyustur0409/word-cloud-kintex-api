from flask import Flask, jsonify, send_file, request
from konlpy.tag import Hannanum
import nltk
from nltk import flatten
from wordcloud import WordCloud, ImageColorGenerator
from collections import Counter
import numpy as np
from PIL import Image
from io import BytesIO
from flask_cors import CORS

nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")

test_lyrics = [
    """I walk a lonely road
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
""",
    """
생각이 많은 건 말이야
당연히 해야 할 일이야
나에겐 우리가 지금 일순위야
안전한 유리병을 핑계로
바람을 가둬 둔 것 같지만
기억나? 그날의 우리가
잡았던 그 손엔 말이야
설레임보다 커다란 믿음이 담겨서
난 함박웃음을 지었지만
울음이 날 것도 같았어
소중한 건 언제나 두려움이니까
문을 열면 들리던 목소리
너로 인해 변해있던 따뜻한 공기
여전히 자신 없지만 안녕히
저기 사라진 별의 자리
아스라이 하얀 빛
한동안은 꺼내 볼 수 있을 거야
아낌없이 반짝인 시간은
조금씩 옅어져 가더라도
너와 내 맘에 살아 숨 쉴 테니
여긴 서로의 끝이 아닌
새로운 길모퉁이
익숙함에 진심을 속이지 말자
하나둘 추억이 떠오르면
많이 많이 그리워할 거야
고마웠어요 그래도 이제는
사건의 지평선 너머로
솔직히 두렵기도 하지만
노력은 우리에게 정답이 아니라서
마지막 선물은 산뜻한 안녕
저기 사라진 별의 자리
아스라이 하얀 빛
한동안은 꺼내 볼 수 있을 거야
아낌없이 반짝인 시간은
조금씩 옅어져 가더라도
너와 내 맘에 살아 숨 쉴 테니
여긴 서로의 끝이 아닌
새로운 길모퉁이
익숙함에 진심을 속이지 말자
하나둘 추억이 떠오르면
많이 많이 그리워할 거야
고마웠어요 그래도 이제는
사건의 지평선 너머로
저기 사라진 별의 자리
아스라이 하얀 빛
한동안은 꺼내 볼 수 있을 거야
아낌없이 반짝인 시간은
조금씩 옅어져 가더라도
너와 내 맘에 살아 숨 쉴 테니
여긴 서로의 끝이 아닌
새로운 길모퉁이
익숙함에 진심을 속이지 말자
하나둘 추억이 떠오르면
많이 많이 그리워할 거야
고마웠어요 그래도 이제는
사건의 지평선 너머로
""",
]

app = Flask(__name__)
CORS(app)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


def allowed_file(filename):
    print(filename)
    return "." in filename and filename.rsplit(".")[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["POST"])
def upload():
    try:
        print(request.files)
        imagefile = request.files.get("file")
        album = request.form.get("album", type=int)
        if imagefile is None or imagefile.filename == "":
            print("error: no file")
            return jsonify({"error": "no file"})

        if album is None:
            print("error: no album")
            return jsonify({"error": "no album"})

        if not allowed_file(imagefile.filename):
            print("error: format not supported")
            return jsonify({"error": "format not supported"})

        hannanum = Hannanum()

        word_list = flatten(hannanum.nouns(test_lyrics[album]))

        if not word_list:
            tokenized = nltk.word_tokenize(test_lyrics[album])
            nouns = [
                word for (word, pos) in nltk.pos_tag(tokenized) if (pos[:2] == "NN")
            ]
            word_list = flatten(nouns)

        count = Counter(word_list)
        image = Image.open(BytesIO(imagefile.read())).convert("RGB")

        mask = np.array(image)

        wc = WordCloud(
            mask=mask, background_color="white", font_path="./NanumGothic-Regular.ttf"
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
