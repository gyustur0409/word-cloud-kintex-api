from flask import Flask, jsonify, send_file, request
from konlpy.tag import Mecab
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
    """영원을 말했죠
꿈이 아니기를
혼돈 속을 지나면
반짝일 거라고
소나기가 내려오면
이건 잠시뿐일 거야
눈이 부신 그날의 기억은
기적이니까요
좀 더 멀리멀리
닿을 수 있을까
길고 긴 여정에
마침표를 찍을 수만 있다면
마지막 땀방울의 결말은
헛된 길이 아닐 걸 잘 알아
반복되는 계절의 중간에 있어
그토록 바랬던 어둠 속의 빛을 찾고 말았어
너에게로 달려가는 이 시공간을 넘어서
닿은 이곳은 여섯 번째
여름의 시작이었단 걸
꿈꿨어
푸른 하늘, 마치 눈물 같아
날이 밝아도 결국 쏟아진다
하염없이 달려온 길 위 oh
속도를 올렸어
망설이다 전부 다 놓쳐 버릴까 또
일곱 번의 여름은 없을 거라고
끝이 없던 평행곡선
결국 같은 곳을 지나왔어
좀 더 멀리멀리
닿을 수 있을까
길고 긴 여정에
마침표를 찍을 수만 있다면
마지막 땀방울의 결말은
헛된 길이 아닐 걸 잘 알아
반복되는 계절의 중간에 있어
그토록 바랬던 어둠 속의 빛을 찾고 말았어
너에게로 달려가는 이 시공간을 넘어서
닿은 이곳은 여섯 번째
여름의 시작이었단 걸
꿈꿨어
꿈일까 꿈일까
우리 함께하는 이 순간
꿈같아
마법 같은 하루가
결국엔 사라질까
그칠까 그칠까 찬란히
반짝이던 눈물의 기적
빛나줘
반복되는 계절의 중간에 있어
그토록 바랬던 어둠 속의 빛을 찾고 말았어
너에게로 달려가는 이 시공간을 넘어서
닿은 이곳은 여섯 번째
여름의 시작이었단 걸
꿈꿨어
""",
    """
I'm the son of rage and love
The Jesus of Suburbia
The bible of none of the above
On a steady diet of
Soda pop and Ritalin
No one ever died for my sins in hell
As far as I can tell
'Least the ones I got away with
And there's nothing wrong with me
This is how I'm supposed to be
In a land of make believe
That don't believe in me
Get my television fix
Sitting on my crucifix
The living room, or my private womb
While the moms and Brads are away
To fall in love and fall in debt
To alcohol and cigarettes and Mary Jane
To keep me insane
Doing someone else's cocaine
And there's nothing wrong with me
This is how I'm supposed to be
In a land of make believe
That don't believe in me
At the center of the Earth, in the parking lot
Of the 7-11 where I was taught
The motto was just a lie
It says, "Home is where your heart is, " but what a shame
'Cause everyone's heart doesn't beat the same
It's beating out of time
City of the dead (hey! Hey!)
At the end of another lost highway (hey! Hey!)
Signs misleading to nowhere
City of the damned (hey! Hey!)
Lost children with dirty faces today (hey! Hey!)
No one really seems to care
I read the graffiti in the bathroom stall
Like the holy scriptures of a shopping mall
And so it seemed to confess
It didn't say much, but it only confirmed that
The center of the earth is the end of the world
And I could really care less
City of the dead (hey! Hey!)
At the end of another lost highway (hey! Hey!)
Signs misleading to nowhere
City of the damned (hey! Hey!)
Lost children with dirty faces today (hey! Hey!)
No one really seems to care...
Hey!
I don't care if you don't
I don't care if you don't
I don't care if you don't care
I don't care if you don't
I don't care if you don't
I don't care if you don't care
I don't care if you don't
I don't care if you don't
I don't care if you don't care
I don't care if you don't
I don't care if you don't
I don't care if you don't care
I don't care!
Everyone's so full of shit
Born and raised by hypocrites
Hearts recycled, but never saved
From the cradles to the grave
We are the kids of war and peace
From Anaheim to the Middle East
We are the stories and disciples of
The Jesus of Suburbia
Land of make believe
And it don't believe in me
Land of make believe
And it don't believe
And I don't care! (Whoo! Whoo! Whoo!)
I don't care! (Whoo! Whoo! Whoo!)
I don't care! (Whoo! Whoo! Whoo!)
I don't care! (Whoo! Whoo! Whoo!)
I don't care!
Dearly beloved, are you listening?
I can't remember a word that you were saying...
Are we demented or am I disturbed?
The space that's in between insane and insecure
Ooh
Ooh, ooh, ooh
Ooh
Ooh, ooh, ooh
Oh therapy, can you please fill the void? (Ooh, ooh)
Am I retarded, or am I just overjoyed? (Ooh, ooh, ooh)
Nobody's perfect, and I stand accused (ooh, ooh)
For lack of a better word, and that's my best excuse (ooh, ooh, ooh)
Ooh
Ooh, ooh, ooh
Ooh
Ooh, ooh
To live and not to breathe
Is to die in tragedy
To run, to run away
To find what you believe
And I leave behind (ooh, ooh)
This hurricane of fucking lies (ooh, ooh)
I lost my faith to this
This town that don't exist
So I run, I run away
To the lights of masochists
And I leave behind (ooh, ooh)
This hurricane of fucking lies (ooh, ooh)
And I've walked this line (ooh, ooh)
A million and one fucking times (ooh, ooh)
But not this time!
I don't feel any shame, I won't apologize
When there ain't nowhere you can go
Running away from pain when you've been victimized
Tales from another broken... home!
You're leaving...
You're leaving...
You're leaving...
Ah, you're leaving home!
""",
    """
이제야 목적지를 정했지만
가려한 날 막아서네
난 갈 길이 먼데
새빨간 얼굴로 화를 냈던
친구가 생각나네
이미 난 발걸음을 떼었지만
가려한 날 재촉하네
걷기도 힘든데
새파랗게 겁에 질려 도망간
친구가 뇌에 맴도네
건반처럼 생긴 도로 위
수많은 동그라미들 모두가
멈췄다 굴렀다 말은 잘 들어
그건 나도 문제가 아냐
붉은색 푸른색
그 사이 3초 그 짧은 시간
노란색 빛을 내는
저기 저 신호등이
내 머릿속을 텅 비워버려
내가 빠른 지도
느린지도 모르겠어
그저 눈앞이 샛노랄 뿐이야
솔직히 말하자면 차라리
운전대를 못 잡던 어릴 때가
더 좋았었던 것 같아
그땐 함께 온 세상을 거닐
친구가 있었으니
건반처럼 생긴 도로 위
수많은 조명들이 날 빠르게
번갈아 가며 비추고 있지만
난 아직 초짜란 말이야
붉은색 푸른색
그 사이 3초 그 짧은 시간
노란색 빛을 내는
저기 저 신호등이
내 머릿속을 텅 비워버려
내가 빠른 지도
느린지도 모르겠어
그저 눈앞이 샛노랄 뿐이야
꼬질꼬질한 사람이나
부자 곁엔 아무도 없는
삼색 조명과 이색 칠 위에
서 있어 괴롭히지 마
붉은색 푸른색
그 사이 3초 그 짧은 시간
노란색 빛을 내는
저기 저 신호등이
내 머릿속을 텅 비워버려
내가 빠른지도
느린지도 모르겠어
그저 눈앞이 샛노랄 뿐이야
""",
    """
    You know
내 스타일이 아닌 음악을 들어도
You know
좋아하지 않는 음식을 먹어도
우산 없이 비가 와 홀딱 다 젖어도 좋아
I love it because I love you

우리 관계 디비디비딥
매일 봐도 처음같이 비기비기닝
춤추고 싶어 빌리빌리진
우리 앞 우리 옆 시기시기질투
자유로운 날갯짓 훨훨훨
꽃송이가 나를 삼킬 걸
알면서 난 뛰어들었어
Jump j-j-jump jump jump jump

So lovely day so lovely
Errday with you so lovely
Du durudu durudu du durudu

Spell L.o.v.e.L.e.e
이름만 불러도 you can feel me
눈빛만 봐도 알면서 my love

You know
아끼는 옷에 coffee를 쏟아도
You know
내가 준 목걸이를 잃어버려도
한 번 더 같은 것 사준 걸 다시 또 잃어도 좋아
I don't care I just care about you

여기 어때 how you like that
다 퍼주고 될게 빈털터리
Jubilate 박수갈채
안 반하고 누가 배겨 love sick
비타민 U가 필요해
만들어줘 샛노랗게
체크무늬 벌집 두 눈에
Honey help honey help

So lovely day so lovely
Errday with you so lovely
Du durudu durudu du durudu

Spell L.o.v.e.L.e.e
이름만 불러도 you can feel me
눈빛만 봐도 알면서 my love

누구 사랑 먹고 그리 이쁘게 컸니
Mommy or your daddy or them both
Lovey-dovey things 너의 곁에 everyday
Good morning good night
너의 이름 부를 때

So lovely day so lovely
Errday with you so lovely
Du durudu durudu du durudu

Spell L.o.v.e.L.e.e
이름만 불러도 you can feel me
눈빛만 봐도 알면서 my love
""",
    """
졸졸 따라다녀 어떡해 나
마치 피리 부는 사나이처럼
내겐 없어 딴 맘
오늘은 좀 바빠 my
Don't you know me?
I'm just staring at your eyes
앞을 향해 달려 Go away
뒤도 돌아보지 말고
나를 떠나 또
Don't leave me
어쩌면 그렇게 너
눈치 보지 말고
조금씩 맘을 열어 Open door
도대체 어디서 뭘 했던 거야
너는 나빠 Bad boy
But I'm 이미 알아
자꾸 도망치지마
주위를 맴돌고 있어
Don't think about it no more
I'm get you ouuta my mind
너는 대체 누구야
Tell me What you got What you got
옳지 내게 와봐 위험하지 않아
내 손을 먼저 줄게 천천히 생각해봐
네가 충분히 가까이 왔을 때 그때
널 잡고 놓아주지 않을 거야
You gotta be kidding kidding yes ah yeh ah yeh
just not a kiddy kiddy no ah yeh ah yeh
오래 기다리진 않아 설렘은 잠깐이니까
You gotta be kidding kidding yes ah yeh ah yeh
Go on Go on, my Warning Warning
너 땜에 worry(ing) worry(ing)
자꾸만 시선을 뺏겨 괜히 넌 툭툭
Hitting my heart
I’ll be your cutie pie
자꾸 가까워져 너너너
네 옆자리를 knock knock knock
Let's Get it Get it Get it Get it Yeah
괜찮아 눈치 보지 말고
가까이 다가와 Take my hand
그렇게 하나하나 시작해봐
너는 나빠 Bad boy
But I'm 이미 알아
자꾸 도망치지마
주위를 맴돌고 있어
Don't think about it no more
I'm get you ouuta my mind
너는 대체 누구야
Tell me What you got What you got
옳지 내게 와봐 위험하지 않아
내 손을 먼저 줄게 천천히 생각해봐
네가 충분히 가까이 왔을 때 그때
널 잡고 놓아주지 않을 거야
You gotta be kidding kidding yes ah yeh ah yeh
just not a kiddy kiddy no ah yeh ah yeh
오래 기다리진 않아 설렘은 잠깐이니까
You gotta be kidding kidding yes ah yeh ah yeh
좁혀지는 이 거리감이 좋아
괜찮을지 걱정하지 마
하나부터 열까지 0부터 100까지
완벽할 필요는 없는 거야
You gonna love me
We got a feeling
You gonna hold me me me
Take me to your world
You gotta be kidding kidding yes ah yeh ah yeh
just not a kiddy kiddy no ah yeh ah yeh
오래 기다리진 않아 설렘은 잠깐이니까
You gotta be kidding kidding yes ah yeh ah yeh
ah yeh ah yeh
ah yeh ah yeh
오래 기다리진 않아 설렘은 잠깐이니까
You gotta be kidding kidding yes ah yeh ah yeh
""",
    """
I'm super shy, super shy
But wait a minute while I
Make you mine, make you mine
떨리는 지금도
You're on my mind
All the time
I wanna tell you but I'm
Super shy, super shy

I'm super shy, super shy
But wait a minute while I
Make you mine, make you mine
떨리는 지금도
You're on my mind
All the time
I wanna tell you but I'm
Super shy, super shy

And I wanna go out with you
Where you wanna go? (Huh?)
Find a lil spot
Just sit and talk
Looking pretty
Follow me
우리 둘이 나란히
보이지? (봐)
내 눈이 (heh)
갑자기
빛나지
When you say
I'm your dream

You don't even know my name
Do ya?
You don't even know my name
Do ya-a?
누구보다도

I'm super shy, super shy
But wait a minute while I
Make you mine, make you mine
떨리는 지금도
You're on my mind
All the time
I wanna tell you but I'm
Super shy, super shy

I'm super shy, super shy
But wait a minute while I
Make you mine, make you mine
떨리는 지금도
You're on my mind
All the time
I wanna tell you but I'm
Super shy, super shy

나 원래 말도 잘하고 그런데 왜 이런지
I don't like that
Something odd about you
Yeah you're special and you know it
You're the top babe

I'm super shy, super shy
But wait a minute while I
Make you mine, make you mine
떨리는 지금도
You're on my mind
All the time
I wanna tell you but I'm
Super shy, super shy

I'm super shy, super shy
But wait a minute while I
Make you mine, make you mine
떨리는 지금도
You're on my mind
All the time
I wanna tell you but I'm
Super shy, super shy

You don't even know my name
Do ya?
You don't even know my name
Do ya-a?
누구보다도
You don't even know my name
Do ya?
You don't even know my name
Do ya-a?
""",
    """
Umm 내가 슬플 때마다
이 노래가 찾아와
세상이 둥근 것처럼 우린 동글동글
인생은 회전목마
우린 매일 달려가
언제쯤 끝나 난 잘 몰라 (huh, huh, huh)
어머 어머, 벌써 벌써 정신없이 달려왔어 왔어
Speed up speed up 어제로 돌아가는 시곌 보다가
어려워 어려워 어른이 되어가는 과정이 uh huh
On the road, twenty four 시간이 아까워 uh huh
Big noise, everything brand new
어렸을 때처럼 바뀌지 않는 걸
찾아 나섰단 말야 왜냐면 그때가 더 좋았어 난
So let me go back
타임머신 타고 I'll go back
승호가 좋았을 때처럼만
내가 슬플 때마다
이 노래가 찾아와
세상이 둥근 것처럼 우리
인생은 회전목마
우린 매일 달려가
언제쯤 끝나 난 잘 몰라
빙빙 돌아가는 회전목마처럼
영원히 계속될 것처럼
빙빙 돌아온 우리의 시간처럼
인생은 회전목마 ayy
어머 어머 벌써 벌써 정신없이 달려왔어 왔어
Speed up speed up 어제로 돌아가는 시곌 보다가
청춘까지 뺏은 현재 현재
탓할 곳은 어디 없네
Twenty two 세에게 너무 큰 벽
그게 말로 하고 싶어도 어려웠어
가끔은 어렸을 때로 돌아가
불가능하단 건 나도 잘 알아
그 순간만 고칠 수 있다면
지금의 나는 더 나았을까
달려가는 미터기 돈은 올라가
기사님과 어색하게 눈이 맞아
창문을 열어보지만 기분은 좋아지지 않아
그래서 손을 밖으로 쭉 뻗어 쭉 뻗어
흔들리는 택시는 어느새
목적지에 도달했다고 해
방 하나 있는 내 집 안의 손에 있던 짐들은
내가 힘들 때마다
이 노래가 찾아와
세상이 둥근 것처럼 우리
인생은 회전목마
우린 계속 달려가
언제쯤 끝날지 잘 몰라
빙빙 돌아가는 회전목마처럼
영원히 계속될 것처럼
빙빙 돌아온 우리의 시간처럼
우 인생은 회전목마
I'm on a TV show
You would never even know
사실 얼마나 많이 불안했는지
정신없이 돌아서
어딜 봐야 할지 모르겠어
들리나요 여길 보란 말이
빙빙 돌아가는 회전목마처럼
영원히 계속될 것처럼
빙빙 돌아온 우리의 시간처럼
인생은 회전목마
빙빙 돌아가는 회전목마처럼
영원히 계속될 것처럼
빙빙 돌아온 우리의 시간처럼
인생은 회전목마
""",
    """
As a child, you would wait
And watch from far away
But you always knew that you'd be the one
That work while they all play
In youth, you'd lay
Awake at night and scheme
Of all the things that you would change
But it was just a dream
Here we are, don't turn away now
We are the warriors that built this town
Here we are, don't turn away now
We are the warriors that built this town
From dust
Here we are, don't turn away now (don't turn away)
We are the warriors that built this town (don't turn away)
We are the warriors that built this town
From dust
""",
    """
내가 만든 집에서 모두 함께 노래를 합시다
소외됐던 사람들 모두 함께 노래를 합시다
우리만의 따뜻한 불, 영원한 꿈, 영혼과 삶
난 오늘 떠날 거라 생각을 했어
날 미워하지 마
No pain, no fail, 음악 없는 세상
Nowhere, no fear, 바다 같은 색깔
No cap, no cry, 이미 죽은 사람 아냐, 사실
태양에 맡겨 뒀던 가족과 모든 분들의 사랑
밤안개 짙어진 뒤 훔치려고 모인 자경단
난, 난 오늘 떠날 거라고 생각했어
날 미워하지 마
No pain, no fail, 음악 없는 세상
Nowhere, no fear, 바다 같은 색깔
No cap, no cry, 이미 죽은 사람 아냐
No pain, no fail, 음악 없는 세상
Nowhere, no fear, 바다 같은 색깔
No cap, no cry, 이미 죽은 사람 아냐, 사실
""",
    """
밤새 모니터에 튀긴 침이
마르기도 전에 강의실로
아 참 교수님이 문신 땜에 긴 팔 입고 오래
난 시작도 전에 눈을 감았지
날 한심하게 볼 게 뻔하니 이게 더 편해
내 새벽은 원래 일몰이 지나고
하늘이 까매진 후에야 해가 뜨네
내가 처량하다고 다 그래
야 야 난 쟤들이 돈 주고 가는
파리의 시간을 사는 중이라 전해
난 이게 궁금해
시계는 둥근데 날카로운 초침이
내 시간들을 아프게
모두가 바쁘게
뭐를 하든 경쟁하라 배웠으니
우린 우리의 시차로 도망칠 수밖에
이미 저 문밖엔 모두 그래
야 일찍 일어나야 성공해 안 그래?
맞는 말이지 다 근데
니들이 꿈을 꾸던 그 시간에
나도 꿈을 꿨지
두 눈 똑바로 뜬 채로
We're livin' in a different time zone
바뀌어버린 낮과 밤이야 yeah
Have a good night 먼저 자
아직 난 일하는 중이야
We are who we are
We a a are who we a a are oh ahh
Don't you know who we are?
4호선 문이 열릴 때
취해 있는 사람들과 날 똑같이 보지마
그들이 휘청거릴 때마다
풍기는 술 냄새마저 부러웠지만
난 적응해야 했거든 이 시차
꿈을 꾸게 해 준 침댄 이 기차
먼지 쌓일 틈이 없던 키보드 위
그리고 2009년부터 지금까지 계속
GRAY on the beat ya
아침은 까맣고 우리의 밤은 하얘
난 계속 칠하고 있고 똑같은 기찰 타네
걱정한 적 없어 막차 시간은 한 번도
얇았던 커튼이 햇빛을 완벽히 못 가려도
난 지금 눈을 감아야 해
내일의 나는 달라져야 해
우린 아무것도 없이 여길 올라왔고
넌 이 밤을 꼭 기억해야 돼
We're livin' in a different time zone
바뀌어버린 낮과 밤이야 yeah
Have a good night 먼저 자
아직 난 일하는 중이야
We are who we are
We a a are who we a a are oh ahh
Don't you know who we are?
밤새 모니터에 튀긴 침이
마르기도 전에 대기실로
아 참 문신 땜에 긴 팔 입고 오래
녹화 전에 눈을 감고 생각하지
똑같은 행동 다른 느낌
시차 부적응에 해당돼
지금 내 옆엔 Loco 그리고 GRAY
모두 비웃었던 동방의 소음이 어느새
전국을 울려대
야 이게 우리 시차의 결과고
우린 아직 여기 산다 전해
We're livin' in a different time zone
바뀌어버린 낮과 밤이야 yeah
Have a good night 먼저 자
아직 난 일하는 중이야
We are who we are
We a a are who we a a are oh ahh
Don't you know who we are?
모두 위험하다는 시간이 우린 되려 편해
Don't you know who we are?
밝아진 창문 밖을 봐야지 비로소 맘이 편해
Don't you know who we are?
모두가 다 피하는 반지하가 우린 편해
Don't you know who we are?
We are We are We are
Don't you know who we are?
""",
    """
When you try your best, but you don't succeed
When you get what you want, but not what you need
When you feel so tired, but you can't sleep
Stuck in reverse
And the tears come streaming down your face
When you lose something you can't replace
When you love someone, but it goes to waste
Could it be worse?
Lights will guide you home
And ignite your bones
And I will try to fix you
And high up above, or down below
When you're too in love to let it go
But if you never try, you'll never know
Just what you're worth
Lights will guide you home
And ignite your bones
And I will try to fix you
Tears stream down your face
When you lose something you cannot replace
Tears stream down your face, and I
Tears stream down your face
I promise you I will learn from my mistakes
Tears stream down your face, and I
Lights will guide you home
And ignite your bones
And I will try to fix you
""",
    """
So are you happy now?
Finally happy now? Yeah
뭐 그대로야 난
다 잃어버린 것 같아
모든 게 맘대로 왔다가 인사도 없이 떠나
이대로는 무엇도 사랑하고 싶지 않아
다 해질 대로 해져버린
기억 속을 여행해
우리는 오렌지 태양 아래
그림자 없이 함께 춤을 춰
정해진 이별 따위는 없어
아름다웠던 그 기억에서 만나
Forever young
우우우 우우우우 우우우 우우우우
Forever we young
우우우 우우우우
이런 악몽이라면 영영 깨지 않을게 SUGA
섬, 그래 여긴 섬, 서로가 만든 작은 섬
예, 음 forever young 영원이란 말은 모래성
작별은 마치 재난문자 같지
그리움과 같이 맞이하는 아침
서로가 이 영겁을 지나
꼭 이 섬에서 다시 만나
지나듯 날 위로하던 누구의 말대로 고작
한 뼘짜리 추억을 잊는 게 참 쉽지 않아
시간이 지나도 여전히
날 붙드는 그 곳에
우리는 오렌지 태양 아래
그림자 없이 함께 춤을 춰
정해진 안녕 따위는 없어
아름다웠던 그 기억에서 만나 forever young
우리는 서로를 베고 누워
슬프지 않은 이야기를 나눠
우울한 결말 따위는 없어
난 영원히 널 이 기억에서 만나
Forever young
우우우 우우우우 우우우 우우우우
Forever we young
우우우 우우우우
이런 악몽이라면 영영 깨지 않을게
""",
    """
I do the same thing, I told you that I never would
I told you I changed, even when I knew I never could
I know that I can't find nobody else as good as you
I need you to stay, need you to stay, hey
I get drunk, wake up, I'm wasted still
I realize the time that I wasted here
I feel like you can't feel the way I feel
I'll be fucked up if you can't be right here
Oh-whoa oh-whoa, whoa
Oh-whoa oh-whoa, whoa
Oh-whoa oh-whoa, whoa
I'll be fucked up if you can't be right here
I do the same thing, I told you that I never would
I told you I changed, even when I knew I never could
I know that I can't find nobody else as good as you
I need you to stay, need you to stay, hey
I do the same thing, I told you that I never would
I told you I changed, even when I knew I never could
I know that I can't find nobody else as good as you
I need you to stay, need you to stay, yeah
When I'm away from you, I miss your touch ooh-ooh
You're the reason I believe in love
It's been difficult for me to trust ooh-ooh
And I'm afraid that I'ma fuck it up
Ain't no way that I can leave you stranded
'Cause you ain't never left me empty-handed
And you know that I know that I can't live without you
So, baby, stay
Oh-whoa oh-whoa, whoa
Oh-whoa oh-whoa, whoa
Oh-whoa oh-whoa, whoa
I'll be fucked up if you can't be right here
I do the same thing, I told you that I never would
I told you I changed, even when I knew I never could
I know that I can't find nobody else as good as you
I need you to stay, need you to stay, yeah
I do the same thing, I told you that I never would
I told you I changed, even when I knew I never could
I know that I can't find nobody else as good as you
I need you to stay, need you to stay, hey
Whoa-oh
I need you to stay, need you to stay, hey
""",
    """
Don't ever say it's over if I'm breathin'
Racin' to the moonlight and I'm speedin'
I'm headed to the stars, ready to go far
I'm star walkin'
Don't ever say it's over if I'm breathin'
Racin' to the moonlight and I'm speedin'
I'm headed to the stars, ready to go far
I'm star walkin' star walkin'
On a mission to get high up
I know that I'ma die reachin' for a life that I don't really need at all
Never listened to replies, learned a lesson from the wise
You should never take advice from a nigga that ain't try
They said I wouldn't make it out alive
They told me I would never see the rise
That's why I gotta kill 'em every time
Gotta watch 'em bleed too
Don't ever say it's over if I'm breathin'
Racin' to the moonlight and I'm speedin'
I'm headed to the stars, ready to go far
I'm star walkin'
Don't ever say it's over if I'm breathin'
Racin' to the moonlight and I'm speedin'
I'm headed to the stars, ready to go far
I'm star walkin'
Been that nigga since I came out my mama woo
Thankin' God daddy never wore a condom woo
Prove 'em wrong every time 'til it's normal
Why worship legends when you know that you can join 'em?
These niggas don't like me, they don't like me
Likely, they wanna fight me
Come on, try it out, try me
They put me down, but I never cried out, "Why me?"
Word from the wise
Don't put worth inside a nigga that ain't try
They said I wouldn't make it out alive
They told me I would never see the rise
That's why I gotta kill 'em every time
Gotta watch 'em bleed too
Don't ever say it's over if I'm breathin'
Racin' to the moonlight and I'm speedin'
I'm headed to the stars, ready to go far
I'm star walkin'
Don't ever say it's over if I'm breathin'
Racin' to the moonlight and I'm speedin'
I'm headed to the stars, ready to go far
I'm star walkin'
I'm star walkin'
Don't ever say it's over if I'm breathin'
Racin' to the moonlight and I'm speedin'
I'm headed to the stars, ready to go far
I'm star walkin'
"""
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

        hannanum = Mecab()

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
