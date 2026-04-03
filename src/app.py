import streamlit as st
from pathlib import Path
import joblib
import numpy as np

# Streamlit Cloud cwd = 레포 루트, app은 src/app.py → pkl은 src/ 또는 루트에 두면 됨
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_MODEL_NAMES = ("nb_model.pkl", "tfidf_vectorizer.pkl")

# 하단 참고 패널 — 질문/본문은 아래 상수만 수정하면 됨
_REF_SCROLL_H = 280

_REF_LEFT_TITLE = "GPT 텍스트"
_REF_LEFT_QUESTION = "what is love please describe in 3000 characters?"
_REF_LEFT_BODY = """Love is one of the most profound and complex human experiences, yet it resists simple definition. It is not merely an emotion that appears suddenly and disappears just as quickly; rather, it is a layered experience that blends feeling, choice, connection, and meaning. People often recognize love not through explanation but through experience — in moments of care, understanding, and shared existence with another person.

At its foundation, love begins with connection. Human beings naturally seek closeness because relationships provide safety and belonging. From early childhood, love appears as attachment between a child and caregiver, forming the emotional blueprint for future relationships. This early bond teaches trust: the belief that another person can provide comfort and stability. As individuals grow, this need for connection evolves into friendships, family bonds, and romantic relationships.

Romantic love often starts with attraction and fascination. A person becomes uniquely important; their presence brings excitement, and their absence feels noticeable. Biology plays a role here, as brain chemicals linked to reward and attachment create feelings of happiness and anticipation. Yet attraction alone does not sustain love. Over time, the intensity of early passion naturally softens, making space for deeper emotional intimacy.

Intimacy develops through understanding. Love grows when people reveal their authentic selves — fears, dreams, weaknesses, and hopes — and feel accepted rather than judged. This acceptance creates emotional safety, allowing vulnerability. Trust becomes central: without trust, affection remains fragile; with it, relationships gain resilience. Love, therefore, is not about perfection but about mutual recognition of imperfection.

An important truth about love is that it is also an action. Feelings fluctuate, but loving behavior involves deliberate choices: listening carefully, showing patience during conflict, offering support without immediate reward. Acts of kindness, forgiveness, and consistency transform affection into lasting bonds. In long-term relationships, love often shifts from intense emotion to steady commitment — less dramatic but more enduring.

Love also expands the sense of self. When people care deeply for someone, empathy increases; another person’s joy becomes meaningful, and their suffering evokes genuine concern. This shared emotional experience creates a feeling of unity while still preserving individuality. Healthy love does not erase identity but encourages growth, allowing both individuals to become stronger versions of themselves.

Beyond romance, love exists in friendship, family ties, and even compassion toward humanity. Friendship offers loyalty and shared understanding without possessiveness. Familial love provides continuity and belonging. Compassionate love extends outward, motivating acts of kindness toward strangers. Each form reflects the same underlying principle: valuing another being’s existence and wellbeing.

Ultimately, love is both simple and infinite. It is found in grand gestures and small daily actions — a conversation, a sacrifice, a moment of presence. Love cannot eliminate pain or uncertainty, yet it gives those experiences meaning. To love is to recognize another life as significant alongside one’s own and to choose connection despite vulnerability. In this way, love becomes not just a feeling but a way of relating to the world — a continuous act of care, understanding, and shared humanity."""

_REF_RIGHT_TITLE = "VOGUE assey"
_REF_RIGHT_QUESTION = "what is love?"
_REF_RIGHT_BODY = """By Harper Lee

EDITOR'S NOTE: Here, for Vogue_, is the first article ever written by Harper Lee, a shy young woman who has an engaging drawl, immense happy eyes, and, this year, the pleasure of having written an uncommon novel:_ To Kill a Mockingbird_. Besides being good,_ Mockingbird is that literary rara avis, a first novel that sells. (In this country, it has sold more than half a million copies; abroad, it delighted the English and has been translated into German, French, Swedish, Danish, Norwegian, Dutch, and Italian. A pair of independent producers now has the movie rights.) Not unlike someone who might crop up in her own fiction, Nelle Harper Lee lives with her father and sister in a small Alabama town; they practice law, she writes. (A nonpracticing lawyer, she studied a year after law school as a Fulbright scholar at Oxford, then worked a stretch as a reservations clerk for BOAC.)

Many years ago an aging member of the House of Hanover, on learning that the duty of providing an heir to the throne of England had suddenly befallen him and his brothers, confided his alarm to his friend Thomas Creevey: " . . . It is now seven-and-twenty years that Madame St. Laurent and I have lived together; we are of the same age and have been in all climates, and in all difficulties together, and you may well imagine the pang it will occasion me to part with her . . . . I protest I don't know what is to become of her if a marriage is to be forced upon me . . . ."


Advertisement

Amused by the Duke of Kent's predicament. Mr. Creevey recorded the incident in his diary and preserved for us a timeless declaration. The man who made it was not overly endowed with brilliance, nor had he led a noteworthy life, yet we remember his cry from the heart and tend to forget his ultimate service to mankind: He was the father of Queen Victoria.

What did the Duke of Kent tell us? That two people had shared their lives on a voluntary basis for nearly thirty years—in itself a remarkable achievement; that they had survived the fevers and frets of intimate relationship; that together they had met the pressures and disappointments of life; that he is in agony at the prospect of leaving her. In one graceful sentence, the Duke of Kent said all there is to say about the love of a man for a woman.

And in so saying, he tells us much about love itself. There is only one kind of love—love. But the different manifestations of love are uncountable:

At an unfamiliar night noise a mother will spring from bed, not to return until every corner of her domain is tucked safely round her anxiety. A man will look up from his golf game to watch a jet cut caterpillar tracks through the sky. A housewife, before driving to town, will give her neighbour a quick call to see if she wants anything from the store. These are manifestations of a power within us that must of necessity be called divine, for it is no invention of man.

What is love? Many things are like love—indeed, love is present in pity, compassion, romance, affection. What made the Duke of Kent's statement a declaration of love, and what makes us perform without second thought small acts of love every day of our lives, is an element conspicuous by its absence. Were it present, the Duke of Kent would have left his mistress without a pang; the sound barrier breaking over her head would not rouse the mother; sinking his putt would be the primary aim of the golfer; the housewife would go straight to the store with no thought of her neighbour. One thing identifies love and isolates it from kindred emotions: Love admits not of self.

Few of us achieve compassion; to some of us romance is a word; in many of us the ability to feel affection has long since died; but all of us at one time or another—be it for an instant or for our lives—have departed from ourselves: We have loved something or someone. Love, then, is a paradox: To have it, we must give it. Love is not an intransitive thing—love is a direct action of mind and body. Without love, life is pointless and dangerous. Man is on his way to Venus, but he still hasn't learned to live with his wife. Man has succeeded in increasing his life span, yet he exterminates his brothers six million at a whack. Man now has the power to destroy himself and his planet: Depend upon it, he will—should he cease to love.


Most Popular
19 Foods With More Protein Than Eggs
Wellness
19 Foods With More Protein Than Eggs
By Philipp Wehsack
21 Chic Hairstyles for Women Over 50, Inspired By Celebrities
Beauty
21 Chic Hairstyles for Women Over 50, Inspired By Celebrities
By Ranyechi Udemezue
Artist Monacco Dunn Looks for Vintage That Has "A Story or Perhaps a Ghost Still in the Threads"
Fashion
Artist Monacco Dunn Looks for Vintage That Has "A Story or Perhaps a Ghost Still in the Threads"
By Laird Borrelli-Persson
Advertisement

The most common barriers to love are greed, envy, pride, and four other drives formerly known as sins. There is one more just as dangerous: boredom. The mind that can find little excitement in life is a dying one; the mind that can not find something in the world that attracts it is dead, and the body housing it might as well be dead, for what are the uses of the five senses to a mind that takes no pleasure in them?

Having at long last realized that he must love or destroy himself, man is proceeding along his usual course by trying to evolve a science for it. The ultimate aim of psychoanalysis, when its special brand of semantics is put to rout, is to release man from his neuroses and thus enable him to love, and man's capacity to love is measured by his degree of freedom from the drives that turn inward upon him. As one holds down a cork to the bottom of a stream, so may love be imprisoned by self: Remove self, and love rises to the surface of man's being.

With love, all things are possible.

Love restores. We have heard many tales of love's power to heal, and we are skeptical of them, for we are human and therefore prone to deny the existence of things we do not understand and cannot explain. But this tale happened:

On an August evening in a tiny Southern hospital, an old man lay dying. His family had been summoned, among them his eldest grandson, a boy of sixteen. The boy's relationship with his grandfather had been a curious, almost wordless one, as such things often are between man and man. All that day the boy said nothing. It seemed that he could not talk. He would not wait out the old man's dying with the rest of his family in the hospital lobby; instead, the boy found a chair and stationed himself in the corridor beside his grandfather's door, where he sat all day, oblivious to the starched scurryings of hospital routine. Late in the evening the family's doctor found the boy still sitting, still silent. The doctor said, "Go home, son. There's nothing you can do for your grandfather." The boy took no notice of him, and the doctor went into the room only to emerge moments later, looking bewildered. "Er—son," said the doctor. The boy looked up. "He's asking for something to eat. He's better." Showing no sign of surprise, the boy nodded: "I reckoned it was about time he was hungry," he said, his first utterance of the day. Then he picked up the chair, put it back where he found it, and walked down the corridor, stretching his lanky frame and yawning. "Where are you going, boy?" called the doctor. "To get him a hamburger," answered the boy. "He likes hamburgers."

There is no satisfactory explanation for extrasensory perception—it simply is. There was no rational explanation for the old man's recovery—it simply happened. One may only wonder."""


def _resolve_pkl(name: str) -> Path:
    for base in (_HERE, _ROOT):
        p = base / name
        if p.is_file():
            return p
    raise FileNotFoundError(
        f"'{name}' 을(를) 찾을 수 없습니다. 다음 중 한 곳에 넣어 주세요: "
        f"{_HERE} 또는 {_ROOT} (Git에 커밋되어 있어야 배포 환경에서 읽힙니다)"
    )


def _reference_panel(title: str, question: str, body: str, key_prefix: str) -> None:
    st.subheader(title)
    st.markdown(f"**{question}**")
    st.text_area(
        "본문",
        value=body,
        height=_REF_SCROLL_H,
        label_visibility="collapsed",
        key=f"{key_prefix}_body",
    )


# --- 1. 저장된 모델 및 벡터라이저 불러오기 ---
@st.cache_resource
def load_models():
    model = joblib.load(_resolve_pkl(_MODEL_NAMES[0]))
    vectorizer = joblib.load(_resolve_pkl(_MODEL_NAMES[1]))
    return model, vectorizer


st.set_page_config(page_title="AI 탐지기 시연", layout="wide")
model, vectorizer = load_models()

# --- 2. 메인 UI ---
st.title("실시간 AI 생성 텍스트 탐지 시스템")
st.write("AI 생성 문장 탐지 최종 모델 (나이브 베이즈) 기반 탐지기입니다.")

user_input = st.text_area("분석할 텍스트를 입력하세요", placeholder="내용을 입력하고 Ctrl+Enter를 누르세요.", height=200)

if st.button("AI 판별 시작"):
    if user_input:
        input_tfidf = vectorizer.transform([user_input])
        prediction = model.predict(input_tfidf)
        probabilities = model.predict_proba(input_tfidf)

        st.divider()
        if prediction[0] == 1:
            st.error("분석 결과: AI 생성 문장일 가능성이 매우 높습니다")
            st.write(f"확률: {probabilities[0][1]*100:.2f}%")
        else:
            st.success("분석 결과: 사람이 작성한 문장일 가능성이 매우 높습니다")
            st.write(f"확률: {probabilities[0][0]*100:.2f}%")

        st.progress(float(probabilities[0][1]))
    else:
        st.warning("텍스트를 입력해주세요.")

# --- 3. 하단: 참고 텍스트 2열 (스크롤) ---
st.divider()
st.markdown("##### 참고 텍스트")
col_left, col_right = st.columns(2)
with col_left:
    _reference_panel(_REF_LEFT_TITLE, _REF_LEFT_QUESTION, _REF_LEFT_BODY, "ref_left")
with col_right:
    _reference_panel(_REF_RIGHT_TITLE, _REF_RIGHT_QUESTION, _REF_RIGHT_BODY, "ref_right")
