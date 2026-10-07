import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pyttsx3

# Tamil voice
from gtts import gTTS
import os
import tempfile

import threading
import time
import av

from collections import deque
from queue import Queue, Empty

from streamlit_webrtc import (
    webrtc_streamer,
    WebRtcMode,
    VideoProcessorBase
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "leakage_free_augmented_word_model.keras"
CLASSES_PATH = "final_word_classes.npy"

SEQUENCE_LENGTH = 16

CONFIDENCE_THRESHOLD = 0.50

STABLE_PREDICTIONS = 6

# Prediction interval
PREDICTION_INTERVAL = 0.5


# ============================================================
# ENGLISH → TAMIL TRANSLATIONS
# ============================================================

TAMIL_TRANSLATIONS = {

    "drink": "குடி",
    "eat": "சாப்பிடு",
    "father": "அப்பா",
    "food": "உணவு",
    "friend": "நண்பர்",
    "go": "போ",
    "he": "அவன்",
    "hello": "வணக்கம்",
    "help": "உதவி",
    "hospital": "மருத்துவமனை",
    "market": "சந்தை",
    "mother": "அம்மா",
    "no": "இல்லை",
    "okay": "சரி",
    "please": "தயவுசெய்து",
    "school": "பள்ளி",
    "she": "அவள்",
    "sister": "சகோதரி",
    "sit": "உட்கார்",
    "student": "மாணவர்",
    "tea": "தேநீர்",
    "teacher": "ஆசிரியர்",
    "thank_you": "நன்றி",
    "today": "இன்று",
    "water": "தண்ணீர்",
    "what": "என்ன",
    "where": "எங்கே",
    "yes": "ஆம்",
    "you": "நீங்கள்"
}


# ============================================================
# TAMIL VOICE
# ============================================================

def speak_tamil(text):

    if not text.strip():
        return

    try:

        temp_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp3"
        )

        audio_path = temp_file.name

        temp_file.close()

        tts = gTTS(
            text=text,
            lang="ta"
        )

        tts.save(audio_path)

        os.startfile(audio_path)

    except Exception as error:

        print(
            "Tamil voice error:",
            error
        )


# ============================================================
# ENGLISH → TAMIL
# ============================================================

def translate_message_to_tamil(message):

    words = message.split()

    tamil_words = []

    for word in words:

        clean_word = word.lower().strip(
            ".,!?"
        )

        if clean_word in TAMIL_TRANSLATIONS:

            tamil_words.append(
                TAMIL_TRANSLATIONS[clean_word]
            )

        else:

            tamil_words.append(word)

    return " ".join(tamil_words)


# ============================================================
# STREAMLIT PAGE
# ============================================================

st.set_page_config(
    page_title="ISL Communication System",
    page_icon="🤟",
    layout="wide"
)


st.title(
    "🤟 Indian Sign Language Communication System"
)


st.write(
    "Show an ISL word using the webcam. "
    "The system recognizes words and builds the message automatically."
)


# ============================================================
# AI NEURAL NETWORK BACKGROUND
# ============================================================

st.markdown("""
<style>

.stApp {
    background: #030814;
    overflow: hidden;
}

.stApp::before {

    content: "";

    position: fixed;

    inset: 0;

    background-image:

        radial-gradient(
            circle at 15% 25%,
            rgba(0, 200, 255, 0.45) 0px,
            rgba(0, 200, 255, 0.20) 3px,
            transparent 6px
        ),

        radial-gradient(
            circle at 40% 70%,
            rgba(0, 255, 220, 0.40) 0px,
            rgba(0, 255, 220, 0.18) 3px,
            transparent 6px
        ),

        radial-gradient(
            circle at 70% 30%,
            rgba(100, 80, 255, 0.45) 0px,
            rgba(100, 80, 255, 0.20) 3px,
            transparent 6px
        ),

        radial-gradient(
            circle at 85% 75%,
            rgba(0, 180, 255, 0.40) 0px,
            rgba(0, 180, 255, 0.18) 3px,
            transparent 6px
        );

    background-size:
        280px 280px,
        360px 360px,
        320px 320px,
        400px 400px;

    animation: neuralNodes 18s linear infinite;

    opacity: 0.8;

    pointer-events: none;

    z-index: 0;
}

.stApp::after {

    content: "";

    position: fixed;

    inset: -20%;

    background:

        linear-gradient(
            35deg,
            transparent 42%,
            rgba(0, 180, 255, 0.10) 43%,
            transparent 44%
        ),

        linear-gradient(
            145deg,
            transparent 48%,
            rgba(0, 220, 200, 0.08) 49%,
            transparent 50%
        ),

        linear-gradient(
            70deg,
            transparent 45%,
            rgba(100, 80, 255, 0.08) 46%,
            transparent 47%
        );

    background-size:
        320px 320px,
        420px 420px,
        380px 380px;

    filter: blur(1px);

    animation: neuralNetwork 22s linear infinite;

    pointer-events: none;

    z-index: 0;
}

@keyframes neuralNodes {

    0% {
        background-position:
            0px 0px,
            0px 0px,
            0px 0px,
            0px 0px;
    }

    50% {
        background-position:
            140px 90px,
            -120px 100px,
            100px -80px,
            -150px -100px;
    }

    100% {
        background-position:
            280px 180px,
            -240px 200px,
            200px -160px,
            -300px -200px;
    }
}

@keyframes neuralNetwork {

    0% {
        transform: translate(0px, 0px);
    }

    50% {
        transform: translate(-40px, 30px);
    }

    100% {
        transform: translate(0px, 0px);
    }
}

[data-testid="stAppViewContainer"] {
    position: relative;
    z-index: 1;
}

[data-testid="stHeader"] {
    background: transparent;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    classes = np.load(
        CLASSES_PATH,
        allow_pickle=True
    )

    return model, classes


model, classes = load_model()


# ============================================================
# SHARED APPLICATION STATE
# ============================================================

if "shared_state" not in st.session_state:

    st.session_state.shared_state = {

        "prediction": "Waiting...",

        "confidence": 0.0,

        "message_words": [],

        "last_added_word": "",

        "last_add_time": 0.0
    }


SHARED = st.session_state.shared_state

STATE_LOCK = threading.Lock()


# ============================================================
# MEDIAPIPE
# ============================================================

mp_hands = mp.solutions.hands


# ============================================================
# EXTRACT POSITION FEATURES
#
# 21 landmarks × 3 coordinates × 2 hands
# = 126 features
# ============================================================

def extract_position_features(results):

    left = None
    right = None

    detected_hands = []

    # --------------------------------------------------------
    # GET DETECTED HANDS
    # --------------------------------------------------------

    if results.multi_hand_landmarks:

        for i, hand_landmarks in enumerate(
            results.multi_hand_landmarks
        ):

            points = np.array(
                [
                    [
                        landmark.x,
                        landmark.y,
                        landmark.z
                    ]
                    for landmark in hand_landmarks.landmark
                ],
                dtype=np.float32
            )

            label = "Right"

            if (
                results.multi_handedness
                and i < len(results.multi_handedness)
            ):

                label = (
                    results
                    .multi_handedness[i]
                    .classification[0]
                    .label
                )

            detected_hands.append(
                (label, points)
            )

    # --------------------------------------------------------
    # ASSIGN LEFT / RIGHT
    # --------------------------------------------------------

    for label, points in detected_hands:

        if label == "Left" and left is None:

            left = points

        elif label == "Right" and right is None:

            right = points

    # --------------------------------------------------------
    # FALLBACK
    # --------------------------------------------------------

    if (
        left is None
        and right is None
        and len(detected_hands) > 0
    ):

        if len(detected_hands) == 1:

            right = detected_hands[0][1]

        else:

            right = detected_hands[0][1]

            left = detected_hands[1][1]

    # --------------------------------------------------------
    # EMPTY HAND
    # --------------------------------------------------------

    if left is None:

        left = np.zeros(
            (21, 3),
            dtype=np.float32
        )

    if right is None:

        right = np.zeros(
            (21, 3),
            dtype=np.float32
        )

    # --------------------------------------------------------
    # WRIST NORMALIZATION
    # --------------------------------------------------------

    left_wrist = left[0].copy()

    right_wrist = right[0].copy()

    left_normalized = (
        left - left_wrist
    )

    right_normalized = (
        right - right_wrist
    )

    # --------------------------------------------------------
    # 126 FEATURES
    # --------------------------------------------------------

    position = np.concatenate(
        [
            left_normalized.flatten(),
            right_normalized.flatten()
        ]
    )

    return position.astype(
        np.float32
    )


# ============================================================
# CREATE 378 FEATURES
#
# POSITION      = 126
# VELOCITY      = 126
# ACCELERATION  = 126
#
# TOTAL         = 378
# ============================================================

def create_378_features(position_sequence):

    position = np.asarray(
        position_sequence,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # VELOCITY
    # --------------------------------------------------------

    velocity = np.zeros_like(
        position
    )

    velocity[1:] = (
        position[1:]
        - position[:-1]
    )

    # --------------------------------------------------------
    # ACCELERATION
    # --------------------------------------------------------

    acceleration = np.zeros_like(
        velocity
    )

    acceleration[1:] = (
        velocity[1:]
        - velocity[:-1]
    )

    # --------------------------------------------------------
    # CONCATENATE
    # --------------------------------------------------------

    features = np.concatenate(
        [
            position,
            velocity,
            acceleration
        ],
        axis=1
    )

    return features.astype(
        np.float32
    )


# ============================================================
# VIDEO PROCESSOR
# ============================================================

class ISLProcessor(VideoProcessorBase):

    def __init__(self):

        # ----------------------------------------------------
        # MEDIAPIPE
        # ----------------------------------------------------

        self.hands = mp_hands.Hands(

            static_image_mode=False,

            max_num_hands=2,

            min_detection_confidence=0.5,

            min_tracking_confidence=0.5
        )

        # ----------------------------------------------------
        # POSITION SEQUENCE
        # ----------------------------------------------------

        self.position_sequence = deque(
            maxlen=SEQUENCE_LENGTH
        )

        # ----------------------------------------------------
        # PREDICTION QUEUE
        # ----------------------------------------------------

        self.prediction_queue = Queue(
            maxsize=1
        )

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        self.result_word = "Waiting..."

        self.result_confidence = 0.0

        self.last_prediction = ""

        self.stable_count = 0

        self.last_submit_time = 0.0

        # ----------------------------------------------------
        # FRAME CONTROL
        # ----------------------------------------------------

        self.frame_count = 0

        # MediaPipe processes every 3rd frame
        self.process_every_n_frames = 4

        # ----------------------------------------------------
        # WORKER
        # ----------------------------------------------------

        self.running = True

        self.worker = threading.Thread(
            target=self.prediction_worker,
            daemon=True
        )

        self.worker.start()

    # ========================================================
    # BACKGROUND PREDICTION
    # ========================================================

    def prediction_worker(self):

        while self.running:

            try:

                sequence = self.prediction_queue.get(
                    timeout=0.1
                )

            except Empty:

                continue

            try:

                # ------------------------------------------------
                # CREATE 378 FEATURES
                # ------------------------------------------------

                features = create_378_features(
                    sequence
                )

                # ------------------------------------------------
                # CHECK SHAPE
                # ------------------------------------------------

                if features.shape != (
                    16,
                    378
                ):

                    print(
                        "ERROR: Wrong feature shape:",
                        features.shape
                    )

                    continue

                # ------------------------------------------------
                # MODEL INPUT
                # ------------------------------------------------

                model_input = np.expand_dims(
                    features,
                    axis=0
                )

                # ------------------------------------------------
                # PREDICTION
                # ------------------------------------------------

                prediction = model(
                    model_input,
                    training=False
                ).numpy()[0]

                # ------------------------------------------------
                # BEST CLASS
                # ------------------------------------------------

                class_index = int(
                    np.argmax(prediction)
                )

                confidence = float(
                    prediction[class_index]
                )

                word = str(
                    classes[class_index]
                )

                # ------------------------------------------------
                # RESULT
                # ------------------------------------------------

                self.result_word = word

                self.result_confidence = confidence

                # ------------------------------------------------
                # SHARED STATE
                # ------------------------------------------------

                with STATE_LOCK:

                    SHARED[
                        "prediction"
                    ] = word

                    SHARED[
                        "confidence"
                    ] = confidence

                # ------------------------------------------------
                # STABILITY
                # ------------------------------------------------

                if confidence >= CONFIDENCE_THRESHOLD:

                    if word == self.last_prediction:

                        self.stable_count += 1

                    else:

                        self.last_prediction = word

                        self.stable_count = 1

                    # ------------------------------------------------
                    # ADD WORD
                    # ------------------------------------------------

                    if (
                        self.stable_count
                        >= STABLE_PREDICTIONS
                    ):

                        self.add_word(
                            word
                        )

                        self.stable_count = 0

                else:

                    self.stable_count = 0

            except Exception as error:

                print(
                    "Prediction error:",
                    error
                )

    # ========================================================
    # ADD WORD
    # ========================================================

    def add_word(self, word):

        current_time = time.time()

        with STATE_LOCK:

            last_word = SHARED[
                "last_added_word"
            ]

            last_time = SHARED[
                "last_add_time"
            ]

            # ------------------------------------------------
            # PREVENT SAME SIGN REPEATING
            # ------------------------------------------------

            if word == last_word:

                return

            # ------------------------------------------------
            # COOLDOWN
            # ------------------------------------------------

            if (
                current_time - last_time
                < 1.0
            ):

                return

            # ------------------------------------------------
            # ADD WORD
            # ------------------------------------------------

            SHARED[
                "message_words"
            ].append(word)

            SHARED[
                "last_added_word"
            ] = word

            SHARED[
                "last_add_time"
            ] = current_time

    # ========================================================
    # CAMERA FRAME
    # ========================================================

    def recv(self, frame):

        # ----------------------------------------------------
        # CAMERA IMAGE
        # ----------------------------------------------------

        image = frame.to_ndarray(
            format="bgr24"
        )

        # ----------------------------------------------------
        # COUNT FRAMES
        # ----------------------------------------------------

        self.frame_count += 1

        # ----------------------------------------------------
        # PROCESS EVERY 3RD FRAME
        # ----------------------------------------------------

        if (
            self.frame_count
            % self.process_every_n_frames
            == 0
        ):

            # ------------------------------------------------
            # SMALL IMAGE FOR MEDIAPIPE
            # ------------------------------------------------

            height, width = image.shape[:2]

            if width > 480:

                scale = 480.0 / width

                process_image = cv2.resize(

                    image,

                    (
                        480,
                        int(height * scale)
                    ),

                    interpolation=cv2.INTER_AREA
                )

            else:

                process_image = image

            # ------------------------------------------------
            # BGR → RGB
            # ------------------------------------------------

            rgb_image = cv2.cvtColor(
                process_image,
                cv2.COLOR_BGR2RGB
            )

            # ------------------------------------------------
            # MEDIAPIPE
            # ------------------------------------------------

            results = self.hands.process(
                rgb_image
            )

            # ------------------------------------------------
            # EXTRACT POSITION
            # ------------------------------------------------

            position = extract_position_features(
                results
            )

            # ------------------------------------------------
            # ADD POSITION
            # ------------------------------------------------

            self.position_sequence.append(
                position
            )

            # ------------------------------------------------
            # SEND TO MODEL
            # ------------------------------------------------

            current_time = time.time()

            if (
                len(self.position_sequence)
                == SEQUENCE_LENGTH
                and
                current_time - self.last_submit_time
                >= PREDICTION_INTERVAL
            ):

                sequence_copy = np.array(
                    self.position_sequence,
                    dtype=np.float32
                )

                # Only submit if worker is free

                if self.prediction_queue.empty():

                    try:

                        self.prediction_queue.put_nowait(
                            sequence_copy
                        )

                        self.last_submit_time = (
                            current_time
                        )

                    except:

                        pass

        # ----------------------------------------------------
        # CAMERA DISPLAY
        # ----------------------------------------------------

        display_image = image.copy()

        # ----------------------------------------------------
        # WORD
        # ----------------------------------------------------

        cv2.putText(

            display_image,

            f"Word: {self.result_word}",

            (15, 35),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.75,

            (0, 255, 0),

            2,

            cv2.LINE_AA
        )

        # ----------------------------------------------------
        # CONFIDENCE
        # ----------------------------------------------------

        cv2.putText(

            display_image,

            f"Confidence: "
            f"{self.result_confidence * 100:.1f}%",

            (15, 65),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.6,

            (255, 255, 0),

            2,

            cv2.LINE_AA
        )

        # ----------------------------------------------------
        # RETURN CAMERA
        # ----------------------------------------------------

        return av.VideoFrame.from_ndarray(
            display_image,
            format="bgr24"
        )


# ============================================================
# WEBRTC CAMERA
# ============================================================

col_camera, col_space = st.columns(
    [1, 2]
)


with col_camera:

    ctx = webrtc_streamer(

        key="isl-camera",

        mode=WebRtcMode.SENDRECV,

        video_processor_factory=ISLProcessor,

        rtc_configuration={
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:stun1.l.google.com:19302"]}
            ]
        },

        media_stream_constraints={
            "video": {
                "width": {"ideal": 640},
                "height": {"ideal": 480},
                "frameRate": {"ideal": 15, "max": 20}
            },
            "audio": False
        },

        async_processing=True
    )


# ============================================================
# MESSAGE DISPLAY
# ============================================================

@st.fragment(
    run_every=1.0
)

def display_results():

    with STATE_LOCK:

        words = list(
            SHARED["message_words"]
        )

        current_word = SHARED[
            "prediction"
        ]

        confidence = SHARED[
            "confidence"
        ]

    # --------------------------------------------------------
    # MESSAGE
    # --------------------------------------------------------

    st.subheader(
        "📝 Your Message"
    )

    message = " ".join(
        words
    )

    if message:

        st.success(
            message
        )

    else:

        st.info(
            "Show an ISL word to start building the message."
        )

    # --------------------------------------------------------
    # CURRENT RECOGNITION
    # --------------------------------------------------------

    st.subheader(
        "🔎 Current Recognition"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Current Word",
            current_word
        )

    with col2:

        st.metric(
            "Confidence",
            f"{confidence * 100:.1f}%"
        )


display_results()


# ============================================================
# CONTROLS
# ============================================================

st.subheader(
    "🎛️ Controls"
)


col1, col2, col3, col4, col5 = st.columns(
    5
)


# ============================================================
# SPEAK ENGLISH MESSAGE
# ============================================================

with col1:

    if st.button(
        "🔊 Speak Message",
        use_container_width=True
    ):

        with STATE_LOCK:

            text = " ".join(
                SHARED["message_words"]
            )

        if text:

            def speak_message():

                try:

                    engine = pyttsx3.init()

                    engine.say(text)

                    engine.runAndWait()

                    engine.stop()

                except Exception as error:

                    print(
                        "TTS error:",
                        error
                    )

            threading.Thread(
                target=speak_message,
                daemon=True
            ).start()

        else:

            st.warning(
                "No message to speak."
            )


# ============================================================
# SPEAK CURRENT ENGLISH WORD
# ============================================================

with col2:

    if st.button(
        "🗣️ Speak Current Word",
        use_container_width=True
    ):

        with STATE_LOCK:

            word = SHARED[
                "prediction"
            ]

        if word != "Waiting...":

            def speak_current_word():

                try:

                    engine = pyttsx3.init()

                    engine.say(word)

                    engine.runAndWait()

                    engine.stop()

                except Exception as error:

                    print(
                        "TTS error:",
                        error
                    )

            threading.Thread(
                target=speak_current_word,
                daemon=True
            ).start()

        else:

            st.warning(
                "No word detected yet."
            )


# ============================================================
# SPEAK TAMIL MESSAGE
# ============================================================

with col3:

    if st.button(
        "🗣️ Speak Tamil",
        use_container_width=True
    ):

        with STATE_LOCK:

            english_message = " ".join(
                SHARED["message_words"]
            )

        if english_message:

            tamil_message = (
                translate_message_to_tamil(
                    english_message
                )
            )

            threading.Thread(
                target=speak_tamil,
                args=(tamil_message,),
                daemon=True
            ).start()

        else:

            st.warning(
                "No message to translate."
            )


# ============================================================
# UNDO
# ============================================================

with col4:

    if st.button(
        "↩️ Undo Last Word",
        use_container_width=True
    ):

        with STATE_LOCK:

            if SHARED["message_words"]:

                removed_word = (
                    SHARED[
                        "message_words"
                    ].pop()
                )

                SHARED[
                    "last_added_word"
                ] = ""

                SHARED[
                    "last_add_time"
                ] = 0.0

                st.success(
                    f"Removed: {removed_word}"
                )

            else:

                st.warning(
                    "No word to undo."
                )


# ============================================================
# CLEAR MESSAGE
# ============================================================

with col5:

    if st.button(
        "🗑️ Clear Message",
        use_container_width=True
    ):

        with STATE_LOCK:

            SHARED[
                "message_words"
            ] = []

            SHARED[
                "last_added_word"
            ] = ""

            SHARED[
                "last_add_time"
            ] = 0.0

        st.success(
            "Message cleared."
        )


# ============================================================
# WORD HISTORY
# ============================================================

with STATE_LOCK:

    history = list(
        SHARED["message_words"]
    )


if history:

    st.subheader(
        "📚 Word History"
    )

    for number, word in enumerate(
        history,
        start=1
    ):

        st.write(
            f"**{number}.** {word}"
        )