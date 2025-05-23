import sys
import nltk

# Download NLTK resource for English POS tagging
try:
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)
except Exception as e:
    print(f"Failed to download NLTK averaged_perceptron_tagger_eng: {str(e)}")

# Mock MeCab before any imports to avoid Japanese processing errors
class MockMeCab:
    class Tagger:
        def __init__(self, *args, **kwargs):
            pass  # Do nothing
        def parse(self, text):
            return text  # Return input text unchanged (bypass Japanese processing)

sys.modules['MeCab'] = MockMeCab

# Now import other modules
import os
import torch
import streamlit as st
from openvoice import se_extractor
from openvoice.api import ToneColorConverter
from melo.api import TTS
import time
from io import BytesIO
import soundfile as sf
import librosa

# Set page configuration
st.set_page_config(page_title="Voice Cloning App", layout="wide")

# Initialize session state
if 'target_se' not in st.session_state:
    st.session_state.target_se = None
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'recorded_audio' not in st.session_state:
    st.session_state.recorded_audio = None

# Cache models and resources
@st.cache_resource
def load_tone_color_converter(_device):
    try:
        converter = ToneColorConverter('checkpoints_v2/converter/config.json', device=_device)
        converter.load_ckpt('checkpoints_v2/converter/checkpoint.pth')
        return converter
    except Exception as e:
        st.error(f"Failed to load tone color converter: {str(e)}")
        raise

@st.cache_resource
def load_tts_model(_language, _device):
    try:
        st.write(f"Loading TTS model for language: {_language}")
        model = TTS(language=_language, device=_device)
        return model
    except Exception as e:
        st.error(f"Failed to load TTS model: {str(e)}")
        raise

@st.cache_resource
def load_source_se(_path, _device):
    try:
        return torch.load(_path, map_location=_device)
    except Exception as e:
        st.error(f"Failed to load source speaker embedding: {str(e)}")
        raise

def save_audio_to_wav(audio_data, output_path, sample_rate=16000):
    """Save audio data to WAV file."""
    try:
        if isinstance(audio_data, BytesIO):
            audio_data.seek(0)
        audio, sr = librosa.load(audio_data, sr=sample_rate, mono=True)
        sf.write(output_path, audio, sr, format='WAV')
    except Exception as e:
        st.error(f"Failed to save audio: {str(e)}")
        raise

def main():
    st.title("Voice Cloning with OpenVoice")
    st.markdown("Record your voice and generate cloned speech with custom text.")

    # Settings
    output_dir = 'outputs_v2'
    src_path = f'{output_dir}/tmp.wav'
    save_path = f'{output_dir}/output_v2_en_india.wav'
    reference_path = f'{output_dir}/reference.wav'
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    st.write(f"Using device: {device}")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load models and resources
    try:
        tone_color_converter = load_tone_color_converter(device)
        tts_model = load_tts_model("EN", device)
        source_se_path = 'checkpoints_v2/base_speakers/ses/en-india.pth'

        if not os.path.exists(source_se_path):
            st.error(f"Base speaker embedding not found at {source_se_path}")
            return

        source_se = load_source_se(source_se_path, device)
    except Exception as e:
        st.error(f"Error loading resources: {str(e)}")
        return

    # Input controls in main area
    st.header("Record Reference Voice")
    st.markdown("Click the microphone to record your voice. Click again to stop.")

    # Audio recorder using st.audio_input
    audio_value = st.audio_input("Record a voice message")

    # Handle recorded audio
    if audio_value and st.session_state.recorded_audio is None:
        try:
            st.session_state.recorded_audio = audio_value
            with st.spinner("Saving recorded audio..."):
                save_audio_to_wav(audio_value, reference_path)
                st.success("Audio recorded successfully!")
        except Exception as e:
            st.warning(f"Error processing audio: {str(e)}")
            st.session_state.recorded_audio = None

    # Extract speaker embedding
    if st.session_state.recorded_audio and st.session_state.target_se is None:
        if os.path.exists(reference_path):
            with st.spinner("Extracting speaker embedding..."):
                try:
                    st.session_state.target_se, _ = se_extractor.get_se(reference_path, tone_color_converter, vad=True)
                    torch.save(st.session_state.target_se, "cached_se.pt")
                    st.success("Speaker embedding extracted successfully!")
                except Exception as e:
                    st.error(f"Failed to extract speaker embedding: {str(e)}")
        elif os.path.exists("cached_se.pt"):
            with st.spinner("Loading cached speaker embedding..."):
                try:
                    st.session_state.target_se = torch.load("cached_se.pt", map_location=device)
                    st.success("Cached speaker embedding loaded!")
                except Exception as e:
                    st.error(f"Failed to load cached speaker embedding: {str(e)}")

    # Text and speed inputs
    st.header("Text and Speed")
    text_input = st.text_area("Text to Synthesize",
                            value="This is the demo of voice cloning and this is cloned voice.",
                            height=100)
    speed = st.slider("Speech Speed", min_value=0.5, max_value=2.0, value=1.0, step=0.1)
    generate_button = st.button("Generate Cloned Voice", disabled=st.session_state.processing)

    # Generate audio
    if generate_button and st.session_state.recorded_audio and text_input and st.session_state.target_se is not None:
        st.session_state.processing = True
        start_time = time.time()
        progress_bar = st.progress(0)

        try:
            # Validate speaker
            speaker_key = "EN_INDIA"
            speaker_ids = tts_model.hps.data.spk2id
            if speaker_key not in speaker_ids:
                st.error(f"Speaker '{speaker_key}' not found. Available: {list(speaker_ids.keys())}")
                return
            speaker_id = speaker_ids[speaker_key]

            # Generate intermediate audio
            progress_bar.progress(0.4, "Generating base audio...")
            t1 = time.time()
            tts_model.tts_to_file(text_input, speaker_id, src_path, speed=speed)
            st.write(f"TTS generation time: {time.time() - t1:.2f} seconds")

            # Convert tone color
            progress_bar.progress(0.8, "Applying voice cloning...")
            t2 = time.time()
            tone_color_converter.convert(
                audio_src_path=src_path,
                src_se=source_se,
                tgt_se=st.session_state.target_se,
                output_path=save_path,
                message="@MyShell"
            )
            st.write(f"Tone conversion time: {time.time() - t2:.2f} seconds")

            progress_bar.progress(1.0, "Complete!")
            st.subheader("Generated Cloned Voice")
            st.audio(save_path, format="audio/wav")
            st.success(f"Voice cloned audio generated in {time.time() - start_time:.2f} seconds!")

        except Exception as e:
            st.error(f"Error during generation: {str(e)}")
        finally:
            st.session_state.processing = False
            progress_bar.empty()

    elif generate_button:
        if not st.session_state.recorded_audio:
            st.warning("Please record a reference voice.")
        if not text_input:
            st.warning("Please enter text to synthesize.")
        if st.session_state.target_se is None and st.session_state.recorded_audio:
            st.warning("Speaker embedding is still processing. Please wait.")

if __name__ == "__main__":
    main()