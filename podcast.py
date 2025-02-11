import os
import logging
import streamlit as st
from openai import AzureOpenAI
import azure.cognitiveservices.speech as speechsdk
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
import docx2txt
import PyPDF2
from bs4 import BeautifulSoup
import re
import wave
import io
from pydub import AudioSegment
import warnings
warnings.filterwarnings("ignore")  # Suppress all warnings
import logging
logging.getLogger("azure").setLevel(logging.ERROR)  # Hide Azure internal logs

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Configuration Azure OpenAI
openai_endpoint = os.getenv("OPENAI_ENDPOINT")
openai_key = os.getenv("OPENAI_KEY")
api_version = os.getenv("OPENAI_API_VERSION")
model = os.getenv("OPENAI_MODEL")

# Initialize Azure OpenAI client
azure_openai_client = AzureOpenAI(
    api_key=openai_key,
    api_version=api_version,
    azure_endpoint=openai_endpoint
)

# Configuration Azure Speech Service
speech_key = os.getenv("SPEECH_KEY")
speech_region = os.getenv("SPEECH_REGION")

# Initialize Azure Speech Service
speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)

# Configuration Azure Document Intelligence
document_intelligence_key = os.getenv("DOCUMENT_INTELLIGENCE_KEY")
document_intelligence_endpoint = os.getenv("DOCUMENT_INTELLIGENCE_ENDPOINT")

# Initialize Azure Document Intelligence client
document_analysis_client = DocumentAnalysisClient(
    endpoint=document_intelligence_endpoint,
    credential=AzureKeyCredential(document_intelligence_key)
)

def extract_text_from_pdf(file_path):
    """
    Extract text content from a PDF file.
    """
    try:
        with open(file_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page_num in range(len(reader.pages)):
                text += reader.pages[page_num].extract_text()
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        return None

def extract_text_from_docx(file_path):
    """
    Extract text content from a DOCX file.
    """
    try:
        text = docx2txt.process(file_path)
        return text
    except Exception as e:
        logger.error(f"Error extracting text from DOCX: {e}")
        return None

def generate_podcast_script(content):
    """
    Use Azure OpenAI to generate a podcast script with emotional cues.
    """
    try:
        logger.info("Generating podcast script using Azure OpenAI...")
        response = azure_openai_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Take a deep breath,you are a podcast script writer that creates podcast scripts to be read by AI voices WITH THIS STRUCTURE (intro summary, then a deep dive, a conclusion). "
                        "Write a casual feeling, expressive conversation between two speakers (starting with Speaker 1 who is female) "
                        "and include natural interruptions between the speakers using -- for cutoffs. Example:\n"
                        "Speaker 1: Hey there! [excited] I've been thinking about this topic [thoughtful] and it really makes me wonder--\n"
                        "Speaker 2: [interrupting] Oh my gosh, yes! [enthusiastic] I was just about to say the same thing!\n\n"
                        f"Create a dynamic conversation about THE ENTIRETY OF THE CONTENT: {content}\n"
                        "Requirements:\n"
                        "1. Use Speaker 1 (female) and Speaker 2 (male) for the dialogue, DON'T ever mention their names when a speaker is talking to the other in their dialogue and the first speaker must say hey there! welcome to the iCSU Podcast.\n"
                        "2. Add emotional cues WITHIN sentences to show mood changes, e.g.:\n"
                        "   - Speaker 1: [excited] I started reading this book and [thoughtful] it really changed my perspective on--\n"
                        "   - Speaker 2: You know, [curious] I've been wondering about that [enthusiastic] especially when it comes to...\n"
                        "3. Include natural and dynamic interruptions using -- for cutoffs\n"
                        "4. Use varied emotions: excited, happy, curious, thoughtful, serious, emphatic, agreeing, explaining, pondering, passionate, skeptical\n"
                        "5. Add pause indicators with [...] for dramatic effect or topic transitions\n"
                        "6. Keep the conversation natural with realistic reactions and tone shifts\n"
                        "7. Use the following structure: (intro summary, then a deep dive, a conclusion)\n"
                        "8. Speaker 1 (female) must speak first\n"
                        "9. DO NOT WRITE THIS Podcast Script: Welcome to the MTTs Podcast."
                    )
                }
            ],
            max_tokens=3000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error generating podcast script: {e}")
        return None

def sanitize_ssml(ssml):
    """Parse and sanitize SSML to remove stray text nodes and guarantee no nested <voice> tags."""
    try:
        soup = BeautifulSoup(ssml, 'xml')
        speak = soup.find('speak')
        if speak:
            # Remove stray text nodes.
            for child in list(speak.contents):
                if child.name is None and not str(child).strip():
                    child.extract()
                elif child.name is None:
                    child.extract()
            # Ensure no <voice> tag is nested within another <voice> tag.
            while True:
                nested_voice = soup.find(lambda tag: tag.name=='voice' and tag.parent and tag.parent.name=='voice')
                if not nested_voice:
                    break
                # Replace nested voice with its inner text.
                nested_voice.replace_with(nested_voice.get_text())
            return str(soup)
        return ssml
    except Exception as e:
        logger.error(f"Error sanitizing SSML: {e}")
        return ssml

def generate_ssml_from_script(script):
    try:
        logger.info("Generating SSML from podcast script using Azure OpenAI...")
        # Updated system prompt: fixed voice names and added mark support for interruptions.
        system_prompt = """Take a deep breath,You are an SSML Expert specializing in taking podcast scripts and turning them into appropriate ssml format and microscopically subtle speech variations.
        
Rules for creating ultra-natural voice expressions:
1. Voice Selection:
   - Speaker 1: <voice name="en-US-AvaMultilingualNeural"> <!-- Most natural female voice -->
   - Speaker 2: <voice name="en-US-AndrewMultilingualNeural"> <!-- Most natural male voice -->
   
2. Micro-Subtle Emotion Mapping:
   [excited]:
     <mstts:express-as style="chat" styledegree="1.05">
       <prosody rate="2%" pitch="2%">text</prosody>
     </mstts:express-as>
   ... (other mappings)
3. Mark Support for Interruptions:
   Example:
   <voice name="en-US-AvaMultilingualNeural">
     <prosody rate="2%">I think that--<mark name="interrupt"/></prosody>
   </voice>
   <break time="5ms"/>
   <voice name="en-US-AndrewMultilingualNeural">
     <prosody rate="10%">Exactly!</prosody>
   </voice>"""
        
        response = azure_openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Convert this script to ultra-natural SSML with barely perceptible emotional shifts:\n{script}"}
            ],
            max_tokens=4000,
            temperature=0.2
        )
        
        raw_ssml = response.choices[0].message.content.strip()
        if not raw_ssml.startswith('<?xml'):
            raw_ssml = ('<?xml version="1.0" encoding="UTF-8"?>\n'
                        '<speak version="1.0" '
                        'xmlns="http://www.w3.org/2001/10/synthesis" '
                        'xmlns:mstts="http://www.w3.org/2001/mstts" '
                        'xml:lang="en-US">\n' + raw_ssml)
        if not raw_ssml.endswith('</speak>'):
            raw_ssml += '\n</speak>'
        # Sanitize SSML to remove stray text nodes.
        cleaned_ssml = sanitize_ssml(raw_ssml)
        return cleaned_ssml
    except Exception as e:
        logger.error(f"Error generating SSML: {e}")
        return None

def validate_voice_tags(ssml):
    """Validate SSML structure and Azure-supported features"""
    try:
        soup = BeautifulSoup(ssml, 'xml')
        if not soup.find('speak'):
            logger.error("Missing speak tag")
            return False
        valid_voices = {"en-US-AvaMultilingualNeural", "en-US-AndrewMultilingualNeural"}
        for voice in soup.find_all('voice'):
            if 'name' not in voice.attrs or voice['name'] not in valid_voices:
                logger.error(f"Invalid voice: {voice.get('name', '')}")
                return False
        # Updated valid styles to include additional supported styles.
        valid_styles = {
            "chat", "gentle", "newscast-casual", "friendly",
            "calm", "curious", "agreeable", "serious",
            "emphatic", "pondering", "passionate", "smiling", "concluding"
        }
        for express in soup.find_all('mstts:express-as'):
            if 'style' not in express.attrs or express['style'] not in valid_styles:
                logger.error(f"Unsupported style: {express.get('style', '')}")
                return False
        return True
    except Exception as e:
        logger.error(f"SSML validation failed: {e}")
        return False

def split_large_voice(voice, max_chunk_size):
    """Split a single voice element if its string length exceeds max_chunk_size."""
    # Build the opening tag with attributes.
    attrs = " ".join(f'{key}="{value}"' for key, value in voice.attrs.items())
    opening_tag = f"<voice {attrs}>" if attrs else "<voice>"
    closing_tag = "</voice>"
    overhead = len(opening_tag) + len(closing_tag)
    # Get the inner text.
    text = voice.get_text() or ""
    # Calculate available size for text.
    available_size = max_chunk_size - overhead
    parts = []
    for i in range(0, len(text), available_size):
        part_text = text[i:i+available_size]
        parts.append(f"{opening_tag}{part_text}{closing_tag}")
    return parts

def split_ssml_into_chunks(ssml, max_chunk_size=2000):
    """Split SSML by grouping complete <voice> elements using BeautifulSoup."""
    try:
        soup = BeautifulSoup(ssml, 'xml')
        voices = soup.find_all('voice')
        chunks = []
        current_content = ""
        for voice in voices:
            voice_str = str(voice)
            if len(current_content) + len(voice_str) > max_chunk_size and current_content:
                chunk = (
                    '<?xml version="1.0" encoding="UTF-8"?>\n'
                    '<speak version="1.0" '
                    'xmlns="http://www.w3.org/2001/10/synthesis" '
                    'xmlns:mstts="http://www.w3.org/2001/mstts" '
                    'xml:lang="en-US">\n' +
                    current_content +
                    '\n</speak>'
                )
                chunks.append(chunk)
                current_content = voice_str
            else:
                current_content += voice_str
        if current_content:
            chunk = (
                '<?xml version="1.0" encoding="UTF-8"?>\n'
                '<speak version="1.0" '
                'xmlns="http://www.w3.org/2001/10/synthesis" '
                'xmlns:mstts="http://www.w3.org/2001/mstts" '
                'xml:lang="en-US">\n' +
                current_content +
                '\n</speak>'
            )
            chunks.append(chunk)
        return chunks
    except Exception as e:
        logger.error(f"Improved chunking failed: {e}")
        return [ssml]

def create_ssml_chunk(voices):
    """Create properly formatted SSML chunk with validation."""
    voices_str = '\n'.join(str(voice) for voice in voices)
    chunk_ssml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<speak version="1.0" '
        'xmlns="http://www.w3.org/2001/10/synthesis" '
        'xmlns:mstts="http://www.w3.org/2001/mstts" '
        'xml:lang="en-US">\n'
        f'{voices_str}\n'
        '</speak>'
    )
    return chunk_ssml

def combine_wav_files(wav_data_list):
    """Combine multiple WAV data chunks into a single WAV file."""
    try:
        combined = AudioSegment.empty()
        for wav_data in wav_data_list:
            audio_segment = AudioSegment.from_file(io.BytesIO(wav_data), format="wav")
            combined += audio_segment
        output = io.BytesIO()
        combined.export(output, format="wav")
        return output.getvalue()
    except Exception as e:
        logger.error(f"Error combining WAV files: {e}")
        return None

def synthesize_ssml_to_audio(ssml, output_file, max_retries=3):
    """Use Azure Speech Service to synthesize SSML into an audio file without playback."""
    try:
        logger.info("Synthesizing SSML to audio...")
        if not validate_voice_tags(ssml):
            logger.error("Invalid SSML detected")
            return None
        
        speech_config.set_property(speechsdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs, "30000")
        speech_config.set_property(speechsdk.PropertyId.Speech_SegmentationSilenceTimeoutMs, "200")
        speech_config.set_speech_synthesis_output_format(speechsdk.SpeechSynthesisOutputFormat.Riff16Khz16BitMonoPcm)
        
        ssml_chunks = split_ssml_into_chunks(ssml, max_chunk_size=1500)
        logger.info(f"Split SSML into {len(ssml_chunks)} chunks")
        wav_data_list = []
        
        import time, contextlib, io
        # Define a dummy callback class that discards data.
        class DummyCallback:
            def write(self, data):
                pass

        # Update create_synthesizer to use DummyCallback.
        def create_synthesizer():
            dummy_stream = speechsdk.audio.PushAudioOutputStream(DummyCallback())
            audio_config = speechsdk.audio.AudioOutputConfig(stream=dummy_stream)
            return speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
        
        for i, chunk in enumerate(ssml_chunks, 1):
            chunk = chunk.strip()
            if not chunk:
                continue
            logger.info(f"Processing chunk {i} (size: {len(chunk)})")
            success = False
            retry_count = 0
            while not success and retry_count < max_retries:
                try:
                    synthesizer = create_synthesizer()
                    with contextlib.redirect_stderr(io.StringIO()):
                        result = synthesizer.speak_ssml_async(chunk).get()
                    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                        wav_data_list.append(result.audio_data)
                        success = True
                        logger.info(f"Chunk {i} synthesized successfully (no playback)")
                    else:
                        logger.warning(f"Chunk {i} failed (attempt {retry_count+1})")
                        if result.cancellation_details:
                            logger.error(f"Error: {result.cancellation_details.reason}")
                            logger.error(f"Error details: {result.cancellation_details.error_details}")
                        retry_count += 1
                        time.sleep(2 ** retry_count)
                except Exception as e:
                    logger.error(f"Chunk {i} exception: {str(e)}")
                    retry_count += 1
                    time.sleep(2 ** retry_count)
            if not success:
                logger.error(f"Failed chunk {i} content:\n{chunk[:500]}...")
                return None
        
        combined_audio = combine_wav_files(wav_data_list)
        if combined_audio:
            with open(output_file, 'wb') as audio_file:
                audio_file.write(combined_audio)
            logger.info(f"Audio saved to {output_file}")
            return output_file
        logger.error("No audio data generated")
        return None
    except Exception as e:
        logger.error(f"Error synthesizing audio: {e}")
        return None

def stitch_audio_with_intro(intro_file, main_audio_file, output_file):
    """
    Stitch the intro.wav file with the main audio file.
    """
    try:
        # Load the intro and main audio files using pydub
        intro_audio = AudioSegment.from_wav(intro_file)
        main_audio = AudioSegment.from_wav(main_audio_file)

        # Combine the audio files
        final_audio = intro_audio + main_audio

        # Export the final audio to a new file
        final_audio.export(output_file, format="wav")
        logger.info(f"Final audio saved to {output_file}")
        return output_file
    except Exception as e:
        logger.error(f"Error stitching audio files: {e}")
        return None

def main():
    """
    Main workflow to generate a podcast script, convert it to SSML, synthesize it into audio,
    and stitch it with an intro.wav file.
    """
    st.title("Podcast Generator")
    st.write("Upload a PDF or DOCX file to generate a podcast script and audio.")

    if "audio_generated" not in st.session_state:
        st.session_state.audio_generated = False

    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx"])
    if uploaded_file is not None and not st.session_state.audio_generated:
        file_name = os.path.splitext(uploaded_file.name)[0]
        if uploaded_file.name.endswith(".pdf"):
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            subject = extract_text_from_pdf("temp.pdf")
        elif uploaded_file.name.endswith(".docx"):
            with open("temp.docx", "wb") as f:
                f.write(uploaded_file.getbuffer())
            subject = extract_text_from_docx("temp.docx")
        else:
            st.error("Unsupported file format. Please upload a PDF or DOCX file.")
            return

        if not subject:
            st.error("Failed to extract text from the file. Exiting.")
            return

        # Step 2: Generate the podcast script
        st.write("Generating podcast script...")
        podcast_script = generate_podcast_script(subject)
        if not podcast_script:
            st.error("Failed to generate podcast script. Exiting.")
            return
        st.write("Podcast Script:\n", podcast_script)

        # Step 3: Generate SSML from the script
        st.write("Generating SSML from script...")
        ssml = generate_ssml_from_script(podcast_script)
        if not ssml:
            st.error("Failed to generate SSML from script. Exiting.")
            return
        st.write("SSML Content:\n", ssml)
        
        # Pre-synthesis validation for debugging.
        if not validate_voice_tags(ssml):
            st.error("Generated SSML failed validation. Please check the logs.")
            logger.error(f"Invalid SSML content:\n{ssml}")
            return

        # Step 4: Synthesize the SSML to audio
        st.write("Synthesizing SSML to audio...")
        generated_audio_file = f"{file_name}_generated.wav"
        audio_file = synthesize_ssml_to_audio(ssml, generated_audio_file)
        if not audio_file:
            st.error("Failed to synthesize audio. Exiting.")
            return

        # Step 5: Stitch the intro.wav with the generated audio
        st.write("Stitching intro with generated audio...")
        intro_file = "intro.wav"  # Ensure intro.wav is in the same directory
        final_audio_file = f"{file_name}_final.wav"
        final_audio = stitch_audio_with_intro(intro_file, generated_audio_file, final_audio_file)
        if not final_audio:
            st.error("Failed to stitch audio files. Exiting.")
            return

        # Step 6: Provide the final audio for download
        with open(final_audio_file, "rb") as f:
            st.download_button(
                label="Download Final Podcast Audio",
                data=f.read(),
                file_name=f"{file_name}_Podcast_episode.wav",
                mime="audio/wav"
            )
        st.session_state.audio_generated = True
        st.write("Podcast generation workflow completed successfully.")

if __name__ == "__main__":
    main()
