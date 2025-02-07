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
                        "1. Use Speaker 1 (female) and Speaker 2 (male) for the dialogue, DON'T ever mention their names when a speaker is talking to the other in their dialogue and the first speaker must say hey there! welcome to the MTTs Podcast.\n"
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

def generate_ssml_from_script(script):
    try:
        logger.info("Generating SSML from podcast script using Azure OpenAI...")
        response = azure_openai_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": """You are an SSML Expert specializing in microscopically subtle speech variations.
                    
                    Rules for creating ultra-natural voice expressions:
                    1. Voice Selection:
                       - Speaker 1: <voice name="en-US-AvaMultilingualNeural"> <!-- Most natural female voice -->
                       - Speaker 2: <voice name=" en-US-AndrewMultilingualNeural"> <!-- Most natural male voice -->
                    
                    2. Micro-Subtle Emotion Mapping:
                       [excited]: 
                         <mstts:express-as style="chat" styledegree="1.05">
                           <prosody rate="2%" pitch="2%">text</prosody>
                         </mstts:express-as>
                       
                       [thoughtful]:
                         <mstts:express-as style="gentle" styledegree="0.95">
                           <prosody rate="-1%" pitch="-1%">text</prosody>
                         </mstts:express-as>
                       
                       [curious]:
                         <mstts:express-as style="chat" styledegree="0.98">
                           <prosody pitch="1%">text</prosody>
                         </mstts:express-as>
                       
                       [serious]:
                         <mstts:express-as style="newscast-casual" styledegree="0.97">
                           <prosody pitch="-1%">text</prosody>
                         </mstts:express-as>
                       
                       [happy]:
                         <mstts:express-as style="friendly" styledegree="1.02">
                           <prosody pitch="1.5%">text</prosody>
                         </mstts:express-as>
                       
                       [interrupting]:
                         <break time="5ms"/>
                         <prosody rate="10%" contour="(0%,+5%) (10%,0%)">text</prosody>

                    3. Ultra-Natural Guidelines:
                       - Zero delay for interruptions
                       - Micro-pauses: <break time="30ms"/>
                       - Use <mark name="interrupt"/> at cut-off points
                       
                    4. Example of instant interruption:
                       <voice name="en-US-AvaMultilingualNeural">
                         I think that the main--<mark name="interrupt"/>
                       </voice>
                       <break time="5ms"/>
                       <voice name="en-US-AndrewMultilingualNeural">
                         <prosody rate="10%">Yes, exactly!</prosody>
                       </voice>"""
                },
                {
                    "role": "user",
                    "content": f"Convert this script to ultra-natural SSML with barely perceptible emotional shifts:\n{script}"
                }
            ],
            max_tokens=4000,
            temperature=0.2  # Lower temperature for more consistent subtle changes
        )
        
        raw_ssml = response.choices[0].message.content.strip()
        
        # Add required XML namespaces
        if not raw_ssml.startswith('<?xml'):
            raw_ssml = ('<?xml version="1.0" encoding="UTF-8"?>\n'
                       '<speak version="1.0" '
                       'xmlns="http://www.w3.org/2001/10/synthesis" '
                       'xmlns:mstts="http://www.w3.org/2001/mstts" '
                       'xml:lang="en-US">\n' + raw_ssml)

        if not raw_ssml.endswith('</speak>'):
            raw_ssml += '\n</speak>'

        return raw_ssml

    except Exception as e:
        logger.error(f"Error generating SSML: {e}")
        return None

def validate_voice_tags(ssml):
    """Ensure all voice tags are properly paired and no content is lost."""
    soup = BeautifulSoup(ssml, 'xml')
    voices = soup.find_all('voice')
    
    # Check for unclosed or malformed tags
    for voice in voices:
        if not voice.string and not voice.contents:
            logger.error(f"Empty voice tag found: {voice}")
            return False
    return True

def split_ssml_into_chunks(ssml, max_chunk_size=2000):
    """Split SSML into smaller chunks while preserving complete sentences and voice tags."""
    try:
        # Clean up any duplicate XML declarations or speak tags
        ssml = re.sub(r'<\?xml[^>]*\?>\s*', '', ssml)
        ssml = re.sub(r'<speak[^>]*>\s*', '', ssml)
        ssml = re.sub(r'</speak>\s*', '', ssml)
        
        # Parse the SSML content
        soup = BeautifulSoup(f"<root>{ssml}</root>", 'xml')
        voices = soup.find_all('voice')
        
        # Group voices into pairs (keep conversation exchanges together)
        pairs = []
        for i in range(0, len(voices), 2):
            if i + 1 < len(voices):
                pairs.append([voices[i], voices[i + 1]])
            else:
                pairs.append([voices[i]])
        
        # Create chunks from pairs
        chunks = []
        current_chunk = []
        current_length = 0
        
        for pair in pairs:
            pair_str = ''.join(str(v) for v in pair)
            pair_length = len(pair_str)
            
            if current_length + pair_length > max_chunk_size and current_chunk:
                chunk_ssml = create_ssml_chunk(current_chunk)
                if validate_voice_tags(chunk_ssml):
                    chunks.append(chunk_ssml)
                current_chunk = []
                current_length = 0
            
            current_chunk.extend(pair)
            current_length += pair_length
        
        # Add remaining voices
        if current_chunk:
            chunk_ssml = create_ssml_chunk(current_chunk)
            if validate_voice_tags(chunk_ssml):
                chunks.append(chunk_ssml)
        
        # Validate all chunks have content
        total_voices = sum(len(BeautifulSoup(chunk, 'xml').find_all('voice')) for chunk in chunks)
        if total_voices != len(voices):
            logger.error(f"Voice count mismatch: {total_voices} vs {len(voices)}")
            return [ssml]  # Return single chunk if validation fails
            
        return chunks

    except Exception as e:
        logger.error(f"Error splitting SSML: {e}")
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
    """Use Azure Speech Service to synthesize the SSML into an audio file."""
    try:
        logger.info("Synthesizing SSML to audio...")
        
        # Validate complete SSML before processing
        if not validate_voice_tags(ssml):
            logger.error("Invalid SSML detected")
            return None
            
        # Configure speech synthesizer
        speech_config.set_property(speechsdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs, "30000")
        speech_config.set_property(speechsdk.PropertyId.Speech_SegmentationSilenceTimeoutMs, "200")
        speech_config.set_speech_synthesis_output_format(speechsdk.SpeechSynthesisOutputFormat.Riff16Khz16BitMonoPcm)
        
        audio_config = speechsdk.audio.AudioOutputConfig(filename=output_file)
        synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=speech_config, 
            audio_config=audio_config
        )
        
        # Split SSML into smaller chunks
        ssml_chunks = split_ssml_into_chunks(ssml, max_chunk_size=2000)  # Reduced chunk size
        logger.info(f"Split SSML into {len(ssml_chunks)} chunks")
        wav_data_list = []
        failed_chunks = []
        
        # First pass: Process all chunks
        for i, chunk in enumerate(ssml_chunks, 1):
            chunk_size = len(chunk)
            logger.info(f"Processing chunk {i}/{len(ssml_chunks)} (size: {chunk_size} bytes)")
            
            success = False
            retries = 0
            
            while not success and retries < max_retries:
                try:
                    result = synthesizer.speak_ssml_async(chunk).get()
                    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                        wav_data_list.append((i, result.audio_data))
                        success = True
                        logger.info(f"Successfully synthesized chunk {i}")
                    else:
                        retries += 1
                        logger.warning(f"Chunk {i} synthesis failed (attempt {retries}/{max_retries})")
                        if result.cancellation_details:
                            logger.error(f"Error details: {result.cancellation_details.reason}")
                        else:
                            logger.error(f"No error details provided for chunk {i}")
                except Exception as e:
                    retries += 1
                    logger.error(f"Error processing chunk {i} (attempt {retries}/{max_retries}): {e}")
                
            if not success:
                failed_chunks.append((i, chunk))
                logger.error(f"Failed to process chunk {i} after {max_retries} attempts")
        
        # Second pass: Retry failed chunks with smaller sizes
        if failed_chunks:
            logger.info(f"Attempting to process {len(failed_chunks)} failed chunks with smaller sizes")
            
            # Sort failed chunks by size (largest first)
            failed_chunks.sort(key=lambda x: len(x[1]), reverse=True)
            
            for chunk_id, chunk in failed_chunks:
                logger.info(f"Retrying failed chunk {chunk_id} (size: {len(chunk)} bytes)")
                
                # Split into increasingly smaller sub-chunks
                sub_chunk_sizes = [500, 250, 100]  # Define sub-chunk sizes
                
                for sub_chunk_size in sub_chunk_sizes:
                    logger.info(f"Attempting sub-chunk size: {sub_chunk_size} bytes")
                    sub_chunks = [chunk[i:i+sub_chunk_size] for i in range(0, len(chunk), sub_chunk_size)]
                    
                    for sub_i, sub_chunk in enumerate(sub_chunks, 1):
                        if not sub_chunk.strip():
                            logger.warning(f"Skipping empty sub-chunk {sub_i} of chunk {chunk_id}")
                            continue
                            
                        retries = 0
                        success = False
                        
                        while not success and retries < max_retries:
                            try:
                                result = synthesizer.speak_ssml_async(sub_chunk).get()
                                if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                                    wav_data_list.append((chunk_id + (sub_i / 1000.0), result.audio_data))  # Unique ID
                                    success = True
                                    logger.info(f"Successfully synthesized sub-chunk {sub_i} of failed chunk {chunk_id}")
                                else:
                                    retries += 1
                                    logger.warning(f"Sub-chunk {sub_i} of chunk {chunk_id} synthesis failed (attempt {retries}/{max_retries})")
                                    if result.cancellation_details:
                                        logger.error(f"Error details: {result.cancellation_details.reason}")
                                    else:
                                        logger.error(f"No error details provided for sub-chunk {sub_i} of chunk {chunk_id}")
                            except Exception as e:
                                retries += 1
                                logger.error(f"Error processing sub-chunk {sub_i} of chunk {chunk_id}: {e}")
        
        # Sort and combine audio data in correct order
        if wav_data_list:
            wav_data_list.sort(key=lambda x: x[0])  # Sort by chunk index
            combined_audio = combine_wav_files([data for _, data in wav_data_list])
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
