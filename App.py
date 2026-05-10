"""
Storytelling Application for Kids (Ages 3-10)
This application generates a short story from an uploaded image and converts it to speech.
"""

import streamlit as st
from PIL import Image
from transformers import pipeline
import gtts
from io import BytesIO
import base64
import requests
from datetime import datetime

# ------------------ CONFIGURATION ------------------
st.set_page_config(
    page_title="StoryTeller for Kids",
    page_icon="📖",
    layout="centered"
)

# ------------------ CACHING MODELS ------------------
@st.cache_resource
def load_captioning_model():
    """Load the image captioning model from Hugging Face"""
    try:
        # Using BLIP for image captioning
        captioner = pipeline(
            "image-to-text",
            model="Salesforce/blip-image-captioning-base"
        )
        return captioner
    except Exception as e:
        st.error(f"Failed to load captioning model: {e}")
        return None

@st.cache_resource
def load_text_generation_model():
    """Load a kid-friendly text generation model"""
    try:
        # Using DistilGPT-2 which is lightweight and suitable for short stories
        generator = pipeline(
            "text-generation",
            model="distilgpt2"
        )
        return generator
    except Exception as e:
        st.error(f"Failed to load text generation model: {e}")
        return None

# ------------------ STORY GENERATION ------------------
def generate_caption(captioner, image):
    """Generate a caption from the uploaded image"""
    if captioner is None:
        return "A happy scene with animals playing together."
    
    try:
        result = captioner(image)
        caption = result[0]['generated_text']
        return caption
    except Exception as e:
        st.warning(f"Caption generation failed: {e}. Using fallback caption.")
        return "A cheerful outdoor scene with friendly animals."

def generate_story(generator, caption):
    """Generate a 50-100 word story from the caption"""
    if generator is None:
        # Fallback story template when model unavailable
        return f"""
        Once upon a time, {caption.lower()}. 
        They were all very happy and decided to have an adventure together. 
        They explored new places and made wonderful memories. 
        At the end of the day, they realized that friendship and kindness 
        make every day special. And they lived happily ever after!
        """
    
    # Craft prompt for kid-friendly story
    prompt = f"""Write a short, happy, and magical story for young children (50-100 words) about: {caption}

Story: Once upon a time,"""
    
    try:
        result = generator(
            prompt,
            max_length=150,
            min_length=50,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            pad_token_id=50256,
            truncation=True
        )
        
        full_story = result[0]['generated_text']
        
        # Clean up and ensure proper story formatting
        story = full_story.replace(prompt, "").strip()
        
        # Ensure story starts properly
        if not story.startswith("Once upon a time,"):
            story = "Once upon a time, " + story
            
        # Limit to 100 words
        words = story.split()[:120]  # Slightly more to account for punctuation
        story = " ".join(words)
        
        # Add happy ending if missing
        if "happily ever after" not in story.lower() and "the end" not in story.lower():
            story += " And they all lived happily ever after!"
            
        return story
        
    except Exception as e:
        st.warning(f"Story generation error: {e}. Using fallback story.")
        return f"""
        Once upon a time, {caption.lower()}. 
        They played together in the sunshine and discovered a magical secret. 
        They learned that being kind and helping friends is the greatest 
        adventure of all. The End.
        """

# ------------------ TEXT TO SPEECH ------------------
def text_to_speech(text):
    """Convert text to speech using gTTS and return audio bytes"""
    try:
        tts = gtts.gTTS(text=text, lang='en', slow=False)
        audio_bytes = BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        return audio_bytes
    except Exception as e:
        st.error(f"Text-to-speech conversion failed: {e}")
        return None

def get_audio_player(audio_bytes):
    """Generate HTML audio player for the generated speech"""
    if audio_bytes is None:
        return None
    
    audio_base64 = base64.b64encode(audio_bytes.read()).decode()
    audio_html = f"""
    <audio controls autoplay style="width: 100%;">
        <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
        Your browser does not support the audio element.
    </audio>
    """
    return audio_html

# ------------------ MAIN APPLICATION ------------------
def main():
    # Title and description
    st.title("📖 StoryTeller for Little Dreamers")
    st.markdown("""
    <div style="background-color: #f0f8ff; padding: 1rem; border-radius: 10px;">
        <p style="font-size: 1.2rem;">✨ Upload a picture, and I'll tell you a magical story! ✨</p>
        <p>For kids aged 3-10 • Stories are 50-100 words • Includes audio!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for instructions
    with st.sidebar:
        st.header("🎨 How to Play")
        st.markdown("""
        1. **Upload a picture** of anything you like!
           - Animals 🦊
           - Nature 🌳
           - Toys 🧸
           - Friends 👫
        
        2. **Click "Tell Me a Story!"**
        
        3. **Listen** to your special story! 🎧
        
        4. **Share** with your friends and family!
        """)
        
        st.divider()
        st.caption("Made with ❤️ for young storytellers")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "📸 Choose an image...",
        type=['jpg', 'jpeg', 'png', 'gif'],
        help="Upload a colorful picture to start your story adventure!"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        
        # Resize image for display
        max_width = 500
        if image.width > max_width:
            ratio = max_width / image.width
            new_size = (max_width, int(image.height * ratio))
            image_display = image.resize(new_size)
        else:
            image_display = image
        
        st.image(image_display, caption="Your Magical Picture", use_container_width=True)
        
        # Generate Story Button
        if st.button("✨ Tell Me a Story! ✨", type="primary"):
            with st.spinner("🧠 Looking at your picture..."):
                captioner = load_captioning_model()
                caption = generate_caption(captioner, image)
                st.success(f"📷 I see: *{caption}*")
            
            with st.spinner("📖 Writing your story..."):
                generator = load_text_generation_model()
                story = generate_story(generator, caption)
            
            # Display story
            st.subheader("📚 Your Story")
            story_box = st.container()
            with story_box:
                st.markdown(f"""
                <div style="background-color: #fff9c4; padding: 1.5rem; border-radius: 15px; font-size: 1.1rem; line-height: 1.6; border-left: 5px solid #ff9800;">
                    {story}
                </div>
                """, unsafe_allow_html=True)
            
            # Generate and play audio
            with st.spinner("🔊 Creating audio for you..."):
                audio_bytes = text_to_speech(story)
            
            if audio_bytes:
                st.subheader("🎧 Listen to Your Story")
                audio_player = get_audio_player(audio_bytes)
                st.markdown(audio_player, unsafe_allow_html=True)
                
                # Download button for audio
                audio_bytes.seek(0)
                st.download_button(
                    label="💾 Download Story Audio",
                    data=audio_bytes,
                    file_name=f"story_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3",
                    mime="audio/mp3"
                )
            
            # Celebration message
            st.balloons()
            st.success("🎉 Your story is ready! Click play to listen or read aloud! 🎉")
    
    else:
        # Show example prompt when no image uploaded
        st.info("👆 **Start by uploading a picture from your computer!**")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("🐶 **Animals**\n\nA dog playing in the park")
        with col2:
            st.markdown("🏰 **Magic**\n\nA princess in a castle")
        with col3:
            st.markdown("🚀 **Adventure**\n\nAn astronaut on the moon")
    
    # Footer
    st.divider()
    st.caption("""
    ⭐ **For best results:** Use clear, colorful images with recognizable objects or animals. 
    The story will be based on what the AI sees in your picture!
    """)

# ------------------ ENTRY POINT ------------------
if __name__ == "__main__":
    main()