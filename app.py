import streamlit as st
import os
import time
import tempfile
from PIL import Image
import io
import cv2
import supervision as sv
from inference_sdk import InferenceHTTPClient
from inference import get_model
import google.generativeai as genai
import speech_recognition as sr
from gtts import gTTS
import base64

# Configure API keys
ROBOFLOW_API_KEY = ""
GEMINI_API_KEY = ""  # Replace with your Gemini API key

# Initialize the Roboflow client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=ROBOFLOW_API_KEY
)

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Define available models
MODELS = {
    "brain tumor": {"id": "brain-tumor-m2pbp/1", "description": "Detects brain tumors in MRI scans"},
    "tuberculosis": {"id": "tuberculosis-detection-6e3dj/1", "description": "Identifies tuberculosis in chest X-rays"},
    "lung cancer": {"id": "cancer-lung/1", "description": "Detects lung cancer in CT scans"},
    "heart enlargement": {"id": "cardiomegaly-vgutp/1", "description": "Identifies cardiomegaly (enlarged heart) in chest X-rays"},
    "bone fracture": {"id": "bone-thybx/1", "description": "Detects fractures in bone X-rays"},
    "kidney stone": {"id": "kidney-stone-detection-itqje/1", "description": "Identifies kidney stones in CT scans"},
    "soft tissue": {"id": "soft-tissue-classifier-a0uzk/1", "description": "Classifies soft tissue conditions"},
    "pox": {"id": "poxclassification/1", "description": "Identifies pox-related skin conditions"},
    "cancer": {"id": "multiple-cancers-segmentation/2", "description": "Segments various cancer types in medical images"},
    "eye condition": {"id": "se_mongta/2", "description": "Detects various eye conditions"}
}

# Set up page config
st.set_page_config(
    page_title="Medical Diagnostic Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to select model based on user prompt
def select_model(prompt):
    # Prepare a prompt for Gemini to analyze the user's description
    analysis_prompt = f"""
    Based on the following patient's description, determine which medical imaging model would be most appropriate.
    
    Patient description: "{prompt}"
    
    Available models:
    - Brain tumor detection (for brain-related issues, headaches, neurological symptoms)
    - Tuberculosis detection (for persistent cough, chest pain, fatigue)
    - Lung cancer detection (for chronic cough, chest pain, shortness of breath)
    - Heart enlargement detection (for chest pain, shortness of breath, swelling in legs)
    - Bone fracture detection (for pain, swelling, deformity in bones)
    - Kidney stone detection (for severe pain in side and back, painful urination, blood in urine)
    - Soft tissue classification (for lumps, bumps, swelling in soft tissues)
    - Pox classification (for rash, skin lesions, blisters)
    - Multiple cancer segmentation (for various cancer-related symptoms)
    - Eye condition detection (for vision problems, eye pain, redness)
    
    Return ONLY the name of the most relevant model in lowercase (e.g., "brain tumor", "lung cancer", etc.).
    """
    
    # Use correct model name for Gemini
    model = genai.GenerativeModel('gemini-2.0-flash')
    response = model.generate_content(analysis_prompt)
    selected_model = response.text.strip().lower()
    
    # Check if the selected model exists in our MODELS dictionary
    for model_name in MODELS.keys():
        if model_name in selected_model:
            return model_name
    
    # Default to multiple cancer segmentation if no match is found
    return "cancer"

def image_to_base64(image_bytes):
    return base64.b64encode(image_bytes).decode('utf-8')

def run_inference(image_bytes, model_id):
    try:
        # Convert image bytes to numpy array for OpenCV processing
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Use Roboflow inference with the supervision library
        model = get_model(model_id=model_id)
        results = model.infer(image)
        
        # Convert results to supervision Detections
        detections = sv.Detections.from_inference(results[0].dict(by_alias=True, exclude_none=True))
        
        # Create supervision annotators for visualization
        bounding_box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()
        
        # Annotate the image with our inference results
        annotated_image = bounding_box_annotator.annotate(scene=image, detections=detections)
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)
        
        # Convert the annotated image back to bytes for display
        is_success, buffer = cv2.imencode(".jpg", annotated_image)
        annotated_bytes = io.BytesIO(buffer)
        
        return {
            "detections": detections.to_dict(),
            "annotated_image": annotated_bytes,
            "raw_results": results[0].dict(by_alias=True, exclude_none=True)
        }
    except Exception as e:
        # st.error(f"Error running inference: {str(e)}")
        pass
        # Fallback to the original inference method
        try:
            result = CLIENT.infer(image_bytes, model_id=model_id)
            return result
        except Exception as e2:
            # st.error(f"Fallback inference also failed: {str(e2)}")
            pass
            return None

# Function to generate lifestyle recommendations
def get_recommendations(condition, result):
    recommendation_prompt = f"""
    Based on the following medical condition and detection results, provide:
    1. A brief explanation of the condition in simple terms
    2. Three lifestyle recommendations that could help the patient
    3. Two questions to better understand the patient's current lifestyle
    
    Condition: {condition}
    Detection results: {result}
    
    Keep your response concise and accessible to non-medical professionals.
    """
    
    # Use correct model name for Gemini
    model = genai.GenerativeModel('gemini-2.0-flash')
    response = model.generate_content(recommendation_prompt)
    return response.text

# Function to convert speech to text
def speech_to_text():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Listening... Speak now")
        audio = r.listen(source)
        st.write("Processing your speech...")
    
    try:
        text = r.recognize_google(audio)
        return text
    except Exception as e:
        st.error(f"Error processing speech: {str(e)}")
        return None

# Function to convert text to speech
def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    fp = tempfile.NamedTemporaryFile(delete=False)
    tts.save(fp.name)
    return fp.name

# Main UI layout
def main():
    st.title("🏥 Medical Diagnostic Assistant")
    
    # Sidebar for app info
    with st.sidebar:
        st.header("About")
        st.info("This application uses AI to analyze medical images and provide preliminary insights. It is not a substitute for professional medical advice.")
        st.subheader("How it works")
        st.write("1. Describe your symptoms")
        st.write("2. Upload a relevant medical image")
        st.write("3. Review the AI analysis")
        st.write("4. Get lifestyle recommendations")
        st.write("5. Have a voice conversation for more personalized advice")
        
        st.subheader("Available Models")
        for name, details in MODELS.items():
            st.write(f"**{name.title()}**: {details['description']}")
    
    # Initialize session state variables if they don't exist
    if 'step' not in st.session_state:
        st.session_state.step = 1
    
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = None
    
    if 'result' not in st.session_state:
        st.session_state.result = None
    
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = None
        
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    
    # Step 1: Get user symptoms
    if st.session_state.step == 1:
        st.header("Step 1: Describe Your Symptoms")
        st.write("Please describe your symptoms or medical concerns in detail.")
        
        user_prompt = st.text_area("Symptoms description", height=150, 
                                   placeholder="Example: I've been experiencing severe headaches and blurry vision for the past two weeks...")
        
        if st.button("Analyze Symptoms", type="primary"):
            if user_prompt:
                with st.spinner("Analyzing your symptoms..."):
                    try:
                        selected_model = select_model(user_prompt)
                        st.session_state.selected_model = selected_model
                        st.session_state.step = 2
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error analyzing symptoms: {str(e)}")
            else:
                st.warning("Please describe your symptoms first.")
    
    # Step 2: Upload image
    elif st.session_state.step == 2:
        st.header(f"Step 2: Upload Medical Image for {st.session_state.selected_model.title()} Analysis")
        st.write(f"Based on your description, we recommend using our {st.session_state.selected_model.title()} detection model.")
        st.write(MODELS[st.session_state.selected_model]["description"])
        
        uploaded_file = st.file_uploader("Upload your medical image", type=["jpg", "jpeg", "png", "dicom", "dcm"])
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Back", type="secondary"):
                st.session_state.step = 1
                st.rerun()
        
        with col2:
            if st.button("Analyze Image", type="primary", disabled=not uploaded_file):
                if uploaded_file:
                    with st.spinner("Processing your image..."):
                        try:
                            # Convert uploaded file to bytes
                            image_bytes = uploaded_file.getvalue()
                            
                            # Display the uploaded image
                            st.image(image_bytes, caption="Uploaded Image", width=300)
                            
                            # Run inference
                            result = run_inference(image_bytes, MODELS[st.session_state.selected_model]["id"])
                            st.session_state.result = result
                            
                            # If we have an annotated image, display it
                            if result and "annotated_image" in result:
                                st.image(result["annotated_image"], caption="Analysis Result", width=300)
                            
                            # Generate recommendations
                            recommendations = get_recommendations(st.session_state.selected_model, result)
                            st.session_state.recommendations = recommendations
                            
                            st.session_state.step = 3
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error processing image: {str(e)}")
    
    # Step 3: Results and recommendations
    elif st.session_state.step == 3:
        st.header("Step 3: Results & Recommendations")
        
        # Display results tab
        tab2, tab3 = st.tabs([ "Recommendations", "Voice Conversation"])
        
        # with tab1:
        #     st.subheader(f"{st.session_state.selected_model.title()} Analysis Results")
            
        #     # If we have an annotated image, display it
        #     if st.session_state.result and "annotated_image" in st.session_state.result:
        #         st.image(st.session_state.result["annotated_image"], caption="Annotated Image", width=600)
            
        #     st.json(st.session_state.result)
        
        with tab2:
            st.subheader("Lifestyle Recommendations")
            st.markdown(st.session_state.recommendations)
        
        with tab3:
            st.subheader("Voice Conversation")
            st.write("You can have a conversation with our AI to get more personalized recommendations.")
            
            # Display conversation history
            for message in st.session_state.conversation_history:
                if message["role"] == "user":
                    st.write(f"**You:** {message['content']}")
                else:
                    st.write(f"**Assistant:** {message['content']}")
            
            # Option to type or speak
            input_method = st.radio("Choose input method:", ["Type", "Speak"])
            
            if input_method == "Type":
                user_input = st.text_input("Your question or response:")
                if st.button("Send", type="primary"):
                    if user_input:
                        st.session_state.conversation_history.append({"role": "user", "content": user_input})
                        
                        try:
                            # Generate response
                            context = f"Medical condition: {st.session_state.selected_model}\nResults: {st.session_state.result}\n"
                            history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.conversation_history])
                            
                            response_prompt = f"""
                            You are a medical assistant providing lifestyle advice based on medical imaging results.
                            
                            {context}
                            
                            Conversation history:
                            {history}
                            
                            Provide a helpful, concise response focused on lifestyle recommendations and general wellness advice.
                            Do NOT provide specific medical advice or diagnosis. Emphasize the importance of consulting healthcare professionals.
                            """
                            
                            # Use correct model name for Gemini
                            model = genai.GenerativeModel('gemini-2.0-flash')
                            response = model.generate_content(response_prompt)
                            
                            st.session_state.conversation_history.append({"role": "assistant", "content": response.text})
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error generating response: {str(e)}")
            
            else:  # Speech input
                if st.button("Start Speaking"):
                    speech_input = speech_to_text()
                    if speech_input:
                        st.write(f"You said: {speech_input}")
                        st.session_state.conversation_history.append({"role": "user", "content": speech_input})
                        
                        try:
                            # Generate response
                            context = f"Medical condition: {st.session_state.selected_model}\nResults: {st.session_state.result}\n"
                            history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.conversation_history])
                            
                            response_prompt = f"""
                            You are a medical assistant providing lifestyle advice based on medical imaging results.
                            
                            {context}
                            
                            Conversation history:
                            {history}
                            
                            Provide a helpful, concise response focused on lifestyle recommendations and general wellness advice.
                            Keep your response brief and conversational, suitable for text-to-speech.
                            Do NOT provide specific medical advice or diagnosis. Emphasize the importance of consulting healthcare professionals.
                            """
                            
                            # Use correct model name for Gemini
                            model = genai.GenerativeModel('gemini-2.0-flash')
                            response = model.generate_content(response_prompt)
                            
                            # Save response
                            assistant_response = response.text
                            st.session_state.conversation_history.append({"role": "assistant", "content": assistant_response})
                            
                            # Convert to speech
                            speech_file = text_to_speech(assistant_response)
                            st.audio(speech_file)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error processing speech response: {str(e)}")
        
        # Navigation buttons
        if st.button("Start Over", type="secondary"):
            # Reset session state
            st.session_state.step = 1
            st.session_state.selected_model = None
            st.session_state.result = None
            st.session_state.recommendations = None
            st.session_state.conversation_history = []
            st.rerun()

if __name__ == "__main__":
    main()
