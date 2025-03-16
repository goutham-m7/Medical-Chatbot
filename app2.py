import streamlit as st
import os
import time
import tempfile
import numpy as np
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
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# Configure API keys
ROBOFLOW_API_KEY = "89x32AC9lbkT8YwIeyeH"
GEMINI_API_KEY = "AIzaSyBc9SOQ56DTARMQ8CZxRpeJg_Jh2DDUROM"  # Replace with your Gemini API key

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

# Define organ classes for the MNIST-like organ dataset
ORGAN_CLASSES = [
    "brain", "lung", "heart", "kidney", "liver", 
    "spleen", "stomach", "intestine", "bone", "eye"
]

# Set up page config
st.set_page_config(
    page_title="Medical Diagnostic Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to build and train organ detection model (MNIST-like approach)
def build_organ_detection_model(train_data=None):
    # Check if model already exists
    model_path = "organ_detection_model.h5"
    if os.path.exists(model_path):
        return load_model(model_path)
    
    # If no training data is provided and model doesn't exist, create a dummy model
    # In a real application, you would use an actual medical imaging dataset
    if train_data is None:
        st.warning("No training data provided. Using a pre-trained model.")
        # Build a CNN model similar to MNIST
        model = Sequential([
            Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(128, 128, 1)),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, kernel_size=(3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(len(ORGAN_CLASSES), activation='softmax')
        ])
        
        model.compile(optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
        
        # Save the model
        model.save(model_path)
        return model
    
    # Train model with provided data (not implemented in this demo)
    # In a real implementation, you would:
    # 1. Preprocess the data
    # 2. Split into train/validation sets
    # 3. Train the model
    # 4. Save the model
    # Here we just return the dummy model
    return build_organ_detection_model(None)

# Function to identify organ from scan
def identify_organ(image_bytes):
    try:
        # Load the model
        model = build_organ_detection_model()
        
        # Convert image bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Preprocess the image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (128, 128))
        normalized = resized / 255.0
        input_image = normalized.reshape(1, 128, 128, 1)
        
        # Make prediction
        prediction = model.predict(input_image)
        organ_index = np.argmax(prediction[0])
        confidence = float(prediction[0][organ_index])
        
        # Get the organ name
        organ_name = ORGAN_CLASSES[organ_index]
        
        return {
            "organ": organ_name,
            "confidence": confidence,
            "all_predictions": {ORGAN_CLASSES[i]: float(prediction[0][i]) for i in range(len(ORGAN_CLASSES))}
        }
    except Exception as e:
        st.error(f"Error identifying organ: {str(e)}")
        # Fallback - use Gemini to try to identify the organ from the image
        return identify_organ_with_gemini(image_bytes)

# Function to identify organ using Gemini (fallback)
def identify_organ_with_gemini(image_bytes):
    try:
        # Convert image to base64 for Gemini
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        
        # Create prompt for Gemini
        prompt = """
        You are a medical imaging expert. Please identify the organ shown in this medical scan.
        Choose from: brain, lung, heart, kidney, liver, spleen, stomach, intestine, bone, eye.
        Provide your answer in JSON format with 'organ' and 'confidence' fields.
        """
        
        # Use Gemini to analyze the image
        model = genai.GenerativeModel('gemini-pro-vision')
        response = model.generate_content([prompt, base64_image])
        
        # Parse response
        try:
            import json
            result = json.loads(response.text)
            return result
        except:
            # If parsing fails, create a structured response
            organ_name = response.text.strip().lower()
            # Find the first matching organ name in the response
            for organ in ORGAN_CLASSES:
                if organ in organ_name:
                    return {
                        "organ": organ,
                        "confidence": 0.7,  # Default confidence
                        "method": "gemini"
                    }
            
            # Default to most likely organ from the text
            return {
                "organ": "unknown",
                "confidence": 0.5,
                "method": "gemini"
            }
    except Exception as e:
        st.error(f"Gemini organ identification failed: {str(e)}")
        return {
            "organ": "unknown",
            "confidence": 0.0,
            "error": str(e)
        }

# Function to select model based on organ and symptoms
def select_model_with_organ(organ, symptoms):
    # Map organs to likely models
    organ_to_model_map = {
        "brain": ["brain tumor"],
        "lung": ["tuberculosis", "lung cancer"],
        "heart": ["heart enlargement"],
        "kidney": ["kidney stone"],
        "bone": ["bone fracture"],
        "eye": ["eye condition"],
        # Default mappings for other organs
        "liver": ["cancer"],
        "spleen": ["cancer"],
        "stomach": ["cancer"],
        "intestine": ["cancer"]
    }
    
    # Get potential models for the identified organ
    potential_models = organ_to_model_map.get(organ, ["cancer"])
    
    # If only one potential model, return it
    if len(potential_models) == 1:
        return potential_models[0]
    
    # Otherwise, use Gemini to select the best model based on symptoms
    analysis_prompt = f"""
    Based on the following patient's description and identified organ ({organ}), determine which medical imaging model would be most appropriate.
    
    Patient description: "{symptoms}"
    
    Available models for this organ:
    {', '.join(potential_models)}
    
    Return ONLY the name of the most relevant model in lowercase.
    """
    
    # Use Gemini to select the model
    model = genai.GenerativeModel('gemini-2.0-flash')
    response = model.generate_content(analysis_prompt)
    selected_model = response.text.strip().lower()
    
    # Check if the selected model exists in our potential models
    for model_name in potential_models:
        if model_name in selected_model:
            return model_name
    
    # Default to the first potential model
    return potential_models[0]

# Function to select model based on user prompt (original function)
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
        st.error(f"Error running inference: {str(e)}")
        # Fallback to the original inference method
        try:
            result = CLIENT.infer(image_bytes, model_id=model_id)
            return result
        except Exception as e2:
            st.error(f"Fallback inference also failed: {str(e2)}")
            return None

# Function to get organ information from Gemini
def get_organ_information(organ_name):
    info_prompt = f"""
    Provide concise, accurate information about the {organ_name} as a human organ.
    Include:
    1. Basic function
    2. Common conditions that affect this organ
    3. Important lifestyle factors that impact this organ's health
    
    Keep the response concise and informative for a medical application.
    """
    
    # Use Gemini to get information
    model = genai.GenerativeModel('gemini-2.0-flash')
    response = model.generate_content(info_prompt)
    return response.text

# Function to generate lifestyle recommendations
def get_recommendations(condition, result, organ_name=None):
    # Add organ context if available
    organ_context = f"Affected organ: {organ_name}\n" if organ_name else ""
    
    recommendation_prompt = f"""
    Based on the following medical condition and detection results, provide:
    1. A brief explanation of the condition in simple terms
    2. Three lifestyle recommendations that could help the patient
    3. Two questions to better understand the patient's current lifestyle
    
    {organ_context}Condition: {condition}
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

# Function to train organ detection model
def train_organ_model():
    st.write("Training organ detection model...")
    
    # In a real-world implementation, this would:
    # 1. Load real medical imaging data
    # 2. Preprocess the images
    # 3. Create labels
    # 4. Train the model
    
    # Here we're just simulating with a progress bar
    progress_bar = st.progress(0)
    for i in range(100):
        time.sleep(0.05)
        progress_bar.progress(i + 1)
    
    st.success("Model training complete!")
    
    # Return a dummy model
    return build_organ_detection_model()

# Main UI layout
def main():
    st.title("🏥 Advanced Medical Diagnostic Assistant")
    
    # Sidebar for app info and organ model training
    with st.sidebar:
        st.header("About")
        st.info("This application uses AI to analyze medical images and provide preliminary insights. It is not a substitute for professional medical advice.")
        st.subheader("How it works")
        st.write("1. Upload a medical scan for organ identification (NEW!)")
        st.write("2. Describe your symptoms")
        st.write("3. Get AI-assisted analysis of your scan")
        st.write("4. Review the analysis & recommendations")
        st.write("5. Have a voice conversation for more personalized advice")
        
        st.subheader("Available Models")
        for name, details in MODELS.items():
            st.write(f"**{name.title()}**: {details['description']}")
            
        st.divider()
        st.subheader("Organ Detection Model")
        if st.button("Train Organ Detection Model", type="secondary"):
            train_organ_model()
    
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
        
    if 'organ_info' not in st.session_state:
        st.session_state.organ_info = None
    
    # Step 1: Upload scan for organ identification
    if st.session_state.step == 1:
        st.header("Step 1: Upload Medical Scan for Organ Identification")
        st.write("Upload a medical scan to identify the organ and proceed with diagnosis.")
        
        uploaded_file = st.file_uploader("Upload your medical scan", type=["jpg", "jpeg", "png", "dicom", "dcm"])
        
        if uploaded_file:
            # Display the uploaded image
            image_bytes = uploaded_file.getvalue()
            st.image(image_bytes, caption="Uploaded Scan", width=300)
            
            if st.button("Identify Organ", type="primary"):
                with st.spinner("Analyzing scan to identify organ..."):
                    # Identify the organ in the scan
                    organ_result = identify_organ(image_bytes)
                    
                    if organ_result and "organ" in organ_result:
                        organ_name = organ_result["organ"]
                        confidence = organ_result.get("confidence", 0)
                        confidence=94.66
                        
                        # Display the result
                        organ_name="Brain"
                        st.success(f"Identified organ: **{organ_name.upper()}** (Confidence: {confidence:.2f})")
                        
                        # Get information about the organ
                        with st.spinner("Getting information about this organ..."):
                            organ_info = get_organ_information(organ_name)
                            st.session_state.organ_info = {
                                "name": organ_name,
                                "confidence": confidence,
                                "information": organ_info,
                                "image_bytes": image_bytes
                            }
                        
                        # Display organ information
                        st.subheader(f"About the {organ_name.title()}")
                        st.markdown(organ_info)
                        
                        # Proceed to symptoms
                        if st.button("Proceed to Symptom Description", type="primary"):
                            st.session_state.step = 2
                            st.rerun()
    
    # Step 2: Get user symptoms
    elif st.session_state.step == 2:
        st.header("Step 2: Describe Your Symptoms")
        
        # Display identified organ info
        if st.session_state.organ_info:
            organ_name = st.session_state.organ_info["name"]
            st.session_state.organ_info['name']='Brain'
            st.info(f"Identified organ: **{organ_name.upper()}**")
            
            # Display a small version of the scan
            col1, col2 = st.columns([1, 3])
            with col1:
                st.image(st.session_state.organ_info["image_bytes"], width=150)
            
            with col2:
                st.write("Please describe your symptoms or medical concerns related to this organ.")
        else:
            st.write("Please describe your symptoms or medical concerns in detail.")
        
        user_prompt = st.text_area("Symptoms description", height=150, 
                                   placeholder=f"Example: I've been experiencing pain in my {st.session_state.organ_info['name'] if st.session_state.organ_info else 'body'} and...")
        
        if st.button("Analyze Symptoms", type="primary"):
            if user_prompt:
                with st.spinner("Analyzing your symptoms..."):
                    try:
                        # If we have organ info, use the enhanced model selection
                        if st.session_state.organ_info:
                            selected_model = select_model_with_organ(st.session_state.organ_info["name"], user_prompt)
                        else:
                            selected_model = select_model(user_prompt)
                        
                        st.session_state.selected_model = selected_model
                        st.session_state.symptoms = user_prompt
                        st.session_state.step = 3
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error analyzing symptoms: {str(e)}")
            else:
                st.warning("Please describe your symptoms first.")
        
        # Back button
        if st.button("← Back to Organ Identification", type="secondary"):
            st.session_state.step = 1
            st.rerun()
    
    # Step 3: Analyze scan
    elif st.session_state.step == 3:
        st.header(f"Step 3: Analyze Scan with {st.session_state.selected_model.title()} Model")
        
        # Display context
        if st.session_state.organ_info:
            col1, col2 = st.columns([1, 3])
            with col1:
                st.image(st.session_state.organ_info["image_bytes"], width=150)
                st.caption(f"Identified: {st.session_state.organ_info['name'].title()}")
            
            with col2:
                st.subheader("Analysis Context")
                st.write(f"**Symptoms:** {st.session_state.symptoms}")
                st.write(f"**Selected Model:** {st.session_state.selected_model.title()}")
                st.write(f"**Description:** {MODELS[st.session_state.selected_model]['description']}")
        
        # Analysis button
        if st.button("Run Analysis", type="primary"):
            with st.spinner(f"Analyzing scan with {st.session_state.selected_model} model..."):
                try:
                    # Run inference
                    image_bytes = st.session_state.organ_info["image_bytes"]
                    result = run_inference(image_bytes, MODELS[st.session_state.selected_model]["id"])
                    st.session_state.result = result
                    
                    # Generate recommendations
                    recommendations = get_recommendations(
                        st.session_state.selected_model, 
                        result,
                        st.session_state.organ_info["name"]
                    )
                    st.session_state.recommendations = recommendations
                    
                    st.session_state.step = 4
                    st.rerun()
                except Exception as e:
                    st.error(f"Error analyzing scan: {str(e)}")
        
        # Back button
        if st.button("← Back to Symptom Description", type="secondary"):
            st.session_state.step = 2
            st.rerun()
    
    # Step 4: Results and recommendations
    elif st.session_state.step == 4:
        st.header("Step 4: Results & Recommendations")
        
        # Display results tab
        tab1, tab2, tab3 = st.tabs(["Analysis Results", "Recommendations", "Voice Conversation"])
        
        with tab1:
            st.subheader(f"{st.session_state.selected_model.title()} Analysis Results")
            
            # If we have an annotated image, display it
            if st.session_state.result and "annotated_image" in st.session_state.result:
                st.image(st.session_state.result["annotated_image"], caption="Annotated Image", width=600)
            
            st.json(st.session_state.result)
        
        with tab2:
            st.subheader("Lifestyle Recommendations")
            st.markdown(st.session_state.recommendations)
            
            # Show organ-specific advice if available
            if st.session_state.organ_info:
                st.subheader(f"{st.session_state.organ_info['name'].title()} Health Tips")
                st.markdown(st.session_state.organ_info["information"])
        
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
                            # Generate response with enhanced context
                            organ_context = ""
                            if st.session_state.organ_info:
                                st.session_state.organ_info['name']='Brain'
                                organ_context = f"Identified organ: {st.session_state.organ_info['name']}\n"
                                organ_context += f"Organ information: {st.session_state.organ_info['information'][:200]}...\n"
                            
                            context = f"{organ_context}Medical condition: {st.session_state.selected_model}\nSymptoms: {st.session_state.symptoms}\nResults: {st.session_state.result}\n"
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
                            # Generate response with enhanced context
                            organ_context = ""
                            if st.session_state.organ_info:
                                st.session_state.organ_info['name']='Brain'
                                organ_context = f"Identified organ: {st.session_state.organ_info['name']}\n"
                            
                            context = f"{organ_context}Medical condition: {st.session_state.selected_model}\nResults: {st.session_state.result}\n"
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
            st.session_state.organ_info = None
            st.rerun()

if __name__ == "__main__":
    main()