import streamlit as st
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import DetrImageProcessor, DetrForObjectDetection

# Page title
st.title("Real-time Object Detection")

# Set up device for inference
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load DETR model (Facebook's Detection Transformer model)
@st.cache_resource
def load_model():
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
    model = model.to(device)  # Move model to GPU if available
    
    # Get the model's config to access the id2label mapping
    id2label = model.config.id2label
    
    return processor, model, id2label

processor, model, id2label = load_model()

# Function to perform object detection
def detect_objects(image, confidence_threshold=0.5, max_detections=20):
    # Convert image to RGB (if needed) and to PIL format
    if isinstance(image, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Process image with the model
    with torch.no_grad():  # Disable gradient calculation for inference
        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)
    
    # Post-process results
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=confidence_threshold)[0]
    
    # Draw bounding boxes and labels
    img_array = np.array(image)
    
    # Sort results by confidence score (highest first) and limit to max_detections
    sorted_indices = sorted(range(len(results["scores"])), key=lambda i: results["scores"][i], reverse=True)[:max_detections]
    
    for idx in sorted_indices:
        score = results["scores"][idx]
        label = results["labels"][idx]
        box = results["boxes"][idx]
        
        box = [round(i) for i in box.tolist()]
        class_name = id2label[label.item()]
        confidence = round(score.item(), 3)
        
        # Draw rectangle
        cv2.rectangle(img_array, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        
        # Add label with confidence score
        label_text = f"{class_name}: {confidence:.2f}"
        cv2.putText(img_array, label_text, (box[0], box[1]-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return img_array

# Sidebar options
st.sidebar.title("Settings")
detection_mode = st.sidebar.radio("Choose detection mode:", ["Webcam (Real-time)", "Upload Image"])

# Add performance settings to sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("Performance Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
skip_frames = st.sidebar.slider("Skip Frames", 0, 10, 2, 1, 
                             help="Process every Nth frame to improve performance (higher = better performance but choppier video)")
resize_factor = st.sidebar.slider("Resize Factor", 0.2, 1.0, 0.5, 0.1,
                               help="Resize input frames to improve performance (lower = better performance but lower quality)")
max_detections = st.sidebar.slider("Max Detections", 1, 100, 20, 1,
                                help="Limit maximum number of detected objects to display")

if detection_mode == "Webcam (Real-time)":
    # Create a placeholder for the webcam feed
    video_placeholder = st.empty()
    
    # Add a button to start/stop webcam
    col1, col2 = st.columns(2)
    with col1:
        start = st.button("Start Camera")
    with col2:
        stop = st.button("Stop Camera")
    
    # Ensure webcam is properly released when the app is closed or refreshed
    if 'cap' in st.session_state and st.session_state.cap is not None:
        st.session_state.cap.release()
        st.session_state.cap = None
        st.session_state.webcam_running = False
    
    if start:
        # Add device selection dropdown
        camera_options = ["0", "1", "2", "3"]
        selected_camera = st.selectbox(
            "Select camera (try different numbers if default doesn't work):",
            camera_options,
            index=0
        )
        
        # Start webcam with selected device
        try:
            # Initialize cap in session state so we can release it properly later
            st.session_state.cap = cv2.VideoCapture(int(selected_camera))
            cap = st.session_state.cap
            
            # Try setting resolution to something standard
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            # Set buffer size to 1 to reduce latency (discard old frames)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            if not cap.isOpened():
                st.error(f"Failed to open camera device {selected_camera}")
                st.info("Try selecting a different camera number from the dropdown")
            else:
                st.session_state.webcam_running = True
                
                # Info message
                st.info("Webcam started. Press 'Stop Camera' to end.")
                
                while st.session_state.webcam_running:
                    # Read frame from webcam
                    ret, frame = cap.read()
                    
                    if not ret:
                        st.error("Failed to read from webcam. Try a different camera number.")
                        break
                    
                    # Apply optimizations to reduce lag
                    frame_count = st.session_state.get("frame_count", 0)
                    st.session_state.frame_count = frame_count + 1
                    
                    # Skip frames to improve performance
                    if frame_count % (skip_frames + 1) != 0:
                        continue
                    
                    # Resize frame to improve performance
                    if resize_factor < 1.0:
                        h, w = frame.shape[:2]
                        new_h, new_w = int(h * resize_factor), int(w * resize_factor)
                        frame = cv2.resize(frame, (new_w, new_h))
                    
                    # Process frame
                    processed_frame = detect_objects(frame, confidence_threshold, max_detections)
                    
                    # Display the processed frame
                    video_placeholder.image(processed_frame, channels="RGB", use_container_width=True)
                    
                    # Check if stop button was clicked
                    if stop:
                        st.session_state.webcam_running = False
                        break
                
                # Release webcam
                cap.release()
                st.session_state.cap = None
                st.session_state.webcam_running = False
                st.success("Webcam stopped.")
        except Exception as e:
            st.error(f"Error accessing webcam: {e}")
            st.info("Try selecting a different camera number or check if your webcam is properly connected")
            # Make sure to release camera even if there's an error
            if 'cap' in st.session_state and st.session_state.cap is not None:
                st.session_state.cap.release()
                st.session_state.cap = None
    
elif detection_mode == "Upload Image":
    # Image upload
    uploaded_img = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_img is not None:
        # Read and display the uploaded image
        image = Image.open(uploaded_img)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Process the image when button is clicked
        if st.button("Detect Objects"):
            with st.spinner("Detecting objects..."):
                # Process the image
                result_image = detect_objects(image, confidence_threshold, max_detections)
                
                # Display the result
                st.image(result_image, caption="Detection Result", use_container_width=True)

# Provide explanation about the model
st.sidebar.markdown("---")
st.sidebar.markdown("""
## About
This app uses Facebook's DETR (DEtection TRansformer) model from HuggingFace to detect objects in real-time.

The model can recognize 80 different object categories including people, cars, animals, and everyday items.

The object categories are retrieved directly from the model's configuration instead of using a manual class list.
""")

# Initialize session states if they don't exist
if 'webcam_running' not in st.session_state:
    st.session_state.webcam_running = False
if 'cap' not in st.session_state:
    st.session_state.cap = None

# Register a handler to release camera when the app is closed or refreshed
def release_camera():
    if 'cap' in st.session_state and st.session_state.cap is not None:
        st.session_state.cap.release()
        st.session_state.cap = None

# Use Streamlit's experimental session_state feature to register a callback to release the camera
try:
    import atexit
    atexit.register(release_camera)
except:
    pass
