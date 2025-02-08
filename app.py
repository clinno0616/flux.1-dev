import streamlit as st
import torch
from diffusers import DiffusionPipeline
from diffusers import AutoencoderKL  # Add this import at the top
from PIL import Image
import os
import gc
import glob
import psutil
import time
import plotly.graph_objects as go
from collections import deque
import threading
import pynvml
from datetime import datetime

# Initialize monitoring data structures
class ResourceMonitor:
    def __init__(self, max_points=100):
        self.max_points = max_points
        self.timestamps = deque(maxlen=max_points)
        self.cpu_usage = deque(maxlen=max_points)
        self.ram_usage = deque(maxlen=max_points)
        self.gpu_usage = deque(maxlen=max_points)
        self.vram_usage = deque(maxlen=max_points)
        self.disk_usage = deque(maxlen=max_points)
        self.disk_io = deque(maxlen=max_points)
        self._last_disk_io = psutil.disk_io_counters()
        self.monitoring = False
        self.lock = threading.Lock()

    def start_monitoring(self):
        self.monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitor_resources)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()

    def stop_monitoring(self):
        self.monitoring = False

    def _monitor_resources(self):
        # Initialize NVIDIA Management Library
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            has_gpu = True
        except:
            has_gpu = False

        while self.monitoring:
            with self.lock:
                current_time = datetime.now()
                self.timestamps.append(current_time)
                
                # CPU Usage
                self.cpu_usage.append(psutil.cpu_percent())
                
                # RAM Usage
                ram = psutil.virtual_memory()
                self.ram_usage.append(ram.percent)
                
                # GPU & VRAM Usage
                if has_gpu:
                    try:
                        gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                        vram_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        vram_percent = (vram_info.used / vram_info.total) * 100
                    except:
                        gpu_util = 0
                        vram_percent = 0
                else:
                    gpu_util = 0
                    vram_percent = 0
                
                self.gpu_usage.append(gpu_util)
                self.vram_usage.append(vram_percent)
                
                # Disk Usage
                disk = psutil.disk_usage('/')
                self.disk_usage.append(disk.percent)
                
                # Disk I/O
                current_disk_io = psutil.disk_io_counters()
                disk_read_write = (
                    (current_disk_io.read_bytes + current_disk_io.write_bytes) -
                    (self._last_disk_io.read_bytes + self._last_disk_io.write_bytes)
                ) / 1024 / 1024  # Convert to MB/s
                self.disk_io.append(min(100, disk_read_write))  # Cap at 100 for visualization
                self._last_disk_io = current_disk_io
            
            time.sleep(1)

    def get_plots(self):
        with self.lock:
            # Create timestamp labels
            labels = [t.strftime('%H:%M:%S') for t in self.timestamps]
            
            # CPU Usage Plot
            cpu_fig = go.Figure()
            cpu_fig.add_trace(go.Scatter(x=labels, y=list(self.cpu_usage),
                                       fill='tozeroy', name='CPU Usage',
                                       line=dict(color='blue')))
            cpu_fig.update_layout(
                title='CPU Usage (%)',
                margin=dict(l=0, r=0, t=30, b=0),
                height=200,
                showlegend=False
            )
            
            # RAM Usage Plot
            ram_fig = go.Figure()
            ram_fig.add_trace(go.Scatter(x=labels, y=list(self.ram_usage),
                                       fill='tozeroy', name='RAM Usage',
                                       line=dict(color='green')))
            ram_fig.update_layout(
                title='RAM Usage (%) (64GB)',
                margin=dict(l=0, r=0, t=30, b=0),
                height=200,
                showlegend=False
            )
            
            # GPU Usage Plot
            gpu_fig = go.Figure()
            gpu_fig.add_trace(go.Scatter(x=labels, y=list(self.gpu_usage),
                                       fill='tozeroy', name='GPU Usage',
                                       line=dict(color='purple')))
            gpu_fig.update_layout(
                title='GPU Usage (%)',
                margin=dict(l=0, r=0, t=30, b=0),
                height=200,
                showlegend=False
            )
            
            # VRAM Usage Plot
            vram_fig = go.Figure()
            vram_fig.add_trace(go.Scatter(x=labels, y=list(self.vram_usage),
                                        fill='tozeroy', name='VRAM Usage',
                                        line=dict(color='red')))
            vram_fig.update_layout(
                title='VRAM Usage (%) (16GB)',
                margin=dict(l=0, r=0, t=30, b=0),
                height=200,
                showlegend=False
            )
            
            # Disk Usage Plot
            disk_fig = go.Figure()
            disk_fig.add_trace(go.Scatter(x=labels, y=list(self.disk_usage),
                                        fill='tozeroy', name='Disk Usage',
                                        line=dict(color='orange')))
            disk_fig.update_layout(
                title='Disk Usage (%)',
                margin=dict(l=0, r=0, t=30, b=0),
                height=200,
                showlegend=False
            )
            
            # Disk I/O Plot
            disk_io_fig = go.Figure()
            disk_io_fig.add_trace(go.Scatter(x=labels, y=list(self.disk_io),
                                           fill='tozeroy', name='Disk I/O',
                                           line=dict(color='brown')))
            disk_io_fig.update_layout(
                title='Disk I/O (MB/s)',
                margin=dict(l=0, r=0, t=30, b=0),
                height=200,
                showlegend=False
            )
            
            return cpu_fig, ram_fig, gpu_fig, vram_fig, disk_fig, disk_io_fig

# Set page config
st.set_page_config(page_title="black-forest-labs/FLUX.1-dev", layout="wide")

# Custom CSS to control input width and fonts
st.markdown("""
<style>
    div[data-baseweb="input"] {
        width: 100px !important;
    }
    .custom-title {
        font-weight: bold;
        font-size: 24px;
    }
</style>
""", unsafe_allow_html=True)

# Add title and description
st.markdown("<p class='custom-title'>black-forest-labs/FLUX.1-dev</p>", unsafe_allow_html=True)

# Function to scan for LORA files
def scan_lora_files(lora_dir="lora"):
    """Scan for .safetensors files in the LORA directory"""
    if not os.path.exists(lora_dir):
        os.makedirs(lora_dir)
    lora_files = glob.glob(os.path.join(lora_dir, "*.safetensors"))
    return {os.path.basename(f): f for f in lora_files}

# Initialize session state for the model and LORA files
if 'pipe' not in st.session_state:
    @st.cache_resource
    def load_model():
        # Load VAE first
        vae = AutoencoderKL.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            subfolder="vae",
            torch_dtype=torch.bfloat16,
            use_safetensors=True
        )
         # Load the main pipeline with the VAE
        pipe = DiffusionPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            vae=vae,  # Pass the loaded VAE object
            torch_dtype=torch.bfloat16,
            use_safetensors=True
        )
        
        pipe.enable_sequential_cpu_offload()
        pipe.enable_attention_slicing()
        
        return pipe
    
    with st.spinner("Loading model... This might take a while."):
        st.session_state.pipe = load_model()

# Scan for LORA files
lora_files = scan_lora_files()

# Create sidebar for model settings
st.sidebar.header("Generation Settings")

# Add optimization mode option
optimization_mode = st.sidebar.radio(
    "Optimization Mode",
    ["Standard"],
    help="Memory Efficient mode uses less VRAM but might be slower"
)

max_dim = 2048
col1, col2 = st.columns(2)
# Input for prompt
with col1:
    prompt = st.text_area("Enter your prompt:", 
                     value="A cat holding a sign that says hello world",
                     height=100)
with col2:
    col3, col4 = st.columns(2)
    with col3:
        width = st.number_input("Image Width", 
                        min_value=256, 
                        max_value=max_dim, 
                        value=576, 
                        step=128)
    with col4:
        height = st.number_input("Image Height", 
                        min_value=256, 
                        max_value=max_dim, 
                        value=768, 
                        step=128)

# Advanced settings in sidebar
guidance_scale = st.sidebar.slider("Guidance Scale", 
                                 min_value=1.0, 
                                 max_value=20.0, 
                                 value=3.5, 
                                 step=0.5)

default_steps = 20 if optimization_mode == "Memory Efficient" else 20
num_inference_steps = st.sidebar.slider("Number of Inference Steps", 
                                      min_value=1, 
                                      max_value=100, 
                                      value=default_steps)
max_sequence_length = 256
# Function to update seed
def update_seed():
    new_seed = torch.randint(0, 2**32 - 1, (1,)).item()
    st.session_state['seed'] = new_seed
    st.sidebar.markdown(f"**Current Seed:** {st.session_state['seed']}")

# Display current VRAM usage if available
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()
    current_vram = torch.cuda.memory_allocated() / 1024**3
    #st.sidebar.markdown(f"Current VRAM Usage: {current_vram:.2f} GB")

# Add LORA selection section
st.sidebar.header("LORA Settings")
selected_loras = {}
if lora_files:
    st.sidebar.markdown("Select LORA files and their weights:")
    for lora_name in lora_files.keys():
        col1, col2 = st.sidebar.columns([1, 3])
        with col1:
            weight = st.number_input(
            "Weight",
            min_value=0.0,
            max_value=1.0,
            value=0.6,
            step=0.1,
            key=f"weight_{lora_name}",
            label_visibility="collapsed"  # This hides the label
            )
        with col2:
            lora_display_name = os.path.splitext(lora_name)[0]  # Remove file extension
            use_lora = st.checkbox(f"{lora_display_name}", key=f"use_{lora_name}")

        if use_lora:
            selected_loras[lora_files[lora_name]] = weight
else:
    st.sidebar.warning("No LORA files found in the 'lora' directory")

# Initialize the resource monitor in session state if not present
if 'resource_monitor' not in st.session_state:
    st.session_state.resource_monitor = ResourceMonitor()

# Generate button
if st.button("Generate Image"):
    try:
        # Start monitoring if not already started
        if not st.session_state.resource_monitor.monitoring:
            st.session_state.resource_monitor.start_monitoring()

        with st.spinner("Generating image..."):
            # Clean up memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            # Set up generator if using seed
            update_seed()  # Update the seed
            generator = torch.Generator("cpu").manual_seed(st.session_state['seed'])

            # Load and apply LORA weights if selected
            pipe = st.session_state.pipe
            if selected_loras:
                for lora_path, weight in selected_loras.items():
                    # Load and apply LORA weights
                    pipe.load_lora_weights(lora_path, weight=weight)
            
            # Generate image
            with torch.inference_mode():
                image = pipe(
                    prompt,
                    height=height,
                    width=width,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    max_sequence_length=max_sequence_length,
                    generator=generator
                ).images[0]
            
            # Unload LORA weights after generation
            if selected_loras:
                pipe.unload_lora_weights()
            
            # Display the generated image
            st.image(image, caption="Generated Image")
            
            # Add download button
            col1, col2 = st.columns(2)
            with col1:
                # Save image temporarily
                temp_path = "generated_image.png"
                image.save(temp_path)
                
                # Create download button
                with open(temp_path, "rb") as file:
                    btn = st.download_button(
                        label="Download Image",
                        data=file,
                        file_name="generated_image.png",
                        mime="image/png"
                    )
                
                # Clean up temporary file
                os.remove(temp_path)
            
            # Display VRAM usage after generation
            if torch.cuda.is_available():
                current_vram = torch.cuda.memory_allocated() / 1024**3
                st.sidebar.markdown(f"VRAM Usage After Generation: {current_vram:.2f} GB")

            # After generation, update and display resource plots
            st.sidebar.markdown("### System Resource Usage")
            cpu_fig, ram_fig, gpu_fig, vram_fig, disk_fig, disk_io_fig = st.session_state.resource_monitor.get_plots()
            
            # Display plots in sidebar
            col1, col2 = st.sidebar.columns([1, 1])
            with col1:
                st.plotly_chart(cpu_fig, use_container_width=True)
            with col2:
                st.plotly_chart(ram_fig, use_container_width=True)
            col1, col2 = st.sidebar.columns([1, 1])
            with col1:
                st.plotly_chart(gpu_fig, use_container_width=True)
            with col2:
                st.plotly_chart(vram_fig, use_container_width=True)
               
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Try using Memory Efficient mode or reducing image dimensions if you're experiencing memory issues.")
