document.addEventListener('DOMContentLoaded', function() {
    // DOM elements
    const recordBtn = document.getElementById('recordBtn');
    const uploadBtn = document.getElementById('uploadBtn');
    const generateBtn = document.getElementById('generateBtn');
    const downloadBtn = document.getElementById('downloadBtn');
    const voiceUpload = document.getElementById('voiceUpload');
    const audioPreview = document.getElementById('audioPreview');
    const resultAudio = document.getElementById('resultAudio');
    const waveformContainer = document.getElementById('waveformContainer');
    const status = document.getElementById('status');
    const speedInput = document.getElementById('speedInput');
    const speedValue = document.getElementById('speedValue');
    const textInput = document.getElementById('textInput');
    
    // Variables
    let mediaRecorder = null;
    let audioBlob = null;
    let wavesurfer = null;
    
    // Initialize WaveSurfer
    function initWaveSurfer() {
        if (wavesurfer) {
            wavesurfer.destroy();
        }
        
        wavesurfer = WaveSurfer.create({
            container: '#audioWaveform',
            waveColor: '#4a6baf',
            progressColor: '#2c4a9a',
            cursorColor: '#1a2f5f',
            barWidth: 2,
            barRadius: 3,
            cursorWidth: 1,
            height: 80,
            barGap: 2,
            responsive: true
        });
        
        wavesurfer.on('ready', function() {
            if (audioBlob) {
                audioPreview.src = URL.createObjectURL(audioBlob);
                waveformContainer.style.display = 'block';
            }
        });
    }
    
    // Update speed display
    speedInput.addEventListener('input', function() {
        speedValue.textContent = this.value;
    });
    
    // Handle file upload
    voiceUpload.addEventListener('change', function(e) {
        if (this.files.length > 0) {
            audioBlob = this.files[0];
            initWaveSurfer();
            wavesurfer.loadBlob(audioBlob);
            status.textContent = 'File selected. Click Upload to proceed.';
            status.className = 'status status-success';
        }
    });
    
    // Record audio
    recordBtn.addEventListener('click', async function() {
        if (!mediaRecorder) {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new RecordRTC(stream, {
                    type: 'audio',
                    mimeType: 'audio/wav',
                    recorderType: RecordRTC.StereoAudioRecorder,
                    desiredSampRate: 16000,
                    numberOfAudioChannels: 1
                });
                
                mediaRecorder.startRecording();
                recordBtn.innerHTML = '<i class="bi bi-stop"></i> Stop';
                recordBtn.classList.add('btn-secondary');
                recordBtn.classList.remove('btn-danger');
                status.textContent = 'Recording...';
                status.className = 'status';
                audioBlob = null; // Clear previous audio
                console.log('Recording started');
                
            } catch (error) {
                console.error('Error accessing microphone:', error);
                status.textContent = 'Microphone access denied. Please allow microphone access.';
                status.className = 'status status-error';
            }
        } else {
            try {
                mediaRecorder.stopRecording(function() {
                    audioBlob = mediaRecorder.getBlob();
                    initWaveSurfer();
                    wavesurfer.loadBlob(audioBlob);
                    recordBtn.innerHTML = '<i class="bi bi-mic"></i> Record';
                    recordBtn.classList.add('btn-danger');
                    recordBtn.classList.remove('btn-secondary');
                    status.textContent = 'Recording complete. Click Upload to proceed.';
                    status.className = 'status status-success';
                    
                    // Stop all tracks
                    const stream = mediaRecorder.getInternalRecorder().stream;
                    stream.getTracks().forEach(track => track.stop());
                    console.log('Recording stopped, stream closed');
                    mediaRecorder = null; // Reset mediaRecorder
                });
            } catch (error) {
                console.error('Error stopping recording:', error);
                status.textContent = 'Failed to stop recording: ' + error.message;
                status.className = 'status status-error';
                mediaRecorder = null; // Reset to allow retry
            }
        }
    });
    
    // Upload audio
    uploadBtn.addEventListener('click', async function() {
        if (!audioBlob) {
            status.textContent = 'No audio to upload. Record or select a file first.';
            status.className = 'status status-error';
            return;
        }
        
        try {
            uploadBtn.disabled = true;
            status.textContent = 'Uploading...';
            status.className = 'status';
            
            const formData = new FormData();
            formData.append('file', audioBlob, 'recording.wav');
            
            const response = await fetch('/upload_voice/', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(errorText || 'Upload failed');
            }
            
            const result = await response.json();
            status.textContent = result.message || 'Upload successful';
            status.className = 'status status-success';
            
        } catch (error) {
            console.error('Upload error:', error);
            status.textContent = 'Upload failed: ' + error.message;
            status.className = 'status status-error';
        } finally {
            uploadBtn.disabled = false;
        }
    });
    
    // Generate cloned voice
    generateBtn.addEventListener('click', async function() {
        const text = textInput.value.trim();
        if (!text) {
            status.textContent = 'Please enter some text';
            status.className = 'status status-error';
            return;
        }
        
        try {
            generateBtn.disabled = true;
            status.textContent = 'Generating cloned voice...';
            status.className = 'status';
            
            const response = await fetch('/generate_cloned_voice/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    text: text,
                    speed: parseFloat(speedInput.value)
                })
            });
            
            if (!response.ok) {
                const errorText = await response.text();
                try {
                    const errorJson = JSON.parse(errorText);
                    throw new Error(errorJson.detail || 'Generation failed');
                } catch {
                    throw new Error(errorText || 'Generation failed');
                }
            }
            
            const audioBlob = await response.blob();
            const audioUrl = URL.createObjectURL(audioBlob);
            resultAudio.src = audioUrl;
            resultContainer.style.display = 'block';
            status.textContent = 'Generation complete! Play or download the audio.';
            status.className = 'status status-success';
            
            downloadBtn.onclick = function() {
                const a = document.createElement('a');
                a.href = audioUrl;
                a.download = 'cloned_voice.wav';
                a.click();
            };
            
        } catch (error) {
            console.error('Generation error:', error);
            status.textContent = 'Generation failed: ' + error.message;
            status.className = 'status status-error';
        } finally {
            generateBtn.disabled = false;
        }
    });
});