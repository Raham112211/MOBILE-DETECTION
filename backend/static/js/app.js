/**
 * RF-DETR Mobile Phone Detection & ByteTrack Studio
 * Client Application Logic & Real-time Telemetry Manager
 */

document.addEventListener('DOMContentLoaded', () => {
  // DOM Element References
  const btnStart = document.getElementById('btnStart');
  const btnPause = document.getElementById('btnPause');
  const btnStop = document.getElementById('btnStop');
  const btnDownload = document.getElementById('btnDownload');
  const btnQuickSample = document.getElementById('btnQuickSample');
  const btnLoadSampleSide = document.getElementById('btnLoadSampleSide');
  
  const videoFeed = document.getElementById('videoFeed');
  const feedPlaceholder = document.getElementById('feedPlaceholder');
  const statusTickerText = document.getElementById('statusTickerText');
  const systemStatusBadge = document.getElementById('systemStatusBadge');
  const systemStatusText = document.getElementById('systemStatusText');
  
  // HUD Elements
  const hudVideoName = document.getElementById('hudVideoName');
  const hudResolutionChip = document.getElementById('hudResolutionChip');
  const hudFpsVal = document.getElementById('hudFpsVal');
  const hudLatencyVal = document.getElementById('hudLatencyVal');
  const hudActivePhones = document.getElementById('hudActivePhones');
  const hudTrackIds = document.getElementById('hudTrackIds');
  
  // Timeline Elements
  const timelineFill = document.getElementById('timelineFill');
  const timelineFrameText = document.getElementById('timelineFrameText');
  const timelinePercentText = document.getElementById('timelinePercentText');
  
  // Sidebar Telemetry
  const metricFps = document.getElementById('metricFps');
  const metricLatency = document.getElementById('metricLatency');
  const metricActivePhones = document.getElementById('metricActivePhones');
  const metricUniqueTracks = document.getElementById('metricUniqueTracks');
  
  // File Upload Elements
  const dropzone = document.getElementById('dropzone');
  const videoFileInput = document.getElementById('videoFileInput');
  const metaFilename = document.getElementById('metaFilename');
  const metaResolution = document.getElementById('metaResolution');
  const metaFps = document.getElementById('metaFps');
  
  // Confidence Slider
  const confSlider = document.getElementById('confSlider');
  const confValueDisplay = document.getElementById('confValueDisplay');

  // Runtime State
  let isProcessing = false;
  let isPaused = false;
  let statusPollTimer = null;
  let confDebounceTimer = null;

  // Initialize
  initEventListeners();
  startStatusPolling(800);

  function initEventListeners() {
    // Pipeline Controls
    btnStart.addEventListener('click', handleStartPipeline);
    btnPause.addEventListener('click', handleTogglePause);
    btnStop.addEventListener('click', handleStopPipeline);
    btnDownload.addEventListener('click', handleDownloadVideo);
    
    // Sample Video Loaders
    btnQuickSample.addEventListener('click', handleLoadSample);
    btnLoadSampleSide.addEventListener('click', handleLoadSample);

    // Confidence Slider
    confSlider.addEventListener('input', (e) => {
      const val = e.target.value;
      confValueDisplay.textContent = `${val}%`;
      
      clearTimeout(confDebounceTimer);
      confDebounceTimer = setTimeout(() => {
        sendConfidenceUpdate(val / 100.0);
      }, 200);
    });

    // File Drag & Drop Upload
    dropzone.addEventListener('dragover', (e) => {
      e.preventDefault();
      dropzone.classList.add('dragover');
    });

    dropzone.addEventListener('dragleave', () => {
      dropzone.classList.remove('dragover');
    });

    dropzone.addEventListener('drop', (e) => {
      e.preventDefault();
      dropzone.classList.remove('dragover');
      if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        uploadVideoFile(e.dataTransfer.files[0]);
      }
    });

    videoFileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files.length > 0) {
        uploadVideoFile(e.target.files[0]);
      }
    });
  }

  /**
   * Start video processing pipeline
   */
  async function handleStartPipeline() {
    const confVal = parseInt(confSlider.value, 10) / 100.0;
    
    btnStart.disabled = true;
    statusTickerText.textContent = "Starting GPU pipeline (NVDEC + RF-DETR Small)...";
    
    try {
      const res = await fetch('/api/pipeline/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ conf: confVal }),
      });
      const data = await res.json();
      
      if (data.status === 'success') {
        isProcessing = true;
        isPaused = false;
        btnPause.disabled = false;
        btnStop.disabled = false;
        btnDownload.disabled = true;

        // Switch to live MJPEG stream
        feedPlaceholder.classList.add('hidden');
        videoFeed.classList.remove('hidden');
        videoFeed.src = `/api/stream/feed?t=${Date.now()}`;

        // Fast status polling during inference (250ms)
        startStatusPolling(250);
      } else {
        alert(data.message || 'Failed to start pipeline');
        btnStart.disabled = false;
      }
    } catch (err) {
      console.error('Error starting pipeline:', err);
      statusTickerText.textContent = 'Error connecting to server.';
      btnStart.disabled = false;
    }
  }

  /**
   * Toggle Pause / Resume
   */
  async function handleTogglePause() {
    try {
      const res = await fetch('/api/pipeline/pause', { method: 'POST' });
      const data = await res.json();
      if (data.status === 'success') {
        isPaused = data.paused;
        if (isPaused) {
          btnPause.querySelector('span').textContent = 'RESUME';
          btnPause.style.background = 'rgba(16, 185, 129, 0.2)';
          btnPause.style.color = '#10b981';
          btnPause.style.borderColor = 'rgba(16, 185, 129, 0.4)';
        } else {
          btnPause.querySelector('span').textContent = 'PAUSE';
          btnPause.style.background = '';
          btnPause.style.color = '';
          btnPause.style.borderColor = '';
        }
      }
    } catch (err) {
      console.error('Error pausing pipeline:', err);
    }
  }

  /**
   * Stop Pipeline
   */
  async function handleStopPipeline() {
    try {
      btnStop.disabled = true;
      btnPause.disabled = true;
      statusTickerText.textContent = 'Stopping pipeline...';
      await fetch('/api/pipeline/stop', { method: 'POST' });
    } catch (err) {
      console.error('Error stopping pipeline:', err);
    }
  }

  /**
   * Load Default Sample Video
   */
  async function handleLoadSample() {
    statusTickerText.textContent = 'Loading sample video...';
    try {
      const res = await fetch('/api/load_sample', { method: 'POST' });
      const data = await res.json();
      if (data.status === 'success') {
        hudVideoName.textContent = 'sample.mp4';
        metaFilename.textContent = 'sample.mp4';
        metaResolution.textContent = '1280 x 720';
        metaFps.textContent = '24.0 FPS';
        btnStart.disabled = false;
        statusTickerText.textContent = 'Sample video loaded. Ready to start detection.';
      }
    } catch (err) {
      console.error('Error loading sample video:', err);
    }
  }

  /**
   * Upload video file from client
   */
  async function uploadVideoFile(file) {
    statusTickerText.textContent = `Uploading ${file.name}...`;
    const formData = new FormData();
    formData.append('video', file);

    try {
      const res = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();

      if (data.status === 'success') {
        hudVideoName.textContent = data.filename;
        metaFilename.textContent = data.filename;
        metaResolution.textContent = `${data.width} x ${data.height}`;
        metaFps.textContent = `${data.fps} FPS`;
        btnStart.disabled = false;
        btnDownload.disabled = true;
        statusTickerText.textContent = `Uploaded '${data.filename}'. Click START to process.`;
      } else {
        alert(data.message || 'Upload failed');
      }
    } catch (err) {
      console.error('Error uploading video:', err);
      statusTickerText.textContent = 'Upload failed.';
    }
  }

  /**
   * Send live confidence update to backend
   */
  async function sendConfidenceUpdate(conf) {
    try {
      await fetch('/api/settings/confidence', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ conf }),
      });
    } catch (err) {
      console.error('Error updating confidence:', err);
    }
  }

  /**
   * Download the processed high-quality video
   */
  function handleDownloadVideo() {
    window.location.href = '/api/video/download';
  }

  /**
   * Periodic Pipeline Status Poller
   */
  function startStatusPolling(intervalMs) {
    if (statusPollTimer) clearInterval(statusPollTimer);
    statusPollTimer = setInterval(pollPipelineStatus, intervalMs);
  }

  async function pollPipelineStatus() {
    try {
      const res = await fetch('/api/pipeline/status');
      if (!res.ok) return;
      const state = await res.json();
      updateUIWithStatus(state);
    } catch (err) {
      // Server might be busy starting up
    }
  }

  /**
   * Update all UI telemetry elements from state
   */
  function updateUIWithStatus(state) {
    // Top Bar Status Badge
    if (state.state === 'processing') {
      systemStatusText.textContent = 'Processing (GPU NVDEC + RF-DETR)';
      systemStatusBadge.style.borderColor = 'rgba(0, 242, 254, 0.4)';
      systemStatusBadge.querySelector('.status-indicator').style.background = '#00f2fe';
    } else if (state.state === 'completed') {
      systemStatusText.textContent = 'Processing Complete';
      systemStatusBadge.style.borderColor = 'rgba(16, 185, 129, 0.4)';
      systemStatusBadge.querySelector('.status-indicator').style.background = '#10b981';
    } else if (state.state === 'paused') {
      systemStatusText.textContent = 'Pipeline Paused';
      systemStatusBadge.style.borderColor = 'rgba(245, 158, 11, 0.4)';
      systemStatusBadge.querySelector('.status-indicator').style.background = '#f59e0b';
    } else {
      systemStatusText.textContent = 'System Ready';
      systemStatusBadge.style.borderColor = '';
      systemStatusBadge.querySelector('.status-indicator').style.background = '#10b981';
    }

    // Status Ticker
    if (state.status_message) {
      statusTickerText.textContent = state.status_message;
    }

    // HUD chips
    hudVideoName.textContent = state.active_video_name || 'sample.mp4';
    hudResolutionChip.textContent = `${state.decoder_type || 'NVDEC'}`;
    hudFpsVal.textContent = state.fps.toFixed(1);
    hudLatencyVal.textContent = `${state.latency_ms.toFixed(1)} ms`;

    // Viewport Overlays
    const activeCount = state.active_tracks_count || 0;
    hudActivePhones.textContent = `${activeCount} Phone${activeCount === 1 ? '' : 's'} Active`;

    if (state.active_track_ids && state.active_track_ids.length > 0) {
      const idsStr = state.active_track_ids.map(id => `#${id}`).join(', ');
      hudTrackIds.textContent = `Tracks: ${idsStr}`;
    } else {
      hudTrackIds.textContent = 'Tracks: None';
    }

    // Timeline Progress Bar
    const pct = state.percent || 0.0;
    timelineFill.style.width = `${pct}%`;
    timelinePercentText.textContent = `${pct.toFixed(1)}%`;
    timelineFrameText.textContent = `Frame: ${state.current_frame || 0} / ${state.total_frames || 0}`;

    // Sidebar Metric Cards
    metricFps.textContent = state.fps.toFixed(1);
    metricLatency.textContent = state.latency_ms.toFixed(1);
    metricActivePhones.textContent = activeCount;
    metricUniqueTracks.textContent = state.total_unique_tracks || 0;

    // Button states based on pipeline status
    if (state.state === 'processing') {
      btnStart.disabled = true;
      btnPause.disabled = false;
      btnStop.disabled = false;
      btnDownload.disabled = true;
    } else if (state.state === 'completed') {
      btnStart.disabled = false;
      btnPause.disabled = true;
      btnStop.disabled = true;
      btnDownload.disabled = !state.output_ready;
      
      // Reduce polling rate when completed
      startStatusPolling(1500);
    } else if (state.state === 'idle') {
      btnStart.disabled = false;
      btnPause.disabled = true;
      btnStop.disabled = true;
      btnDownload.disabled = !state.output_ready;
    }
  }
});
