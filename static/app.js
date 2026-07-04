(() => {
  const fileInput = document.getElementById("file-input");
  const dropZone = document.getElementById("drop-zone");
  const selectedFile = document.getElementById("selected-file");
  const startBtn = document.getElementById("start-btn");

  const uploadCard = document.getElementById("upload-card");
  const tuneCard = document.getElementById("tune-card");
  const progressCard = document.getElementById("progress-card");
  const errorCard = document.getElementById("error-card");
  const resultCard = document.getElementById("result-card");

  const progressFill = document.getElementById("progress-fill");
  const stageItems = [...document.querySelectorAll("#stage-list li")];
  const errorMessage = document.getElementById("error-message");
  const retryBtn = document.getElementById("retry-btn");
  const newBtn = document.getElementById("new-btn");

  const cleanAudio = document.getElementById("clean-audio");
  const originalAudio = document.getElementById("original-audio");
  const downloadWav = document.getElementById("download-wav");
  const downloadMp3 = document.getElementById("download-mp3");
  const downloadTxt = document.getElementById("download-txt");
  const downloadSrt = document.getElementById("download-srt");
  const transcriptBlock = document.getElementById("transcript-block");
  const transcriptText = document.getElementById("transcript-text");

  // --- Tune panel elements ---
  const tuneAudio = document.getElementById("tune-audio");
  const renderedBlock = document.getElementById("rendered-block");
  const renderedAudio = document.getElementById("rendered-audio");
  const previewBtn = document.getElementById("preview-btn");
  const resetBtn = document.getElementById("reset-btn");
  const previewStatus = document.getElementById("preview-status");
  const tuneStartBtn = document.getElementById("tune-start-btn");

  const DEFAULTS = {
    noise_reduction: 55,
    low_cut_hz: 90,
    high_cut_hz: 7500,
    eq_freq: 2500,
    eq_gain: 0,
    notch_freq: 0,
    notch_db: -18,
    vocal_boost: 100,
    compression: 42,
    gain_db: 0,
    gate_threshold: -60,
    df_strength: 100,
    use_ai_denoise: true,
    df_postfilter: false,
    use_transcription: true,
  };

  const sliders = {
    noise_reduction: document.getElementById("s-noise"),
    low_cut_hz: document.getElementById("s-lowcut"),
    high_cut_hz: document.getElementById("s-highcut"),
    eq_freq: document.getElementById("s-eqfreq"),
    eq_gain: document.getElementById("s-eqgain"),
    notch_freq: document.getElementById("s-notchfreq"),
    notch_db: document.getElementById("s-notchdb"),
    vocal_boost: document.getElementById("s-boost"),
    compression: document.getElementById("s-comp"),
    gain_db: document.getElementById("s-gain"),
    gate_threshold: document.getElementById("s-gate"),
    df_strength: document.getElementById("s-dfstrength"),
  };
  const sliderVals = {
    noise_reduction: document.getElementById("s-noise-val"),
    low_cut_hz: document.getElementById("s-lowcut-val"),
    high_cut_hz: document.getElementById("s-highcut-val"),
    eq_freq: document.getElementById("s-eqfreq-val"),
    eq_gain: document.getElementById("s-eqgain-val"),
    notch_freq: document.getElementById("s-notchfreq-val"),
    notch_db: document.getElementById("s-notchdb-val"),
    vocal_boost: document.getElementById("s-boost-val"),
    compression: document.getElementById("s-comp-val"),
    gain_db: document.getElementById("s-gain-val"),
    gate_threshold: document.getElementById("s-gate-val"),
    df_strength: document.getElementById("s-dfstrength-val"),
  };
  const aiToggle = document.getElementById("s-ai");
  const dfPostfilterToggle = document.getElementById("s-dfpf");
  const transcribeToggle = document.getElementById("s-transcribe");

  let chosenFile = null;
  let pollTimer = null;
  let fileObjectUrl = null;

  function currentParams() {
    return {
      noise_reduction: Number(sliders.noise_reduction.value),
      low_cut_hz: Number(sliders.low_cut_hz.value),
      high_cut_hz: Number(sliders.high_cut_hz.value),
      eq_freq: Number(sliders.eq_freq.value),
      eq_gain: Number(sliders.eq_gain.value),
      notch_freq: Number(sliders.notch_freq.value),
      notch_db: Number(sliders.notch_db.value),
      vocal_boost: Number(sliders.vocal_boost.value),
      compression: Number(sliders.compression.value),
      gain_db: Number(sliders.gain_db.value),
      gate_threshold: Number(sliders.gate_threshold.value),
      df_strength: Number(sliders.df_strength.value),
      use_ai_denoise: aiToggle.checked,
      df_postfilter: dfPostfilterToggle.checked,
      use_transcription: transcribeToggle.checked,
    };
  }

  function applyDefaultsToControls() {
    for (const key of Object.keys(sliders)) {
      sliders[key].value = DEFAULTS[key];
      sliderVals[key].textContent = DEFAULTS[key];
    }
    aiToggle.checked = DEFAULTS.use_ai_denoise;
    dfPostfilterToggle.checked = DEFAULTS.df_postfilter;
    transcribeToggle.checked = DEFAULTS.use_transcription;
  }

  // --- Live rough preview via Web Audio API ---
  // Instant tone/gain/gate-ish feedback while dragging sliders, using the
  // locally chosen file -- no server round-trip. This is an approximation
  // (no real spectral noise reduction happens client-side); "Render accurate
  // preview" gets the real ffmpeg + DeepFilterNet result for a short clip.
  let audioCtx = null;
  let graphReady = false;
  let sourceNode = null;
  let noiseNode = null; // AudioWorklet gate + expander (may be null if unsupported)
  let highpassNode = null;
  let lowpassNode = null;
  let notchNode = null;
  let eqNode = null;
  let presenceNode = null;
  let compressorNode = null;
  let gainNode = null;

  async function ensureLiveGraph() {
    if (graphReady) return;
    const Ctx = window.AudioContext || window.webkitAudioContext;
    if (!Ctx) return; // unsupported browser: live preview just won't run
    if (!audioCtx) audioCtx = new Ctx();

    // Try to load the noise gate / expander worklet (needs a secure context).
    if (audioCtx.audioWorklet) {
      try {
        await audioCtx.audioWorklet.addModule("/noise-worklet.js");
        noiseNode = new AudioWorkletNode(audioCtx, "noise-reducer");
      } catch (err) {
        noiseNode = null; // fall back to no gate/expander in the live preview
      }
    }

    // createMediaElementSource can only be called once per element.
    if (!sourceNode) sourceNode = audioCtx.createMediaElementSource(tuneAudio);
    highpassNode = audioCtx.createBiquadFilter();
    highpassNode.type = "highpass";
    lowpassNode = audioCtx.createBiquadFilter();
    lowpassNode.type = "lowpass";
    // A narrow peaking cut (not a fixed "notch" type) so its depth is tunable,
    // matching the server-side ffmpeg equalizer notch.
    notchNode = audioCtx.createBiquadFilter();
    notchNode.type = "peaking";
    notchNode.frequency.value = 1000;
    notchNode.Q.value = 8;
    eqNode = audioCtx.createBiquadFilter();
    eqNode.type = "peaking";
    eqNode.frequency.value = 2500;
    eqNode.Q.value = 1.4;
    presenceNode = audioCtx.createBiquadFilter();
    presenceNode.type = "peaking";
    presenceNode.frequency.value = 2500;
    presenceNode.Q.value = 1;
    compressorNode = audioCtx.createDynamicsCompressor();
    gainNode = audioCtx.createGain();

    let head = sourceNode;
    if (noiseNode) head = head.connect(noiseNode);
    head
      .connect(highpassNode)
      .connect(lowpassNode)
      .connect(notchNode)
      .connect(eqNode)
      .connect(presenceNode)
      .connect(compressorNode)
      .connect(gainNode)
      .connect(audioCtx.destination);

    graphReady = true;
    updateLiveGraph();
  }

  function updateLiveGraph() {
    if (!graphReady || !audioCtx) return;
    const p = currentParams();
    const now = audioCtx.currentTime;
    highpassNode.frequency.setTargetAtTime(p.low_cut_hz, now, 0.02);
    lowpassNode.frequency.setTargetAtTime(p.high_cut_hz, now, 0.02);
    eqNode.frequency.setTargetAtTime(p.eq_freq, now, 0.02);
    eqNode.gain.setTargetAtTime(p.eq_gain, now, 0.02);
    if (p.notch_freq > 0) {
      notchNode.frequency.setTargetAtTime(p.notch_freq, now, 0.02);
      notchNode.gain.setTargetAtTime(p.notch_db, now, 0.02);
    } else {
      notchNode.gain.setTargetAtTime(0, now, 0.02); // off
    }
    presenceNode.gain.setTargetAtTime((p.vocal_boost / 100) * 12, now, 0.02);
    compressorNode.threshold.setTargetAtTime(-24 - (p.compression / 100) * 20, now, 0.02);
    compressorNode.ratio.setTargetAtTime(1 + (p.compression / 100) * 11, now, 0.02);
    gainNode.gain.setTargetAtTime(Math.pow(10, p.gain_db / 20), now, 0.02);
    if (noiseNode) {
      // DeepFilterNet can't run in the browser, so when "AI noise removal" is
      // on we drive the live gate/expander a bit harder as a rough stand-in so
      // the toggle audibly does something; the real AI denoise is applied in
      // Render accurate preview and the full run.
      const aiBump = aiToggle.checked ? 0.2 : 0.0;
      const reduction = Math.min(1, p.noise_reduction / 100 + aiBump);
      noiseNode.parameters.get("gateThreshold").setValueAtTime(p.gate_threshold, now);
      noiseNode.parameters.get("reduction").setValueAtTime(reduction, now);
    }
  }

  function formatBytes(bytes) {
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }

  function pickFile(file) {
    if (!file) return;
    chosenFile = file;
    selectedFile.hidden = false;
    selectedFile.textContent = `${file.name} (${formatBytes(file.size)})`;
    startBtn.disabled = false;

    if (fileObjectUrl) URL.revokeObjectURL(fileObjectUrl);
    fileObjectUrl = URL.createObjectURL(file);
    tuneAudio.src = fileObjectUrl;

    renderedBlock.hidden = true;
    previewStatus.textContent = "";
    tuneCard.hidden = false;
  }

  fileInput.addEventListener("change", () => pickFile(fileInput.files[0]));

  ["dragover", "dragenter"].forEach((evt) =>
    dropZone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropZone.classList.add("dragover");
    })
  );
  ["dragleave", "drop"].forEach((evt) =>
    dropZone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropZone.classList.remove("dragover");
    })
  );
  dropZone.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files && e.dataTransfer.files[0];
    pickFile(file);
  });

  tuneAudio.addEventListener("play", async () => {
    await ensureLiveGraph();
    if (audioCtx && audioCtx.state === "suspended") audioCtx.resume();
  });

  for (const key of Object.keys(sliders)) {
    sliders[key].addEventListener("input", () => {
      sliderVals[key].textContent = sliders[key].value;
      updateLiveGraph();
    });
  }
  // The AI-denoise toggle nudges the live gate/expander (a rough stand-in for
  // DeepFilterNet, which can't run in the browser), so refresh on change.
  aiToggle.addEventListener("change", updateLiveGraph);

  startBtn.addEventListener("click", () => {
    if (chosenFile) startJob(chosenFile, DEFAULTS);
  });
  tuneStartBtn.addEventListener("click", () => {
    if (chosenFile) startJob(chosenFile, currentParams());
  });
  resetBtn.addEventListener("click", () => {
    applyDefaultsToControls();
    updateLiveGraph();
  });
  previewBtn.addEventListener("click", () => {
    if (chosenFile) renderAccuratePreview(chosenFile);
  });

  retryBtn.addEventListener("click", resetToUpload);
  newBtn.addEventListener("click", resetToUpload);

  function resetToUpload() {
    chosenFile = null;
    fileInput.value = "";
    selectedFile.hidden = true;
    startBtn.disabled = true;
    tuneCard.hidden = true;
    if (pollTimer) clearTimeout(pollTimer);
    showOnly(uploadCard);
  }

  function showOnly(card) {
    for (const c of [uploadCard, progressCard, errorCard, resultCard]) {
      c.hidden = c !== card;
    }
    if (card !== uploadCard) tuneCard.hidden = true;
  }

  function setStage(stageName) {
    let seenActive = false;
    for (const li of stageItems) {
      const name = li.dataset.stage;
      li.classList.remove("active", "done");
      if (name === stageName) {
        li.classList.add("active");
        seenActive = true;
      } else if (!seenActive) {
        li.classList.add("done");
      }
    }
  }

  function paramsToFormFields(form, params) {
    for (const [k, v] of Object.entries(params)) {
      form.append(k, typeof v === "boolean" ? String(v) : v);
    }
  }

  async function renderAccuratePreview(file) {
    previewBtn.disabled = true;
    previewStatus.textContent = "Rendering real preview… this can take a little while.";

    const form = new FormData();
    form.append("file", file);
    paramsToFormFields(form, currentParams());

    try {
      const resp = await fetch("/api/preview", { method: "POST", body: form });
      if (!resp.ok) {
        const detail = await safeDetail(resp);
        previewStatus.textContent = detail || "Preview render failed.";
        return;
      }
      const blob = await resp.blob();
      renderedAudio.src = URL.createObjectURL(blob);
      renderedBlock.hidden = false;
      previewStatus.textContent = "Preview ready — press play above.";
      renderedAudio.play().catch(() => {});
    } catch {
      previewStatus.textContent = "Preview failed — check your connection and try again.";
    } finally {
      previewBtn.disabled = false;
    }
  }

  async function startJob(file, params) {
    showOnly(progressCard);
    setStage("extracting audio");
    progressFill.style.width = "0%";

    const form = new FormData();
    form.append("file", file);
    paramsToFormFields(form, params);

    let resp;
    try {
      resp = await fetch("/api/jobs", { method: "POST", body: form });
    } catch (err) {
      return showError("Upload failed — check your connection and try again.");
    }

    if (!resp.ok) {
      const detail = await safeDetail(resp);
      return showError(detail || "Upload was rejected by the server.");
    }

    const job = await resp.json();
    poll(job.id);
  }

  async function safeDetail(resp) {
    try {
      const data = await resp.json();
      return data.detail;
    } catch {
      return null;
    }
  }

  async function poll(jobId) {
    let job;
    try {
      const resp = await fetch(`/api/jobs/${jobId}`);
      if (!resp.ok) {
        return showError("Lost track of that job — please try again.");
      }
      job = await resp.json();
    } catch {
      pollTimer = setTimeout(() => poll(jobId), 2000);
      return;
    }

    if (job.status === "error") {
      return showError(job.error || "Processing failed.");
    }

    if (job.status === "done") {
      return showResult(jobId, job);
    }

    setStage(job.status);
    const overallStageIndex = Math.max(
      0,
      stageItems.findIndex((li) => li.dataset.stage === job.status)
    );
    const overallPct =
      ((overallStageIndex + (job.progress || 0)) / stageItems.length) * 100;
    progressFill.style.width = `${Math.min(99, overallPct).toFixed(0)}%`;

    pollTimer = setTimeout(() => poll(jobId), 1500);
  }

  function showResult(jobId, job) {
    progressFill.style.width = "100%";
    const wavUrl = `/api/jobs/${jobId}/download/voice_clean.wav`;
    const mp3Url = `/api/jobs/${jobId}/download/voice_clean.mp3`;
    const originalUrl = `/api/jobs/${jobId}/download/original.wav`;

    cleanAudio.src = mp3Url;
    originalAudio.src = originalUrl;
    downloadWav.href = wavUrl;
    downloadMp3.href = mp3Url;

    const downloads = job.downloads || [];
    const hasTxt = downloads.includes("transcript.txt");
    const hasSrt = downloads.includes("captions.srt");

    if (job.transcript) {
      transcriptText.textContent = job.transcript;
      transcriptBlock.hidden = false;
    } else {
      transcriptBlock.hidden = true;
    }

    if (hasTxt) {
      downloadTxt.href = `/api/jobs/${jobId}/download/transcript.txt`;
      downloadTxt.hidden = false;
    } else {
      downloadTxt.hidden = true;
    }
    if (hasSrt) {
      downloadSrt.href = `/api/jobs/${jobId}/download/captions.srt`;
      downloadSrt.hidden = false;
    } else {
      downloadSrt.hidden = true;
    }

    showOnly(resultCard);
  }

  function showError(message) {
    errorMessage.textContent = message;
    showOnly(errorCard);
  }
})();
