(() => {
  const fileInput = document.getElementById("file-input");
  const dropZone = document.getElementById("drop-zone");
  const selectedFile = document.getElementById("selected-file");
  const startBtn = document.getElementById("start-btn");

  const uploadCard = document.getElementById("upload-card");
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

  let chosenFile = null;
  let pollTimer = null;

  function showOnly(card) {
    for (const c of [uploadCard, progressCard, errorCard, resultCard]) {
      c.hidden = c !== card;
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

  startBtn.addEventListener("click", () => {
    if (chosenFile) startJob(chosenFile);
  });
  retryBtn.addEventListener("click", resetToUpload);
  newBtn.addEventListener("click", resetToUpload);

  function resetToUpload() {
    chosenFile = null;
    fileInput.value = "";
    selectedFile.hidden = true;
    startBtn.disabled = true;
    if (pollTimer) clearTimeout(pollTimer);
    showOnly(uploadCard);
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

  async function startJob(file) {
    showOnly(progressCard);
    setStage("extracting audio");
    progressFill.style.width = "0%";

    const form = new FormData();
    form.append("file", file);

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
