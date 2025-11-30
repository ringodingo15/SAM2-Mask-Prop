let jobId = null;
let frames = [];
let masks = [];
let currentIdx = 0;

const jobInfo = document.getElementById('job-info');
const framesStatus = document.getElementById('frames-status');
const lsStatus = document.getElementById('ls-status');
const progressDiv = document.getElementById('progress');
const frameImg = document.getElementById('frame-img');
const maskImg = document.getElementById('mask-img');
const frameIdxText = document.getElementById('frame-idx');

document.getElementById('btn-new-job').addEventListener('click', async () => {
  const res = await fetch('/api/new_job', { method: 'POST' });
  const data = await res.json();
  jobId = data.job_id;
  jobInfo.textContent = `Job ID: ${jobId}`;
  frames = [];
  masks = [];
  currentIdx = 0;
  updateViewer();
});

document.getElementById('btn-upload-video').addEventListener('click', async () => {
  if (!jobId) return alert('Create a job first.');
  const file = document.getElementById('video-file').files[0];
  if (!file) return alert('Select a video.');
  const form = new FormData();
  form.append('job_id', jobId);
  form.append('file', file);
  framesStatus.textContent = 'Uploading...';
  const res = await fetch('/api/upload_video', { method: 'POST', body: form });
  const data = await res.json();
  if (res.ok) {
    framesStatus.textContent = `Uploaded. Frame count: ${data.frame_count}`;
    await refreshFrames();
  } else {
    framesStatus.textContent = `Error: ${data.detail || 'Upload failed'}`;
  }
});

document.getElementById('btn-upload-frames').addEventListener('click', async () => {
  if (!jobId) return alert('Create a job first.');
  const file = document.getElementById('frames-zip').files[0];
  if (!file) return alert('Select a frames ZIP.');
  const form = new FormData();
  form.append('job_id', jobId);
  form.append('file', file);
  framesStatus.textContent = 'Uploading ZIP...';
  const res = await fetch('/api/upload_frames_zip', { method: 'POST', body: form });
  const data = await res.json();
  if (res.ok) {
    framesStatus.textContent = `Frames extracted. Count: ${data.frame_count}`;
    await refreshFrames();
  } else {
    framesStatus.textContent = `Error: ${data.detail || 'Upload failed'}`;
  }
});

document.getElementById('btn-upload-ls').addEventListener('click', async () => {
  if (!jobId) return alert('Create a job first.');
  const file = document.getElementById('ls-json').files[0];
  if (!file) return alert('Select a JSON export.');
  const form = new FormData();
  form.append('job_id', jobId);
  form.append('file', file);
  lsStatus.textContent = 'Uploading Label Studio JSON...';
  const res = await fetch('/api/upload_labelstudio', { method: 'POST', body: form });
  const data = await res.json();
  if (res.ok) {
    lsStatus.textContent = 'Label Studio export uploaded.';
  } else {
    lsStatus.textContent = `Error: ${data.detail || 'Upload failed'}`;
  }
});

document.getElementById('btn-propagate').addEventListener('click', async () => {
  if (!jobId) return alert('Create a job first.');
  progressDiv.textContent = 'Starting...';
  const form = new FormData();
  form.append('job_id', jobId);
  form.append('labels_mode', document.getElementById('labels-mode').value);
  const res = await fetch('/api/propagate', { method: 'POST', body: form });
  const data = await res.json();
  if (!res.ok) {
    progressDiv.textContent = `Error: ${data.detail || 'Failed to start propagation.'}`;
    return;
  }
  pollStatus();
});

document.getElementById('prev-frame').addEventListener('click', () => {
  if (frames.length === 0) return;
  currentIdx = Math.max(0, currentIdx - 1);
  updateViewer();
});

document.getElementById('next-frame').addEventListener('click', () => {
  if (frames.length === 0) return;
  currentIdx = Math.min(frames.length - 1, currentIdx + 1);
  updateViewer();
});

document.getElementById('btn-export').addEventListener('click', async () => {
  if (!jobId) return alert('Create a job first.');
  const url = `/api/export/${jobId}`;
  window.location.href = url;
});

async function refreshFrames() {
  const res = await fetch(`/api/frames/${jobId}/list`);
  const data = await res.json();
  frames = data.frames || [];
  currentIdx = 0;
  updateViewer();
}

async function refreshMasks() {
  const res = await fetch(`/api/masks/${jobId}/list`);
  const data = await res.json();
  masks = data.masks || [];
  updateViewer();
}

function updateViewer() {
  frameIdxText.textContent = `${frames.length ? (currentIdx + 1) : 0} / ${frames.length}`;
  frameImg.src = frames[currentIdx] || '';
  maskImg.src = masks[currentIdx] || '';
}

async function pollStatus() {
  if (!jobId) return;
  let done = false;
  while (!done) {
    await new Promise(r => setTimeout(r, 1000));
    const res = await fetch(`/api/status/${jobId}`);
    if (!res.ok) break;
    const s = await res.json();
    progressDiv.textContent = `${s.status} - ${s.progress}% ${s.message ? '(' + s.message + ')' : ''}`;
    if (s.status === 'completed') {
      done = true;
      await refreshMasks();
    } else if (s.status === 'failed') {
      done = true;
    }
  }
}