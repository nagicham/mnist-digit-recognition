const canvas = document.getElementById('drawCanvas');
const ctx = canvas.getContext('2d');
const clearBtn = document.getElementById('clearBtn');
const predictBtn = document.getElementById('predictBtn');
const resultContent = document.getElementById('resultContent');

let drawing = false;
let lastX = 0;
let lastY = 0;

function initCanvas() {
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.lineWidth = 18;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.strokeStyle = '#111111';
}
initCanvas();

function getPos(evt) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const point = evt.touches ? evt.touches[0] : evt;
    return {
	x: (point.clientX - rect.left) * scaleX,
	y: (point.clientY - rect.top) * scaleY,
    };
}

function startDraw(evt) {
    evt.preventDefault();
    drawing = true;
    const pos = getPos(evt);
    lastX = pos.x;
    lastY = pos.y;
}

function moveDraw(evt) {
    if (!drawing) return;
    evt.preventDefault();
    const pos = getPos(evt);
    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(pos.x, pos.y);
    ctx.stroke();
    lastX = pos.x;
    lastY = pos.y;
}

function endDraw() {
    drawing = false;
}

canvas.addEventListener('mousedown', startDraw);
canvas.addEventListener('mousemove', moveDraw);
window.addEventListener('mouseup', endDraw);

canvas.addEventListener('touchstart', startDraw);
canvas.addEventListener('touchmove', moveDraw);
window.addEventListener('touchend', endDraw);

clearBtn.addEventListener('click', () => {
    initCanvas();
    resultContent.innerHTML = '<p class="empty-state">Push "Predict"</p>';
});

function renderResult(data) {
    const top = data.predicted_digit;
    const rows = data.probabilities.map((p, digit) => {
	const pct = Math.round(p * 1000) / 10;
	const isTop = digit == top;
	return `
<div class="prob-row ${isTop ? 'top' : '' }">
<span>${digit}</span>
<span class="prob-bar-track"><span class="prob-bar-fill" style="width:${pct}%"></span></span>
<span>${pct.toFixed(1)}%</span>
</div>
`;
    }).join('');

    resultContent.innerHTML = `
<div class="digit-display">${top}</div>
<div class="confidence-line">confidence: <strong>${(data.confidence * 100).toFixed(1)}%</strong></div>
${rows}
`;
}

function renderError(message) {
    resultContent.innerHTML = '<p class="error-message">${message}</p>';
}

predictBtn.addEventListener('click', () => {
    predictBtn.disabled = true;
    predictBtn.textContent = 'Predicting...';

    canvas.toBlob(async (blob) => {
	try {
	    const formData = new FormData();
	    formData.append('file', blob, 'digit.png');

	    const response = await fetch('/predict', {
		method: 'POST',
		body: formData,
	    });

	    if (!response.ok) {
		const errBody = await response.json().catch(() => ({}));
		renderError(errBody.detail || 'Error (status ${response.status})');
		return;
	    }

	    const data = await response.json();
	    renderResult(data);
	}
	catch (err) {
	    renderError('An error has occured. Check the server.');
	} finally {
	    predictBtn.disabled = false;
	    predictBtn.textContent = 'Predict';
	}
    }, 'image/png');
});
