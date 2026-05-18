document.addEventListener('DOMContentLoaded', () => {
    // 🚀 BACKEND API CONFIGURATION
    // Change this to your deployed Render/Railway backend URL once deployed (e.g., "https://your-backend.onrender.com")
    // Keep it as an empty string "" to use the local FastAPI server during development.
    const API_BASE_URL = "https://saged00-titanic-predictor-api.hf.space";

    const form = document.getElementById('prediction-form');
    const ageInput = document.getElementById('age');
    const ageVal = document.getElementById('age-val');
    const fareInput = document.getElementById('fare');
    const fareVal = document.getElementById('fare-val');
    const predictBtn = document.getElementById('predict-btn');
    const resultContainer = document.getElementById('result-container');
    const predictorCard = document.getElementById('predictor-card');
    const resetBtn = document.getElementById('reset-btn');
    const toast = document.getElementById('toast');

    ageInput.addEventListener('input', (e) => {
        ageVal.textContent = e.target.value;
    });

    fareInput.addEventListener('input', (e) => {
        fareVal.textContent = parseFloat(e.target.value).toFixed(2);
    });

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        const payload = {
            pclass: parseInt(form.pclass.value, 10),
            sex: form.sex.value,
            age: parseFloat(ageInput.value),
            title: form.title.value,
            family_size: parseInt(form.family_size.value, 10),
            fare: parseFloat(fareInput.value),
            embarked: form.embarked.value,
        };

        if (!payload.pclass || !payload.title || !payload.embarked) {
            showToast('Please fill in all fields.');
            return;
        }

        predictBtn.classList.add('loading');
        predictBtn.disabled = true;

        try {
            const res = await fetch(`${API_BASE_URL}/predict`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });

            const body = await res.json().catch(() => ({}));

            if (!res.ok) {
                throw new Error(formatApiError(body.detail) || 'Prediction failed');
            }

            displayResult(body);
        } catch (err) {
            showToast(err.message || 'Something went wrong. Try again.');
        } finally {
            predictBtn.classList.remove('loading');
            predictBtn.disabled = false;
        }
    });

    resetBtn.addEventListener('click', () => {
        resultContainer.classList.add('hidden');
        predictorCard.classList.remove('hidden');
        form.reset();
        syncSliders();
        document.getElementById('prob-bar-fill').style.width = '0%';
        document.getElementById('input-summary').innerHTML = '';
    });

    function syncSliders() {
        ageVal.textContent = ageInput.value;
        fareVal.textContent = parseFloat(fareInput.value).toFixed(2);
    }

    function formatApiError(detail) {
        if (!detail) return '';
        if (typeof detail === 'string') return detail;
        if (Array.isArray(detail)) {
            return detail.map((d) => d.msg || String(d)).join(' · ');
        }
        return String(detail);
    }

    function showToast(message) {
        toast.textContent = message;
        toast.classList.add('show');
        clearTimeout(showToast._timer);
        showToast._timer = setTimeout(() => toast.classList.remove('show'), 4000);
    }

    function displayResult(result) {
        const survived = result.survived;
        document.getElementById('result-status').textContent =
            result.survived_label || (survived ? 'Survived' : 'Did not survive');
        document.getElementById('result-icon').textContent = survived ? '✓' : '✕';
        document.getElementById('result-icon').className =
            'status-icon ' + (survived ? 'status-survived' : 'status-died');

        const pct = (result.probability * 100).toFixed(1);
        document.getElementById('prob-value').textContent = pct + '%';

        const probBar = document.getElementById('prob-bar-fill');
        probBar.style.width = '0%';
        probBar.style.background = survived
            ? 'linear-gradient(90deg, #10b981, #34d399)'
            : 'linear-gradient(90deg, #ef4444, #f87171)';
        requestAnimationFrame(() => {
            probBar.style.width = pct + '%';
        });

        const s = result.input_summary;
        if (s) {
            const port = { S: 'Southampton', C: 'Cherbourg', Q: 'Queenstown' }[s.embarked] || s.embarked;
            const row = (label, value) =>
                '<div class="summary-item"><span class="summary-label">' +
                label +
                '</span><span class="summary-value">' +
                value +
                '</span></div>';
            document.getElementById('input-summary').innerHTML =
                '<div class="summary-grid">' +
                row('Class', s.pclass) +
                row('Gender', s.sex) +
                row('Age', s.age) +
                row('Title', s.title) +
                row('Family', s.family_size) +
                row('Port', port) +
                '</div>';
        }

        predictorCard.classList.add('hidden');
        resultContainer.classList.remove('hidden');
    }
});
