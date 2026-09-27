const textarea = document.getElementById("review-text");
const charCount = document.getElementById("char-count");
const analyzeBtn = document.getElementById("analyze-btn");
const errorMsg = document.getElementById("error-msg");

const resultEmpty = document.getElementById("result-empty");
const resultFilled = document.getElementById("result-filled");
const resultLabel = document.getElementById("result-label");
const resultConfidence = document.getElementById("result-confidence");
const breakdown = document.getElementById("breakdown");
const cleanedText = document.getElementById("cleaned-text");

textarea.addEventListener("input", () => {
  charCount.textContent = `${textarea.value.length} characters`;
});

analyzeBtn.addEventListener("click", async () => {
  const text = textarea.value.trim();
  errorMsg.textContent = "";

  if (!text) {
    errorMsg.textContent = "Please enter some text to analyze.";
    return;
  }

  analyzeBtn.disabled = true;
  analyzeBtn.textContent = "Analyzing...";

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });

    const data = await response.json();

    if (!response.ok) {
      errorMsg.textContent = data.error || "Something went wrong.";
      return;
    }

    renderResult(data);
  } catch (err) {
    errorMsg.textContent = "Could not reach the server. Is app.py running?";
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = "Analyze sentiment";
  }
});

function renderResult(data) {
  resultEmpty.hidden = true;
  resultFilled.hidden = false;

  resultLabel.textContent = data.sentiment;
  resultLabel.className = `result-label ${data.sentiment}`;
  resultConfidence.textContent = `${data.confidence}% confident`;

  const order = ["Positive", "Neutral", "Negative"];
  breakdown.innerHTML = "";
  order.forEach((label) => {
    const pct = data.breakdown[label] ?? 0;
    const row = document.createElement("div");
    row.className = "breakdown-row";
    row.innerHTML = `
      <span>${label}</span>
      <span class="breakdown-track">
        <span class="breakdown-fill ${label}" style="width:${pct}%"></span>
      </span>
      <span>${pct}%</span>
    `;
    breakdown.appendChild(row);
  });

  cleanedText.textContent = data.cleaned_text || "(nothing left after cleaning)";
}
