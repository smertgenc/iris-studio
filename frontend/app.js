"use strict";

(() => {
  const keys = ["sepal_length", "sepal_width", "petal_length", "petal_width"];
  const labels = ["Çanak yaprak uzunluğu", "Çanak yaprak genişliği", "Taç yaprak uzunluğu", "Taç yaprak genişliği"];
  const species = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"];
  const colors = ["#9278b1", "#98a981", "#d0b27c"];
  const presets = [[5.1, 3.5, 1.4, 0.2], [6.0, 2.9, 4.5, 1.5], [6.5, 3.0, 5.8, 2.2]];
  const storageKey = "iris-studio.history.v1";
  const $ = (id) => document.getElementById(id);
  const nf = new Intl.NumberFormat("tr-TR", { minimumFractionDigits: 1, maximumFractionDigits: 1 });
  let pendingTimer;
  let toastTimer;
  let activeController;
  let sequence = 0;
  let lastPrediction = null;
  let history = [];
  let storageNoticeShown = false;

  function node(tag, className, text) {
    const el = document.createElement(tag);
    if (className) el.className = className;
    if (text !== undefined) el.textContent = text;
    return el;
  }

  function notify(message) {
    clearTimeout(toastTimer);
    $("toast").textContent = message;
    $("toast").hidden = false;
    toastTimer = setTimeout(() => { $("toast").hidden = true; }, 3500);
  }

  function setStatus(ready, text) {
    $("connection-status").classList.toggle("offline", !ready);
    $("status-text").textContent = text || (ready ? "Model hazır · PyTorch" : "Bağlantı yok · tekrar dene");
  }

  function showError(message = "") {
    $("form-error").textContent = message;
    $("form-error").hidden = !message;
  }

  function readMeasurements() {
    const values = {};
    for (let i = 0; i < keys.length; i++) {
      const input = $(keys[i]);
      const value = input.valueAsNumber;
      const valid = Number.isFinite(value) && input.checkValidity();
      input.setAttribute("aria-invalid", String(!valid));
      if (!valid) {
        return { error: `${labels[i]} için ${nf.format(Number(input.min))}–${nf.format(Number(input.max))} cm arasında, 0,1 adımlı bir sayı gir.` };
      }
      values[keys[i]] = value;
    }
    return { values };
  }

  function paintRange(key) {
    const range = $(`${key}_range`);
    const fill = ((Number(range.value) - Number(range.min)) / (Number(range.max) - Number(range.min))) * 100;
    range.style.setProperty("--fill", `${fill}%`);
  }

  function renderIllustration(values) {
    const sepal = `${nf.format(values.sepal_length)} × ${nf.format(values.sepal_width)} `;
    const petal = `${nf.format(values.petal_length)} × ${nf.format(values.petal_width)} `;
    $("sepal-callout").replaceChildren(document.createTextNode(sepal), node("small", "", "cm"));
    $("petal-callout").replaceChildren(document.createTextNode(petal), node("small", "", "cm"));
    // Temsili çizim: bir biyolojik ölçüm veya birebir ölçekli geometri değildir.
    const petalY = 0.88 + (values.petal_length / 10) * 0.28;
    const petalX = 0.92 + (values.petal_width / 4) * 0.18;
    const sepalY = 0.88 + (values.sepal_length / 10) * 0.18;
    const sepalX = 0.88 + (values.sepal_width / 6) * 0.2;
    $("petal-shape").style.transform = `scale(${petalX}, ${petalY})`;
    $("sepal-shape").style.transform = `scale(${sepalX}, ${sepalY})`;
    document.querySelectorAll("[data-preset]").forEach((button) => {
      const match = keys.every((key, i) => Math.abs(values[key] - presets[Number(button.dataset.preset)][i]) < 0.001);
      button.classList.toggle("selected", match);
      button.setAttribute("aria-pressed", String(match));
    });
  }

  function cancelPending() {
    clearTimeout(pendingTimer);
    if (activeController) activeController.abort();
    sequence += 1; // Önceki isteğin daha geç gelen sonucu yeni ölçümleri ezmesin.
    document.body.classList.remove("is-busy");
  }

  function markStale() {
    $("download-button").disabled = true;
    if (lastPrediction) $("result-content").classList.add("stale");
    $("predict-label").textContent = "Tahmini güncelle";
  }

  function onMeasurementChange() {
    cancelPending();
    markStale();
    const { values, error } = readMeasurements();
    showError(error);
    if (error) return;
    renderIllustration(values);
    if ($("live-toggle").checked) pendingTimer = setTimeout(runPrediction, 350);
  }

  for (const key of keys) {
    const number = $(key);
    const range = $(`${key}_range`);
    paintRange(key);
    range.addEventListener("input", () => {
      number.value = range.value;
      paintRange(key);
      onMeasurementChange();
    });
    number.addEventListener("input", () => {
      // Geçersiz girdiyi sessizce sınırlama veya varsayılanla değiştirme.
      if (number.value !== "" && number.checkValidity()) {
        range.value = number.value;
        paintRange(key);
      }
      onMeasurementChange();
    });
  }

  function applyValues(values) {
    keys.forEach((key, i) => {
      $(key).value = Array.isArray(values) ? values[i] : values[key];
      $(`${key}_range`).value = $(key).value;
      $(key).setAttribute("aria-invalid", "false");
      paintRange(key);
    });
    cancelPending();
    markStale();
    showError();
    const current = readMeasurements();
    if (current.values) renderIllustration(current.values);
    runPrediction();
  }

  function isPrediction(data) {
    return data && typeof data === "object" && species.includes(data.species)
      && Number.isFinite(data.probability) && data.probability >= 0 && data.probability <= 1
      && data.probabilities && species.every((s) => Number.isFinite(data.probabilities[s]) && data.probabilities[s] >= 0 && data.probabilities[s] <= 1)
      && data.measurements && keys.every((k) => Number.isFinite(data.measurements[k]))
      && Array.isArray(data.warnings) && data.warnings.every((w) => typeof w === "string")
      && Number.isFinite(data.inference_ms) && typeof data.model_version === "string"
      && typeof data.request_id === "string" && Number.isFinite(Date.parse(data.predicted_at));
  }

  function renderPrediction(data) {
    const index = species.indexOf(data.species);
    const name = data.species.replace("Iris-", "");
    const displayName = name[0].toUpperCase() + name.slice(1);
    $("result-content").classList.remove("stale");
    $("species-name").textContent = name;
    $("species-dot").style.background = colors[index];
    $("species-subtitle").textContent = "Iridaceae ailesi · Iris cinsi";
    $("score-value").textContent = nf.format(data.probability * 100);
    species.forEach((s, i) => {
      const p = data.probabilities[s] * 100;
      $(`prob-${i}`).textContent = p > 0 && p < 0.1 ? "<0,1%" : `${nf.format(p)}%`;
      $(`bar-${i}`).style.width = `${p}%`;
      document.querySelector(`[data-class="${s}"]`).classList.toggle("is-winner", i === index);
    });
    const ranked = [...species].sort((a, b) => data.probabilities[b] - data.probabilities[a]);
    const close = data.probabilities[ranked[0]] - data.probabilities[ranked[1]] < 0.2;
    $("species-description").textContent = close
      ? "İlk iki sınıfın tahmin payları birbirine yakın. Küçük ölçüm değişimleri sonucu değiştirebilir."
      : `Bu ölçümlerde modelin en yüksek tahmin payı ${displayName} sınıfında.`;
    $("inference-time").textContent = `${new Intl.NumberFormat("tr-TR", { maximumFractionDigits: 2 }).format(data.inference_ms)} ms`;
    $("inference-time").title = "Sunucudaki model hesaplama süresi. Ağ süresi dahil değildir.";
    $("model-version").textContent = data.model_version;
    $("prediction-warnings").replaceChildren(...data.warnings.map((message) => node("p", "", message)));
    $("prediction-warnings").hidden = data.warnings.length === 0;
    $("download-button").disabled = false;
    $("predict-label").textContent = "Yeniden keşfet";
  }

  async function runPrediction() {
    clearTimeout(pendingTimer);
    const { values, error } = readMeasurements();
    if (error) { showError(error); return; }
    showError();
    if (activeController) activeController.abort();
    const currentSequence = ++sequence;
    const controller = new AbortController();
    activeController = controller;
    const timeout = setTimeout(() => controller.abort("timeout"), 10000);
    document.body.classList.add("is-busy");
    $("predict-label").textContent = "Keşfediliyor…";
    $("download-button").disabled = true;
    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(values),
        signal: controller.signal,
      });
      if (!response.ok) {
        if (response.status === 422) throw new Error("Ölçümler geçersiz. Dört alanı ve izin verilen aralıkları kontrol et.");
        throw new Error("Model yanıt veremedi. FastAPI terminalini kontrol edip tekrar dene.");
      }
      const data = await response.json();
      if (currentSequence !== sequence) return;
      if (!isPrediction(data)) throw new Error("Sunucudan beklenmeyen bir yanıt geldi. API sürümünü kontrol et.");
      lastPrediction = data;
      renderPrediction(data);
      setStatus(true);
      remember(data);
    } catch (error) {
      if (currentSequence !== sequence) return;
      if (controller.signal.aborted && controller.signal.reason !== "timeout") return;
      const message = controller.signal.reason === "timeout"
        ? "Sunucu yanıtı zaman aşımına uğradı. Yeniden dene."
        : error instanceof TypeError
          ? "API'ye ulaşılamadı. FastAPI'yi çalıştır ve sayfayı http://127.0.0.1:8000 adresinden aç."
          : error.message;
      showError(message);
      setStatus(false);
      markStale();
      $("predict-label").textContent = "Tekrar dene";
      if (!lastPrediction) {
        $("species-name").textContent = "sonuç bekleniyor";
        $("species-subtitle").textContent = "Model bağlantısını kontrol et";
      }
    } finally {
      clearTimeout(timeout);
      if (currentSequence === sequence) {
        document.body.classList.remove("is-busy");
        activeController = null;
      }
    }
  }

  $("prediction-form").addEventListener("submit", (event) => { event.preventDefault(); runPrediction(); });
  $("live-toggle").addEventListener("change", () => {
    if ($("live-toggle").checked) runPrediction();
    else {
      cancelPending();
      $("predict-label").textContent = lastPrediction ? "Tahmini güncelle" : "Türü keşfet";
    }
  });
  document.querySelectorAll("[data-preset]").forEach((button) => {
    button.addEventListener("click", () => applyValues(presets[Number(button.dataset.preset)]));
  });
  $("reset-button").addEventListener("click", () => { applyValues(presets[0]); notify("Başlangıç ölçümlerine dönüldü."); });

  function persistHistory() {
    try { localStorage.setItem(storageKey, JSON.stringify(history)); }
    catch {
      if (!storageNoticeShown) { notify("Geçmiş, bu oturum boyunca kullanılabilir; tarayıcıya kaydedilemedi."); storageNoticeShown = true; }
    }
  }

  function remember(data) {
    // Aynı ölçümlerin peş peşe çalıştırılması defteri tekrarlarla doldurmasın.
    const same = history[0] && history[0].model_version === data.model_version
      && keys.every((k) => history[0].measurements[k] === data.measurements[k]);
    if (same) history[0] = data;
    else history.unshift(data);
    history = history.slice(0, 8);
    persistHistory();
    renderHistory();
  }

  function renderHistory() {
    $("history-count").textContent = history.length;
    $("clear-history").disabled = history.length === 0;
    const list = $("history-list");
    list.replaceChildren();
    if (!history.length) { list.append(node("p", "history-empty", "İlk keşfin burada görünecek.")); return; }
    history.forEach((record) => {
      const index = species.indexOf(record.species);
      const button = node("button", "history-item");
      button.type = "button";
      const top = node("span", "history-item-top");
      const name = node("span", "history-item-name");
      name.append(node("i", `dot dot-${index}`), document.createTextNode(record.species.replace("Iris-", "")));
      top.append(name, node("span", "history-item-score", `${nf.format(record.probability * 100)}%`));
      const values = keys.map((k) => nf.format(record.measurements[k])).join(" · ") + " cm";
      const date = new Date(record.predicted_at).toLocaleString("tr-TR", { day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit" });
      button.append(top, node("span", "history-item-values", values), node("span", "history-item-time", date));
      button.setAttribute("aria-label", `${record.species} ölçümlerini yeniden kullan: ${values}`);
      button.addEventListener("click", () => {
        applyValues(record.measurements);
        $("analysis").scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "instant" : "smooth", block: "start" });
        notify("Ölçümler yeniden analiz için yüklendi.");
      });
      list.append(button);
    });
  }

  try {
    const saved = JSON.parse(localStorage.getItem(storageKey) || "[]");
    if (Array.isArray(saved)) history = saved.filter(isPrediction).slice(0, 8);
  } catch { history = []; }
  renderHistory();
  $("clear-history").addEventListener("click", () => { history = []; persistHistory(); renderHistory(); notify("Analiz geçmişi temizlendi."); });

  $("download-button").addEventListener("click", () => {
    if (!lastPrediction || $("download-button").disabled) return;
    const blob = new Blob([JSON.stringify(lastPrediction, null, 2)], { type: "application/json;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const anchor = node("a");
    anchor.href = url;
    anchor.download = `iris-analiz-${lastPrediction.request_id}.json`;
    document.body.append(anchor);
    anchor.click();
    anchor.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    notify("Analiz sonucu JSON olarak indirildi.");
  });

  document.querySelectorAll("[data-open-model]").forEach((button) => button.addEventListener("click", () => $("model-dialog").showModal()));
  $("close-dialog").addEventListener("click", () => $("model-dialog").close());
  $("model-dialog").addEventListener("click", (event) => {
    if (event.target !== $("model-dialog")) return;
    const r = $("model-dialog").getBoundingClientRect();
    if (event.clientX < r.left || event.clientX > r.right || event.clientY < r.top || event.clientY > r.bottom) $("model-dialog").close();
  });

  async function boot() {
    renderIllustration(readMeasurements().values);
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 8000);
    try {
      const response = await fetch("/api/model", { signal: controller.signal });
      if (!response.ok) throw new Error("Model bilgisi alınamadı.");
      const info = await response.json();
      $("model-parameters").textContent = new Intl.NumberFormat("tr-TR").format(info.parameter_count);
      $("model-version").textContent = info.model_version;
      setStatus(true);
    } catch { setStatus(false); }
    finally { clearTimeout(timeout); }
    // Kullanıcı bağlantı sırasında ölçümleri değiştirdiyse en son değerleri kullan.
    runPrediction();
  }
  boot();
})();
